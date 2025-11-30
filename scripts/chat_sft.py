"""
Finetune a base model to be a chat model.
Run on one GPU e.g. for debugging:

python -m scripts.chat_sft
"""

import os
import time
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import wandb
import numpy as np
import traceback
import sys

from nanochat.common import print0, DummyWandb, get_base_dir, setup_default_logging
from nanochat.checkpoint_manager import load_model, CheckpointManager
from nanochat.engine import Engine
from nanochat.muon import get_muon

from tasks.common import TaskMixture
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee

# -----------------------------------------------------------------------------
# SFT Hyperparameters
run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
# input model options
source = "mid" # base|mid , which checkpoint to load the model from (base model or midtrained model)
model_tag = None # model tag to load the model from (base model or midtrained model)
load_step = None # step to load the model from (base model or midtrained model)
# compute/precision
device_batch_size = 4 # max to avoid OOM
# optimization
num_epochs = 1
num_iterations = -1 # override number of iterations (-1 = disable, use num_epochs to derive it)
target_examples_per_step = 32
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.02
# evaluation and logging there of
eval_every = 100
eval_steps = 100
eval_metrics_every = 200
eval_metrics_max_problems = 1024
# now allow CLI to override the settings via the configurator lol
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # possibly useful for logging
# -----------------------------------------------------------------------------

def main():
    setup_default_logging()

    # Init JAX Distributed
    try:
        jax.distributed.initialize()
    except Exception as e:
        print0(f"JAX distributed init failed (expected if single process): {e}")

    process_index = jax.process_index()
    process_count = jax.process_count()
    device_count = jax.device_count()
    print0(f"JAX process: {process_index}/{process_count}, devices: {device_count}")
    master_process = process_index == 0

    # wandb logging init
    use_dummy_wandb = run == "dummy" or not master_process
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-sft", name=run, config=user_config, save_code=True)

    # Load the model and tokenizer
    model, tokenizer, meta = load_model(source, phase="train", model_tag=model_tag, step=load_step, allow_missing=True)
    model_config = model.config
    engine = Engine(model, tokenizer) # will be used for inline model evaluation only

    # -----------------------------------------------------------------------------
    # Task data mixture we'll train on
    identity_conversations_filepath = os.path.join(get_base_dir(), "identity_conversations.jsonl")
    train_ds = TaskMixture([
        ARC(subset="ARC-Easy", split="train"), # 2.3K rows
        ARC(subset="ARC-Challenge", split="train"), # 1.1K rows
        GSM8K(subset="main", split="train"), # 8K rows
        SmolTalk(split="train", stop=10_000), # 10K rows of smoltalk
        CustomJSON(filepath=identity_conversations_filepath), # 1K rows of synthetic identity conversations
        SimpleSpelling(size=300, split="train"), # 300 rows of Simple Spelling (e.g. spell the word 'apple')
        SpellingBee(size=300, split="train"), # 300 rows of Spelling Bee (e.g. how many 'r' are in 'strawberry'?)
    ]) # 2.3K + 1.1K + 8K + 10K + 1K + 0.3K + 0.3K = 23K rows
    val_ds = SmolTalk(split="test") # general conversations, 24K rows (though we don't actually use all of it)

    # -----------------------------------------------------------------------------
    # DataLoader

    def sft_data_generator(dataset, batch_size):
        pad_token_id = tokenizer.encode_special("<|assistant_end|>") # use <|assistant_end|> as the pad token is ok, these positions are masked in the loss
        
        def collate_and_yield(batch):
            nrows = len(batch)
            ncols = max(len(ids) for ids, mask in batch) - 1 # seq of n creates inputs/targets of n-1
            inputs = np.full((nrows, ncols), pad_token_id, dtype=np.int32)
            targets = np.full((nrows, ncols), -1, dtype=np.int32) # -1 is ignore index
            for i, (ids, mask) in enumerate(batch):
                n = len(ids)
                inputs[i, :n-1] = ids[:-1]
                # mask[1:] omits the mask for the BOS token, which is never a target atm so it's ok
                row_targets = np.array(ids[1:], dtype=np.int32)
                mask_np = np.array(mask[1:], dtype=np.int32)
                row_targets[mask_np == 0] = -1 # mask out targets where mask is 0
                targets[i, :n-1] = row_targets
            return inputs, targets

        # iterates over the dataset in epochs, tokenizes
        batch = []
        while True:
            for i in range(process_index, len(dataset), process_count):
                doc = dataset[i]
                ids, mask = tokenizer.render_conversation(doc)
                batch.append((ids, mask))
                if len(batch) == batch_size:
                    yield collate_and_yield(batch)
                    batch = []

    examples_per_step = device_batch_size * device_count
    assert target_examples_per_step % examples_per_step == 0, "Target examples per step must be divisible by examples per step"
    grad_accum_steps = target_examples_per_step // examples_per_step

    # Use a local num_iterations to avoid modifying the global one
    local_num_iterations = num_iterations
    if local_num_iterations == -1:
        # derive num_iterations from num_epochs and the size of the dataset
        assert num_epochs > 0, "num_epochs must be positive if num_iterations is -1"
        local_num_iterations = (len(train_ds) // target_examples_per_step) * num_epochs
    train_loader = sft_data_generator(train_ds, batch_size=device_batch_size)
    build_val_loader = lambda: sft_data_generator(val_ds, batch_size=device_batch_size)

    # -----------------------------------------------------------------------------
    # Optimizer Setup

    def get_schedule_multiplier():
        if local_num_iterations > 0:
            # Linear decay from 1.0 to 0.0
            return optax.linear_schedule(init_value=1.0, end_value=0.0, transition_steps=local_num_iterations)
        return lambda count: 1.0

    sched_mult = get_schedule_multiplier()

    def make_schedule(base_lr):
        return lambda count: base_lr * init_lr_frac * sched_mult(count)

    adamw_opt = optax.adamw(learning_rate=make_schedule(embedding_lr), weight_decay=weight_decay)
    muon_opt = get_muon(learning_rate=make_schedule(matrix_lr), momentum=0.95)

    def param_labels(model):
        params, _ = nnx.split(model, nnx.Param)
        def label_fn(path, leaf):
            for p in path:
                if 'kernel' in str(p):
                    return 'muon'
            return 'adamw'
        return jax.tree_util.tree_map_with_path(
            label_fn, params, is_leaf=lambda x: isinstance(x, nnx.Variable)
        )

    labels = param_labels(model)
    tx = optax.multi_transform(
        {'adamw': adamw_opt, 'muon': muon_opt},
        labels
    )

    if grad_accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=grad_accum_steps)

    optimizer = nnx.Optimizer(model, tx, wrt=nnx.All(nnx.Param))

    # -----------------------------------------------------------------------------
    # Training Step
    @nnx.jit
    def train_step(model, optimizer, inputs, targets):
        def loss_fn(model):
            loss = model(inputs, targets)
            return loss
        
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss

    # -----------------------------------------------------------------------------
    # Training loop
    train_iter = iter(train_loader)

    # Checkpoint Manager
    base_dir = get_base_dir()
    output_dirname = f"d{model_config.n_layer}"
    checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_dirname)
    ckpt_mgr = CheckpointManager(checkpoint_dir)

    for training_step in range(local_num_iterations):
        last_step = training_step == local_num_iterations - 1

        # evaluate the validation loss
        if last_step or training_step % eval_every == 0:
            val_iter = iter(build_val_loader())
            losses = []
            for _ in range(eval_steps):
                val_inputs, val_targets = next(val_iter)
                # In JAX, we can just call the model. 
                # For evaluation, we might want to JIT this too.
                @nnx.jit
                def eval_step(model, inputs, targets):
                    return model(inputs, targets)
                
                loss = eval_step(model, val_inputs, val_targets)
                losses.append(float(loss))
            val_loss = sum(losses) / len(losses)
            wandb_run.log({
                "step": training_step,
                "val_loss": val_loss,
            })

        if last_step:
            break

        # evaluate the gradient
        # In JAX, we handle accumulation in optimizer.update via MultiSteps
        # But we need to feed micro-batches.
        
        total_loss = 0
        for micro_step in range(grad_accum_steps):
            train_inputs, train_targets = next(train_iter)
            loss = train_step(model, optimizer, train_inputs, train_targets)
            total_loss += float(loss)
        
        avg_loss = total_loss / grad_accum_steps
        
        # logging
        wandb_run.log({
            "step": training_step,
            "train_loss": avg_loss,
        })


    # Save the model at the end of the run
    if master_process:
        ckpt_mgr.save(training_step, nnx.state(model), optimizer.opt_state, {
            "step": training_step,
            "model_config": model_config.__dict__,
            "user_config": user_config,
        })
        print0(f"✅ Saved model checkpoint to {checkpoint_dir}")

    wandb_run.finish()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"!!! SFT script failed with an exception !!!")
        print(traceback.format_exc())
        print(f"!!! SFT script failed with an exception !!!", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
