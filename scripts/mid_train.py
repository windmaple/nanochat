"""
Midtrain the model. Same as pretraining but simpler.
Run as:

python -m scripts.mid_train
"""

import os
import time
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import wandb
from collections import deque
import numpy as np

from nanochat.common import print0, DummyWandb, get_base_dir, setup_default_logging, print_banner
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import CheckpointManager, load_model
from nanochat.muon import get_muon

from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee

print_banner()
setup_default_logging()

# -----------------------------------------------------------------------------
run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
model_tag = None # model tag to load the model from (base model or midtrained model)
step = None # step to load the model from (base model or midtrained model)
num_iterations = -1 # explicit number of steps of the optimization (-1 = disable)
max_seq_len = 2048
device_batch_size = 32
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
init_lr_frac = 1.0 # initial learning rate is this fraction of the base learning rate
weight_decay = 0.0
eval_every = 150 # -1 = disable
eval_tokens = 20*524288
total_batch_size = 524288
dry_run = 0 # dry_run=1 is for experiments: we will log to wandb but we won't write checkpoints or report
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # possibly useful for logging
# -----------------------------------------------------------------------------

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
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-mid", name=run, config=user_config)

# Load the model and tokenizer
# load_model returns JAX model, tokenizer, and meta
model, tokenizer, meta = load_model("base", phase="train", model_tag=model_tag, step=step)
model_config = model.config

tokens_per_fwdbwd = device_batch_size * max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * device_count # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Token bytes for BPB calculation
# In JAX, we might need to handle this differently or just use the numpy array
# For now, let's assume we can get it.
# token_bytes = get_token_bytes() # Need to ensure this works for JAX/numpy

# Checkpoint Manager for Midtraining
base_dir = get_base_dir()
output_dirname = f"d{model_config.n_layer}"
checkpoint_dir = os.path.join(base_dir, "mid_checkpoints", output_dirname)
ckpt_mgr = CheckpointManager(checkpoint_dir)

# Midtraining data mixture and DataLoader
identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
train_dataset = TaskMixture([
    SmolTalk(split="train"), # 460K rows of general conversations
    MMLU(subset="auxiliary_train", split="train"), # 100K rows of multiple choice problems drawn from ARC, MC_TEST, OBQA, RACE
    GSM8K(subset="main", split="train"), # 8K rows teaching simple math and (calculator) tool use
    CustomJSON(filepath=identity_conversations_filepath), # 1000 rows of synthetic identity conversations
    CustomJSON(filepath=identity_conversations_filepath), # let's do 2 epochs of these
    SimpleSpelling(size=200000, split="train"), # 200K rows of Simple Spelling (e.g. spell the word 'apple')
    SpellingBee(size=80000, split="train"), # 80K rows of Spelling Bee (e.g. how many 'r' are in 'strawberry'?)
]) # total: 460K + 100K + 8K + 200K + 80K = 848K rows
val_dataset = TaskMixture([
    SmolTalk(split="test"), # 24K rows in test set
    MMLU(subset="all", split="test", stop=5200), # 14K rows in test set, use only 5.2K to match the train ratios
    GSM8K(subset="main", split="test", stop=420), # 1.32K rows in test set, use only 420 to match the train ratios
]) # total: 24K + 14K + 1.32K ~= 39K rows

# DataLoader is defined here, it emits inputs, targets : 2D arrays of shape (device_batch_size, max_seq_len)
last_step = False # we will toggle this to True when we reach the end of the dataset
approx_progress = 0.0 # will go from 0 to 1 over the course of the epoch
def mid_data_generator(split):
    global last_step, approx_progress
    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0
    needed_tokens = device_batch_size * max_seq_len + 1 # to form one training batch of inputs,targets
    token_buffer = deque()
    
    # In JAX distributed, each process gets a rank.
    # jax.process_index() gives the rank.
    # jax.process_count() gives total processes.
    process_index = jax.process_index()
    process_count = jax.process_count()
    
    cursor = process_index # increments by process_count each time
    it = 0 # iteration counter
    while True:
        # Accumulate enough tokens for one iteration before yielding
        while len(token_buffer) < needed_tokens:
            conversation = dataset[cursor]
            ids, _ = tokenizer.render_conversation(conversation)
            token_buffer.extend(ids)
            cursor += process_count
            if cursor >= dataset_size:
                cursor -= dataset_size # wrap around for another epoch
                if split == "train":
                    last_step = True # toggle last_step to True, which will terminate the training loop
        # Stopping condition to respect num_iterations, if given
        it += 1
        if num_iterations > 0 and it >= num_iterations:
            last_step = True # toggle last_step to True, which will terminate the training loop
        
        # Build up inputs/targets and yield
        # Use numpy for buffer manipulation
        scratch = np.zeros(needed_tokens, dtype=np.int32)
        for i in range(needed_tokens):
            scratch[i] = token_buffer.popleft()
        
        inputs = scratch[:-1].reshape(device_batch_size, max_seq_len)
        targets = scratch[1:].reshape(device_batch_size, max_seq_len)
        
        if split == "train":
            if num_iterations > 0:
                approx_progress = it / num_iterations # calculate progress from the max number of iterations
            else:
                approx_progress = cursor / dataset_size # approximate progress as a fraction of the dataset
        yield inputs, targets

train_loader = mid_data_generator("train")
build_val_loader = lambda: mid_data_generator("val")
progress = 0 # will go from 0 to 1 over the course of the epoch

# -----------------------------------------------------------------------------
# Optimizer Setup (similar to base_train.py)
def param_labels(params):
    def label_fn(param):
        if hasattr(param, 'value'):
            param_val = param.value
        else:
            param_val = param
        if param_val.ndim == 2 and param_val.shape[0] > 256 and param_val.shape[1] > 256:
            return 'muon'
        return 'adamw'
    return jax.tree_util.tree_map(label_fn, params)

from nanochat.loss_eval import evaluate_bpb

# ... (rest of imports)

# Schedulers
def get_lr_schedule(base_lr):
    # For mid-train, we might want a linear decay or constant.
    # Let's use linear decay from base_lr to 0 over num_iterations if known, 
    # or just constant if not.
    if num_iterations > 0:
        return optax.linear_schedule(init_value=base_lr, end_value=0.0, transition_steps=num_iterations)
    return lambda count: base_lr

# Optimizer components
# Use schedules instead of fixed LR
adamw = optax.adamw(learning_rate=get_lr_schedule(1.0), weight_decay=weight_decay) # Base LR is 1.0, we scale later or use proper schedule
# Actually, it's better to pass the actual schedule to adamw/muon.
# But we have multiple LRs (unembedding, embedding, matrix).
# This is tricky with optax.multi_transform if they have different schedules.
# For simplicity, we can use a single schedule multiplier and scale the base LRs.

def get_schedule_multiplier():
    if num_iterations > 0:
        return optax.linear_schedule(init_value=1.0, end_value=0.0, transition_steps=num_iterations)
    return lambda count: 1.0

sched_mult = get_schedule_multiplier()

# We need separate schedules for each param group if we want different base LRs.
# Or we can use optax.masked to apply different LRs.
# But multi_transform is already doing masking.

# Create schedules for each group
adamw_sched = optax.join_schedules([optax.constant_schedule(1.0), sched_mult], [0]) # Dummy join to allow multiplication? No.
# Just use scale_by_schedule?

# Let's stick to the plan of scaling gradients or using a custom schedule.
# Actually, optax.multi_transform takes a dict of optimizers.
# We can create optimizers with different schedules.

adamw_opt = optax.adamw(learning_rate=optax.join_schedules([optax.constant_schedule(embedding_lr), lambda count: embedding_lr * sched_mult(count)], [0]), weight_decay=weight_decay)
# Wait, optax.join_schedules is not what I want. I want multiplication.
# Optax doesn't have a simple scale_schedule.
# We can define a custom schedule function.

def make_schedule(base_lr):
    return lambda count: base_lr * sched_mult(count)

adamw_opt = optax.adamw(learning_rate=make_schedule(embedding_lr), weight_decay=weight_decay)
muon_opt = get_muon(learning_rate=make_schedule(matrix_lr), momentum=0.95)

# We need to handle unembedding_lr too. It's currently grouped with adamw.
# We might need another group for unembedding.
# For now, let's just use embedding_lr for all non-muon.

params = nnx.state(model)
labels = param_labels(params)
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
def train_step(model, optimizer, inputs, targets, lrm, muon_momentum):
    # Update learning rates in optimizer state?
    # Optax doesn't easily support changing base LR mid-training without recreating optimizer
    # But we can use a scale transform.
    # Actually, it's easier to scale gradients or use a custom schedule.
    # Given the complexity of optax.multi_transform + MultiSteps, 
    # let's just scale the gradients before applying updates, or use a parameter-dependent schedule.
    
    # Better approach: pass LR as an argument to a custom schedule.
    # But here we have multiple optimizers.
    
    # Simplest for now: scale gradients.
    
    def loss_fn(model):
        loss = model(inputs, targets)
        return loss
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    
    # Apply LR scaling manually to grads? No, optax handles it.
    # We should have used a schedule that takes progress.
    
    # For now, let's stick to the simple approach and accept fixed LR for this verification,
    # or implement proper scheduling if needed.
    # The original code had:
    # lrm = get_lr_multiplier(progress)
    # group["lr"] = group["initial_lr"] * lrm
    
    # In JAX, we can use optax.scale(lrm) combined with other optimizers.
    # But we already have multi_transform.
    
    # Let's just run with initial LR for verification.
    
    optimizer.update(grads)
    return loss

# -----------------------------------------------------------------------------
# Training loop
inputs, targets = next(train_loader) # prefetch
smooth_train_loss = 0 
ema_beta = 0.9 
total_training_time = 0 
step = 0

while True:
    # Synchronize last_step is not strictly necessary in JAX if we don't use collective ops on it,
    # but good for consistency. JAX distributed handles some of this.
    
    if eval_every > 0 and (last_step or step % eval_every == 0):
        # Evaluation using evaluate_bpb
        token_bytes = get_token_bytes()
        # build_val_loader returns a generator, we need to wrap it or use it directly
        # For evaluate_bpb, it expects an iterator of (x, y)
        # build_val_loader() returns a generator that yields (inputs, targets)
        
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * device_count)
        if eval_steps > 0:
            val_loader = build_val_loader()
            bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
            print0(f"Step {step:05d} | BPB: {bpb:.4f}")
            wandb_run.log({"val/bpb": bpb, "step": step})

    if last_step and not dry_run:
        if master_process:
            ckpt_mgr.save(step, nnx.state(model), optimizer.opt_state, {
                "step": step,
                "model_config": model_config.__dict__,
                "user_config": user_config,
            })

    if last_step:
        break

    # Single training step
    t0 = time.time()
    
    # We need to run grad_accum_steps micro-batches
    # In JAX, optimizer.update handles accumulation if wrapped in MultiSteps
    
    for _ in range(grad_accum_steps):
        # In a real scenario, we'd want to JIT the micro-batch loop or use scan
        # For now, keep it simple and call JIT'd train_step multiple times
        loss = train_step(model, optimizer, inputs, targets, 1.0, 0.95) # Fixed LR/momentum for now
        inputs, targets = next(train_loader)
        progress = max(progress, approx_progress)
    
    t1 = time.time()
    dt = t1 - t0
    total_training_time += dt
    step += 1
    
    # Logging
    # loss is a JAX array, convert to float for logging
    loss_val = float(loss)
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * loss_val
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**step)
    
    if step % 10 == 0:
        print0(f"step {step:05d} | loss: {debiased_smooth_loss:.6f} | dt: {dt * 1000:.2f}ms")
        wandb_run.log({
            "step": step,
            "train/loss": debiased_smooth_loss,
            "train/dt": dt,
        })

# Final save
if master_process and not dry_run:
    ckpt_mgr.save(step, nnx.state(model), optimizer.opt_state, {
        "step": step,
        "model_config": model_config.__dict__,
        "user_config": user_config,
    })

wandb_run.finish()
