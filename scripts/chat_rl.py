"""
Reinforcement learning on GSM8K via "GRPO".

I put GRPO in quotes because we actually end up with something a lot
simpler and more similar to just REINFORCE:

1) Delete trust region, so there is no KL regularization to a reference model
2) We are on policy, so there's no need for PPO ratio+clip.
3) We use GAPO style normalization that is token-level, not sequence-level.
4) Instead of z-score normalization (r - mu)/sigma, only use (r - mu) as the advantage.

1 GPU:
python -m scripts.chat_rl

8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=default
"""

import os
import itertools
import re
import wandb
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np

from nanochat.common import print0, get_base_dir, DummyWandb, setup_default_logging, print_banner
from nanochat.checkpoint_manager import CheckpointManager, load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K
from nanochat.muon import get_muon

print_banner()
setup_default_logging()

# RL hyperparameters
run = "dummy" # wandb run name
source = "sft" # mid|sft
device_batch_size = 1 # no forward pass will go above this to not OOM
examples_per_step = 16 # in total and across all ranks (note: examples, not samples/completions!)
num_samples = 16 # number of samples per example (/question)
max_new_tokens = 256
temperature = 1.0
top_k = 50 # TODO: try None?
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.05
num_epochs = 1 # how many epochs of gsm8k to train on
save_every = 60 # every how many steps to save the model
eval_every = 60 # every how many steps to evaluate the model for val pass@k
eval_examples = 400 # number of examples used for evaluating pass@k
# now allow CLI to override the settings via the configurator lol
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # will be useful for logging
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
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-rl", name=run, config=user_config)

# Init model and tokenizer
model, tokenizer, meta = load_model(source, phase="eval", allow_missing=True)
model_config = model.config
engine = Engine(model, tokenizer) # for sampling rollouts

# -----------------------------------------------------------------------------
# Rollout / sampling generator loop that yields batches of examples for training

train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")
num_steps = (len(train_task) // examples_per_step) * num_epochs
print0(f"Calculated number of steps: {num_steps}")

def get_batch(step_ref):
    # step_ref is a list containing the current step to allow updating from outside if needed, 
    # but here we just use it for seeding. Actually, we can just pass step.
    assistant_end = tokenizer.encode_special("<|assistant_end|>") # ok to use this token, it's only for padding and isn't used in the loss.
    rank_indices = range(process_index, len(train_task), process_count) # each rank is responsible for different examples in the training data
    for example_idx in itertools.cycle(rank_indices):

        # First get the full conversation of both user and assistant messages
        conversation = train_task[example_idx]

        # Tokenize the conversation, deleting the last Assistant message and priming the Assistant for a completion instead
        # (i.e. keep the <|assistant_start|>, but delete everything after it)
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # Generate num_samples samples using batched generation, use loop to avoid OOMs
        # In JAX, we don't have explicit eval mode in the same way, but we can control dropout if present.
        # GPT model doesn't seem to have dropout currently.
        generated_token_sequences = []
        masks = []
        num_sampling_steps = num_samples // device_batch_size # go sequentially to prevent OOMs
        for sampling_step in range(num_sampling_steps):
            # Seed generation
            seed = hash((step_ref[0], example_idx, sampling_step)) & 0x7FFFFFFF # positive half of int32
            generated_token_sequences_batch, masks_batch = engine.generate_batch(
                tokens,
                num_samples=device_batch_size,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=seed, # must make sure to change the seed for each sampling step
            )
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)

        # Calculate the rewards for each sample
        rewards = []
        for sample_tokens in generated_token_sequences:
            # Get just the generated tokens (after the prompt)
            generated_tokens = sample_tokens[prefix_length:]
            # Decode the generated response
            generated_text = tokenizer.decode(generated_tokens)
            # Calculate the reward
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)

        # Pad the sequences so that their lengths (in time) match
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_generated_token_sequences = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        
        # Convert to numpy arrays
        ids = np.array(padded_generated_token_sequences, dtype=np.int32)
        mask_ids = np.array(padded_masks, dtype=np.int32)
        
        # Generate autoregressive inputs and targets
        inputs = ids[:, :-1]
        targets = ids[:, 1:].copy()
        targets[mask_ids[:, 1:] == 0] = -1 # -1 is the ignore index
        
        rewards_np = np.array(rewards, dtype=np.float32)
        # Calculate the advantages by simply subtracting the mean (instead of z-score (x-mu)/sigma)
        mu = rewards_np.mean()
        advantages = rewards_np - mu
        
        # yield inputs/targets as (B, T) of ids and rewards as (B,) of floats
        yield generated_token_sequences, inputs, targets, rewards_np, advantages

# -----------------------------------------------------------------------------
# Simple evaluation loop for GSM8K pass@k
def run_gsm8k_eval(task, tokenizer, engine,
    max_examples=None,
    num_samples=1,
    max_completion_tokens=256,
    temperature=0.0,
    top_k=50
):
    """
    Evaluates GSM8K task and returns a list of records of evaluation outcomes.
    In a distributed setting, all ranks cooperate but this function will NOT
    do the reduction across ranks. This is the responsibility of the caller.
    Because the evaluation can take a while, this function will yield records one by one.
    """
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(process_index, max_examples, process_count):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        # Generate k samples using batched generation inside the Engine
        assert num_samples <= device_batch_size # usually this is true. we can add a loop if not...
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k
        )
        # Check each sample for correctness
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({
                "is_correct": is_correct
            })
        # A bit bloated because I wanted to do more complex logging at one point.
        record = {
            "idx": idx,
            "outcomes": outcomes,
        }
        yield record

# -----------------------------------------------------------------------------
# Calculate the number of examples each rank handles to achieve the desired examples_per_step
print0(f"Total sequences per step: {examples_per_step * num_samples}") # total batch size in sequences/step
assert examples_per_step % process_count == 0, "Desired examples per step must be divisible by the number of ranks"
examples_per_rank = examples_per_step // process_count # per GPU
print0(f"Calculated examples per rank: {examples_per_rank}")

# Optimizer Setup
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

def get_schedule_multiplier():
    if num_steps > 0:
        # Linear decay from 1.0 to 0.0
        return optax.linear_schedule(init_value=1.0, end_value=0.0, transition_steps=num_steps)
    return lambda count: 1.0

sched_mult = get_schedule_multiplier()

def make_schedule(base_lr):
    return lambda count: base_lr * init_lr_frac * sched_mult(count)

adamw_opt = optax.adamw(learning_rate=make_schedule(embedding_lr), weight_decay=weight_decay)
muon_opt = get_muon(learning_rate=make_schedule(matrix_lr), momentum=0.95)

params = nnx.state(model)
labels = param_labels(params)
tx = optax.multi_transform(
    {'adamw': adamw_opt, 'muon': muon_opt},
    labels
)
# No MultiSteps here, we handle accumulation manually if needed, 
# but the original code did it per example.
# Wait, the original code did:
# for example_step in range(examples_per_rank):
#     ...
#     for pass_idx in range(num_passes):
#         ...
#         loss.backward()
#     ...
# opt.step()
# This is gradient accumulation over `examples_per_rank * num_passes` micro-batches.
# So we should use MultiSteps or manual accumulation.
# Total micro-batches per update = examples_per_rank * num_passes.
# num_passes = num_samples // device_batch_size.

num_passes = num_samples // device_batch_size
total_micro_batches = examples_per_rank * num_passes
if total_micro_batches > 1:
    tx = optax.MultiSteps(tx, every_k_schedule=total_micro_batches)

optimizer = nnx.Optimizer(model, tx, wrt=nnx.All(nnx.Param))

# -----------------------------------------------------------------------------
# Training Step
@nnx.jit
def train_step(model, optimizer, inputs, targets, advantages):
    # advantages is (B,)
    # inputs is (B, T)
    # targets is (B, T)
    
    def loss_fn(model):
        logits = model(inputs) # (B, T, V)
        # Calculate log_probs
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        # Get log_prob of target tokens
        # targets has -1 for ignored tokens. We need to handle this.
        
        # One-hot encoding of targets, handling -1
        # Mask for valid targets
        mask = (targets != -1)
        # Replace -1 with 0 for indexing, we will mask out later
        targets_safe = jnp.where(mask, targets, 0)
        
        # Gather log_probs
        # log_probs is (B, T, V), targets_safe is (B, T)
        # We want (B, T)
        gathered_log_probs = jnp.take_along_axis(log_probs, targets_safe[..., None], axis=-1).squeeze(-1)
        
        # Apply mask
        gathered_log_probs = gathered_log_probs * mask
        
        # PG objective: sum(log_prob * advantage) per sequence
        # advantages is (B,)
        # gathered_log_probs is (B, T)
        # We want sum over T, then multiply by advantage, then sum over B.
        
        seq_log_probs = jnp.sum(gathered_log_probs, axis=-1) # (B,)
        pg_obj = seq_log_probs * advantages # (B,)
        
        # Normalize by number of valid tokens?
        # Original code did: pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
        # Here we are inside a single micro-batch.
        # Optax MultiSteps will handle averaging over micro-batches if we return mean loss.
        # But we want sum of PG obj, then divide by total tokens later?
        # Actually, Optax MultiSteps averages the gradients.
        # So we should return the loss for this micro-batch.
        
        num_valid = jnp.sum(mask).astype(jnp.float32)
        num_valid = jnp.maximum(num_valid, 1.0)
        
        loss = -jnp.sum(pg_obj) / num_valid # Average over valid tokens in this batch
        return loss
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss

# -----------------------------------------------------------------------------
# Training loop

# Checkpoint Manager
base_dir = get_base_dir()
output_dirname = f"d{model_config.n_layer}"
checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", output_dirname)
ckpt_mgr = CheckpointManager(checkpoint_dir)

# Kick off the training loop
step_ref = [0] # Use a list to allow updating from outside
batch_iterator = get_batch(step_ref)

for step in range(num_steps):
    step_ref[0] = step

    # Evaluate the model once in a while and log to wandb
    if step % eval_every == 0:
        # JAX evaluation
        # We need to collect records from run_gsm8k_eval
        records_iter = run_gsm8k_eval(val_task, tokenizer, engine, num_samples=device_batch_size, max_examples=eval_examples, temperature=1.0)
        records = list(records_iter) # collect all records from this rank
        
        # Calculate pass@k locally
        passk = np.zeros(device_batch_size)
        for k in range(1, device_batch_size + 1):
            passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
        
        num_records = len(records)
        
        # In JAX distributed, we would need to reduce. For now, assume single process or manual reduction if needed.
        # If multi-process, we need to use jax.distributed to gather and sum.
        # Since we don't have easy all_reduce outside JIT, we skip for multi-process for now or just log local.
        
        if num_records > 0:
            passk = passk / num_records
            print_passk = [f"Pass@{k}: {passk[k - 1]:.4f}" for k in range(1, device_batch_size + 1)]
            print0(f"Step {step} | {', '.join(print_passk)}")
            log_passk = {f"pass@{k}": passk[k - 1] for k in range(1, device_batch_size + 1)}
            wandb_run.log({
                "step": step,
                **log_passk,
            })

    # Forward/Backward on rollouts over multiple examples in the dataset
    rewards_list = []
    sequence_lengths = []
    for example_step in range(examples_per_rank):
        # Get one batch corresponding to one example in the training dataset
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)
        
        # We need one more loop because we can never exceed the device_batch_size
        assert inputs_all.shape[0] % device_batch_size == 0
        num_passes = inputs_all.shape[0] // device_batch_size
        for pass_idx in range(num_passes):
            # Pluck out the batch for this pass
            b0, b1 = pass_idx * device_batch_size, (pass_idx + 1) * device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]
            
            # Run JIT'd train step
            loss = train_step(model, optimizer, inputs, targets, advantages)
            
            print0(f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | loss: {float(loss):.6f} | Average reward: {rewards.mean():.4f}")
        
        # For logging
        rewards_list.append(rewards_all.mean())
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    # A bunch of logging for how the rollouts went this step
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
    
    # JAX distributed reduction for logging
    # We can use jax.lax.pmean if we were inside JIT, but here we are outside.
    # For now, just log local mean or implement manual reduction if needed.
    # Given single process for now, it's fine. For multi-process, we need jax.distributed.
    
    print0(f"Step {step}/{num_steps} | Average reward: {mean_reward} | Average sequence length: {mean_sequence_length:.2f}")
    wandb_run.log({
        "step": step,
        "reward": mean_reward,
        "sequence_length": mean_sequence_length,
    })

    # Learning rate scheduler: simple rampdown to zero over num_steps
    # In this JAX version, we haven't implemented LR schedule yet.
    # TODO: Implement LR schedule in Optax.

    # Master process saves the model once in a while. Skip first step. Save last step.
    if master_process and ((step > 0 and step % save_every == 0) or step == num_steps - 1):
        ckpt_mgr.save(step, nnx.state(model), optimizer.opt_state, {
            "step": step,
            "model_config": model_config.__dict__,
            "user_config": user_config,
        })
        print0(f"✅ Saved model checkpoint to {checkpoint_dir}")

wandb_run.finish()
