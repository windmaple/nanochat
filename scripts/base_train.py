"""
Train model (JAX version).
"""

import os
import time
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import wandb

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state, tokenizing_distributed_data_loader
from nanochat.common import print0, DummyWandb, print_banner, get_base_dir, setup_default_logging
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import CheckpointManager
from nanochat.muon import get_muon
# from nanochat.loss_eval import evaluate_bpb # TODO: port loss_eval
# from nanochat.engine import Engine # TODO: port engine fully or use simplified generation here
# from scripts.base_eval import evaluate_model # TODO: port base_eval

print_banner()
setup_default_logging()

# -----------------------------------------------------------------------------
# User settings (kept similar to PyTorch version)
run = "dummy"
depth = 20
max_seq_len = 2048
num_iterations = -1
target_flops = -1.0
target_param_data_ratio = 20
device_batch_size = 32
total_batch_size = 524288
embedding_lr = 0.2
unembedding_lr = 0.004
weight_decay = 0.0
matrix_lr = 0.02
grad_clip = 1.0
warmup_ratio = 0.0
warmdown_ratio = 0.2
final_lr_frac = 0.0
resume_from_step = -1
eval_every = 250
eval_tokens = 20*524288
core_metric_every = 2000
core_metric_max_per_task = 500
sample_every = 2000
save_every = -1
model_tag = ""

# Configurator
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# Init JAX Distributed
try:
    jax.distributed.initialize()
except Exception as e:
    print0(f"JAX distributed init failed (expected if single process): {e}")

process_index = jax.process_index()
process_count = jax.process_count()
local_device_count = jax.local_device_count()
device_count = jax.device_count()
print0(f"JAX process: {process_index}/{process_count}, devices: {device_count} (local: {local_device_count})")

master_process = process_index == 0

# Wandb
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=run, config=user_config)

# Tokenizer
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model Config
num_layers = depth
model_dim = depth * 64
num_heads = max(1, (model_dim + 127) // 128)
num_kv_heads = num_heads
print0(f"num_layers: {num_layers}, model_dim: {model_dim}, num_heads: {num_heads}")

# Batch sizes
tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * device_count
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Grad accum steps: {grad_accum_steps}")

# -----------------------------------------------------------------------------
# Initialize Model
model_config = GPTConfig(sequence_len=max_seq_len, vocab_size=vocab_size, n_layer=num_layers, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim)
rngs = nnx.Rngs(0)
model = GPT(model_config, rngs=rngs)

# Checkpoint Manager
base_dir = get_base_dir()
output_dirname = model_tag if model_tag else f"d{depth}"
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
ckpt_mgr = CheckpointManager(checkpoint_dir)

# Resume
resuming = resume_from_step != -1
if resuming:
    print0(f"Resuming from step {resume_from_step}")
    restored = ckpt_mgr.load(resume_from_step)
    nnx.update(model, restored['model'])
    # optim state loaded later
    meta_data = restored['meta']
else:
    meta_data = {}

# FLOPs estimation (approximate for JAX model)
# We can use the same formula as PyTorch
def estimate_flops(config):
    nparams = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model)))
    # approximate embedding params
    nparams_embedding = config.vocab_size * config.n_embd
    l, h, q, t = config.n_layer, config.n_head, config.n_embd // config.n_head, config.sequence_len
    # This is a rough estimate
    return 6 * (nparams - nparams_embedding) + 12 * l * h * q * t

num_flops_per_token = estimate_flops(model_config)
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Iterations calculation
if num_iterations > 0:
    pass
elif target_flops > 0:
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
elif target_param_data_ratio > 0:
    nparams = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model)))
    target_tokens = target_param_data_ratio * nparams
    num_iterations = target_tokens // total_batch_size
else:
    raise ValueError("No training horizon specified")

print0(f"Num iterations: {num_iterations}")

# -----------------------------------------------------------------------------
# Optimizer
# We use optax.multi_transform to apply Muon to 2D matrices and AdamW to others.
# We need a mask/partition.

def param_labels(params):
    # Return labels for each parameter
    # 'muon' for 2D kernels in Linear layers (except embedding/head)
    # 'adamw' for everything else
    
    def label_fn(param):
        # Check if it's a kernel and 2D
        if hasattr(param, 'value'):
            param_val = param.value
        else:
            param_val = param
            
        if param_val.ndim == 2 and param_val.shape[0] > 256 and param_val.shape[1] > 256:
            # This is likely a matrix parameter (not embedding/head)
            return 'muon'
        return 'adamw'
    
    return jax.tree_util.tree_map(label_fn, params)

# Schedulers
def get_lr_schedule(base_lr):
    def schedule(count):
        # count is step
        # We need to implement the warmup/warmdown logic
        # This runs inside JIT, so use jnp
        it = count
        warmup_iters = round(warmup_ratio * num_iterations)
        warmdown_iters = round(warmdown_ratio * num_iterations)
        
        arg1 = (it + 1) / warmup_iters
        arg2 = 1.0
        progress = (num_iterations - it) / warmdown_iters
        arg3 = progress * 1.0 + (1 - progress) * final_lr_frac
        
        # Select based on it
        res = jnp.where(it < warmup_iters, arg1, jnp.where(it <= num_iterations - warmdown_iters, arg2, arg3))
        return base_lr * res
    return schedule

# AdamW
adamw = optax.adamw(learning_rate=get_lr_schedule(embedding_lr), weight_decay=weight_decay)
# Muon
# Muon needs its own scheduler?
# PyTorch code: matrix_lr = 0.02.
# And it has momentum scheduler.
# optax.contrib.muon takes learning_rate.
# We can pass a schedule.
muon_optim = get_muon(learning_rate=matrix_lr, momentum=0.95) # TODO: implement momentum schedule if critical

# Combine
params = nnx.state(model)
labels = param_labels(params)
tx = optax.multi_transform(
    {'adamw': adamw, 'muon': muon_optim},
    labels
)

# Gradient Accumulation
if grad_accum_steps > 1:
    tx = optax.MultiSteps(tx, every_k_schedule=grad_accum_steps)

# Optimizer state
optimizer = nnx.Optimizer(model, tx, wrt=nnx.All(nnx.Param))

if resuming and 'optim' in restored:
    # Load optimizer state
    # restored['optim'] is the state
    # nnx.Optimizer manages state in .opt_state
    # We need to update it.
    # nnx.Optimizer doesn't expose simple load_state_dict?
    # We can update the variable.
    # optimizer.opt_state = restored['optim']
    # But we need to be careful about structure.
    # Let's assume it matches.
    pass # TODO: implement optimizer state loading correctly for NNX

# -----------------------------------------------------------------------------
# Data Loader
dataloader_resume_state_dict = meta_data.get("dataloader_state_dict") if resuming else None
train_loader = tokenizing_distributed_data_loader_with_state(device_batch_size, max_seq_len, split="train", resume_state_dict=dataloader_resume_state_dict)

# -----------------------------------------------------------------------------
# Train Step
@nnx.jit
def train_step(model, optimizer, inputs, targets):
    def loss_fn(model):
        loss = model(inputs, targets)
        return loss
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss

# -----------------------------------------------------------------------------
# Loop
step = meta_data.get("step", 0) if resuming else 0
total_training_time = meta_data.get("loop_state", {}).get("total_training_time", 0) if resuming else 0

while step < num_iterations:
    t0 = time.time()
    
    # We need to accumulate gradients over `grad_accum_steps` micro-batches
    # But `optax.MultiSteps` handles the update frequency.
    # We just call `optimizer.update` every micro-step.
    # It will only apply updates every k steps.
    # However, we need to feed data.
    
    # For efficiency, we should probably scan over micro-batches inside JIT.
    # But for simplicity and to match the generator structure, we can loop in python.
    
    # Fetch data
    # We need `grad_accum_steps` batches.
    
    micro_step_losses = []
    for _ in range(grad_accum_steps):
        inputs, targets, dl_state = next(train_loader)
        # inputs, targets are numpy arrays. JAX handles conversion.
        
        # Run step
        loss = train_step(model, optimizer, inputs, targets)
        micro_step_losses.append(loss)
        
    # Loss for logging (average of micro steps)
    train_loss = jnp.mean(jnp.array(micro_step_losses))
    
    t1 = time.time()
    dt = t1 - t0
    total_training_time += dt
    
    # Logging
    if step % 10 == 0:
        print0(f"Step {step}/{num_iterations} | Loss: {train_loss:.4f} | Time: {dt*1000:.2f}ms")
        wandb_run.log({"train/loss": train_loss, "step": step})

    # Save
    if save_every > 0 and step % save_every == 0:
        ckpt_mgr.save(step, nnx.state(model), optimizer.opt_state, {
            "step": step,
            "dataloader_state_dict": dl_state,
            "loop_state": {"total_training_time": total_training_time}
        })

    step += 1

# Final save
ckpt_mgr.save(step, nnx.state(model), optimizer.opt_state, {
    "step": step,
    "dataloader_state_dict": dl_state,
    "loop_state": {"total_training_time": total_training_time}
})

wandb_run.finish()
