"""
Utilities for saving and loading model/optim/state checkpoints using Orbax (JAX).
"""
import os
import json
import logging
import jax
import orbax.checkpoint as ocp
from flax import nnx

from nanochat.common import get_base_dir, setup_default_logging
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    # In JAX/JIT, we might not have explicit rank env vars if using jax.distributed
    # But we can check jax.process_index()
    if jax.process_index() == 0:
        logger.info(message)

class CheckpointManager:
    def __init__(self, directory, max_to_keep=5):
        self.directory = directory
        # Use StandardCheckpointer which handles PyTrees (model, optim, meta)
        self.mgr = ocp.CheckpointManager(
            os.path.abspath(directory),
            ocp.StandardCheckpointer(),
            options=ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
        )

    def save(self, step, model_state, optimizer_state=None, meta_data=None):
        # Bundle everything into a single PyTree
        # meta_data is a dict, model_state is a PyTree, optimizer_state is a PyTree
        save_args = {'model': model_state}
        if optimizer_state is not None:
            save_args['optim'] = optimizer_state
        if meta_data is not None:
            save_args['meta'] = meta_data
            
        # We only save on rank 0 usually, but Orbax handles distributed saving.
        # We should call save on all processes, Orbax coordinates.
        self.mgr.save(step, save_args)
        # Wait for save to complete? Orbax usually blocks or handles it.
        
    def load(self, step, target=None):
        # target is an optional PyTree with the same structure as saved data
        # If target is provided, Orbax restores into it (useful for sharding)
        return self.mgr.restore(step, items=target)

    def latest_step(self):
        return self.mgr.latest_step()

# Wrapper functions to maintain some compatibility or ease of use
def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    # This function is deprecated in favor of using CheckpointManager directly in the loop
    # But for compatibility with existing structure, we can instantiate a manager.
    # Warning: Instantiating manager every time might be slow or problematic.
    # It's better to update the training loop.
    # I will implement this as a helper that creates a manager.
    mgr = CheckpointManager(checkpoint_dir)
    mgr.save(step, model_data, optimizer_data, meta_data)

def load_checkpoint(checkpoint_dir, step, device=None, load_optimizer=False, rank=0):
    # device argument is ignored in JAX (handled by sharding)
    mgr = CheckpointManager(checkpoint_dir)
    restored = mgr.load(step)
    model_data = restored.get('model')
    optimizer_data = restored.get('optim') if load_optimizer else None
    meta_data = restored.get('meta')
    return model_data, optimizer_data, meta_data

def build_model(checkpoint_dir, step, device=None, phase="eval"):
    """
    Builds a model from a checkpoint.
    """
    mgr = CheckpointManager(checkpoint_dir)
    if step is None:
        step = mgr.latest_step()
        if step is None:
             raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    restored = mgr.load(step)
    meta_data = restored['meta']
    model_config_kwargs = meta_data["model_config"]
    log0(f"Building model with config: {model_config_kwargs}")
    
    config = GPTConfig(**model_config_kwargs)
    rngs = nnx.Rngs(0)
    model = GPT(config, rngs=rngs)
    
    # In NNX, we update the model state
    # restored['model'] should be the state
    # We need to ensure the structure matches.
    # If we saved nnx.state(model), we can load it back.
    
    # However, we need to be careful about abstract values vs concrete.
    # When we init GPT, it has concrete weights (random).
    # We update them.
    
    nnx.update(model, restored['model'])
    
    tokenizer = get_tokenizer()
    return model, tokenizer, meta_data

def load_model(source, device=None, phase="eval", model_tag=None, step=None):
    model_dir = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    
    if model_tag is None:
        # Find largest model (directory with latest step?)
        # This logic is a bit different with Orbax structure.
        # Orbax creates subdirs in the checkpoint_dir.
        # But here `checkpoints_dir` contains `model_tag` directories.
        # We need to find the `model_tag` directory.
        if os.path.exists(checkpoints_dir):
            tags = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d))]
            if not tags:
                raise FileNotFoundError(f"No model tags found in {checkpoints_dir}")
            # Sort by modification time or name
            tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
            model_tag = tags[0]
        else:
            raise FileNotFoundError(f"Checkpoints dir {checkpoints_dir} not found")
            
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    return build_model(checkpoint_dir, step, device, phase)
