"""
Utilities for saving and loading model/optim/state checkpoints using Orbax (JAX).
"""
import os
import json
import logging
import jax
import numpy as np
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
    def __init__(self, directory, max_to_keep=5, old_style=False):
        self.directory = directory
        self.old_style = old_style
        if old_style:
            self.mgr = ocp.CheckpointManager(
                os.path.abspath(directory),
                ocp.StandardCheckpointer(),
                options=ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
            )
        else:
            self.mgr = ocp.CheckpointManager(
                os.path.abspath(directory),
                options=ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
            )

    def save(self, step, model_state, optimizer_state=None, meta_data=None):
        if self.old_style:
            # Fallback to old behavior if needed, but we should avoid this for new checkpoints
            save_args = {'model': model_state}
            if optimizer_state is not None:
                save_args['optim'] = optimizer_state
            if meta_data is not None:
                save_args['meta'] = meta_data
            self.mgr.save(step, save_args)
            self.mgr.wait_until_finished()
            return

        # Bundle everything into a single PyTree
        # meta_data is a dict, model_state is a PyTree, optimizer_state is a PyTree
        save_items = {
            'model': ocp.args.StandardSave(model_state),
        }
        if optimizer_state is not None:
            save_items['optim'] = ocp.args.StandardSave(optimizer_state)
        if meta_data is not None:
            if 'user_config' in meta_data:
                uc = meta_data['user_config']
                if not isinstance(uc, dict) and hasattr(uc, '__dict__'):
                    uc = vars(uc)
                meta_data['user_config'] = json.loads(json.dumps(uc, default=str))
            save_items['meta'] = ocp.args.JsonSave(meta_data)

        self.mgr.save(step, args=ocp.args.Composite(**save_items))
        # Wait for save to complete to avoid async issues on shutdown
        self.mgr.wait_until_finished()
        
    def load(self, step, items=None):
        if items is None:
            return self.mgr.restore(step)

        restore_args = {}
        if 'model' in items:
            restore_args['model'] = ocp.args.StandardRestore()
        if 'optim' in items:
            restore_args['optim'] = ocp.args.StandardRestore()
        if 'meta' in items:
            restore_args['meta'] = ocp.args.JsonRestore()

        return self.mgr.restore(step, args=ocp.args.Composite(**restore_args))

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
    try:
        restored = mgr.load(step)
    except Exception:
        mgr = CheckpointManager(checkpoint_dir, old_style=True)
        restored = mgr.load(step)
    model_data = restored.get('model')
    optimizer_data = restored.get('optim') if load_optimizer else None
    meta_data = restored.get('meta')
    return model_data, optimizer_data, meta_data

def convert_keys_to_int(d):
    if isinstance(d, dict):
        new_dict = {}
        for k, v in d.items():
            new_key = int(k) if isinstance(k, str) and k.isdigit() else k
            new_dict[new_key] = convert_keys_to_int(v)
        return new_dict
    elif isinstance(d, list):
        return [convert_keys_to_int(elem) for elem in d]
    else:
        return d

def convert_keys_to_int(d):
    if isinstance(d, dict):
        new_dict = {}
        for k, v in d.items():
            new_key = int(k) if isinstance(k, str) and k.isdigit() else k
            new_dict[new_key] = convert_keys_to_int(v)
        return new_dict
    elif isinstance(d, list):
        return [convert_keys_to_int(elem) for elem in d]
    else:
        return d

def build_model(checkpoint_dir, step, device=None, phase="eval"):
    """
    Builds a model from a checkpoint.
    """
    mgr = CheckpointManager(checkpoint_dir)
    if step is None:
        step = mgr.latest_step()
        if step is None:
             raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    try:
        # Load just the metadata first to configure the model
        restored_meta = mgr.load(step, items={'meta': True})
        meta_data = restored_meta['meta']
    except Exception as e:
        logger.warning(f"Failed to load meta with new style: {e}, retrying with old style")
        mgr = CheckpointManager(checkpoint_dir, old_style=True)
        restored_meta = mgr.load(step)
        meta_data = restored_meta['meta']

    model_config_kwargs = meta_data["model_config"]
    log0(f"Building model with config: {model_config_kwargs}")
    config = GPTConfig(**model_config_kwargs)
    rngs = nnx.Rngs(0)
    model = GPT(config, rngs=rngs)

    # Now restore the model state into the initialized model
    restored_model = None
    try:
        # Load the raw model state tree
        restored_model = mgr.load(step, items={'model': True})['model']
    except Exception as e:
        logger.warning(f"Failed to load model state with new style: {e}, retrying with old style")
        mgr = CheckpointManager(checkpoint_dir, old_style=True)
        restored_model = mgr.load(step)['model']

    # Recursively convert keys in the loaded state
    restored_model = convert_keys_to_int(restored_model)

    # Create a state object from the restored dictionary and update the model
    restored_state = nnx.State(restored_model)
    nnx.update(model, restored_state)

    tokenizer = get_tokenizer()
    return model, tokenizer, meta_data

def load_model(source, device=None, phase="eval", model_tag=None, step=None, allow_missing=False, default_config=None):
    model_dir = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    
    try:
        if model_tag is None:
            # Find largest model (directory with latest step?)
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
    except (FileNotFoundError, ValueError) as e:
        if allow_missing:
            log0(f"Warning: Could not load model from {source} ({e}). Falling back to random initialization.")
            if default_config is None:
                # Default config if none provided
                from nanochat.gpt import GPTConfig
                tokenizer = get_tokenizer()
                default_config = GPTConfig(vocab_size=tokenizer.get_vocab_size()) # Use default other params
            
            rngs = nnx.Rngs(0)
            model = GPT(default_config, rngs=rngs)
            tokenizer = get_tokenizer()
            meta = {"step": 0, "model_config": default_config.__dict__}
            return model, tokenizer, meta
        else:
            raise e
