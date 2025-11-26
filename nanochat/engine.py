"""
Engine for efficient inference of our models (JAX version).
"""

import jax
import jax.numpy as jnp
import signal
import warnings
from contextlib import contextmanager
from collections import deque
from nanochat.common import compute_init, autodetect_device_type
# from nanochat.checkpoint_manager import load_model # TODO: implement load_model for JAX
from contextlib import nullcontext 

# -----------------------------------------------------------------------------
# Calculator tool helpers (Same as before)
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)
        return None

def use_calculator(expr):
    expr = expr.replace(",", "")
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr: return None
        return eval_with_timeout(expr)
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]): return None
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns): return None
    if '.count(' not in expr: return None
    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------
class KVCache:
    """
    JAX compatible KV Cache.
    Stores keys and values as JAX arrays.
    """

    def __init__(self, batch_size, num_layers, seq_len, head_dim, n_kv_head):
        self.kv_shape = (num_layers, batch_size, n_kv_head, seq_len, head_dim)
        self.k = jnp.zeros(self.kv_shape)
        self.v = jnp.zeros(self.kv_shape)
        self.pos = 0

    def reset(self):
        self.pos = 0

    def get_pos(self):
        return self.pos

    def prefill(self, other):
        # other is another KVCache
        # We assume shapes are compatible for broadcasting or copying
        # In JAX, we can't modify in-place easily if we are inside JIT, 
        # but here we are likely outside JIT or managing state explicitly.
        
        # For simplicity, we just copy the data
        # Assuming other.k and other.v are available
        self.k = self.k.at[:, :, :, :other.pos, :].set(other.k[:, :, :, :other.pos, :])
        self.v = self.v.at[:, :, :, :other.pos, :].set(other.v[:, :, :, :other.pos, :])
        self.pos = other.pos
        
        # Handle batch expansion if needed
        if other.k.shape[1] == 1 and self.k.shape[1] > 1:
             # Broadcast copy
             # This is a bit tricky with .at[...].set
             # We can just repeat the data
             k_slice = other.k[:, :, :, :other.pos, :]
             v_slice = other.v[:, :, :, :other.pos, :]
             k_rep = jnp.repeat(k_slice, self.k.shape[1], axis=1)
             v_rep = jnp.repeat(v_slice, self.v.shape[1], axis=1)
             self.k = self.k.at[:, :, :, :other.pos, :].set(k_rep)
             self.v = self.v.at[:, :, :, :other.pos, :].set(v_rep)

    def update(self, layer_idx, k_new, v_new):
        # k_new, v_new: (B, H, T_add, D)
        # Update the cache at self.pos
        B, H, T_add, D = k_new.shape
        t0 = self.pos
        t1 = t0 + T_add
        
        # We need to update self.k and self.v
        # Since we are in JAX, we return the updated cache?
        # Or if this is a python class managing state, we update self.k/v
        
        self.k = self.k.at[layer_idx, :, :, t0:t1, :].set(k_new)
        self.v = self.v.at[layer_idx, :, :, t0:t1, :].set(v_new)
        
        # Return the full view up to t1
        k_view = self.k[layer_idx, :, :, :t1, :]
        v_view = self.v[layer_idx, :, :, :t1, :]
        
        # We only increment pos once per step (at the last layer)
        # But here we don't know if it's the last layer easily unless passed.
        # The caller (GPT model) calls update.
        # We can increment pos in the loop in GPT model or here if we track layer idx.
        
        return k_view, v_view

    def advance(self, steps):
        self.pos += steps

# -----------------------------------------------------------------------------
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    # logits: (B, V)
    if temperature == 0.0:
        return jnp.argmax(logits, axis=-1)[:, None] # (B, 1)
    
    if top_k is not None:
        # top_k in JAX
        vals, idx = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))
        # Mask out others
        min_val = jnp.min(vals, axis=-1, keepdims=True)
        logits = jnp.where(logits < min_val, -float('inf'), logits)
        
    logits = logits / temperature
    # categorical sampling
    next_token = jax.random.categorical(rng, logits, axis=-1)
    return next_token[:, None]

# -----------------------------------------------------------------------------

class RowState:
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []
        self.forced_tokens = deque()
        self.in_python_block = False
        self.python_expr_tokens = []
        self.completed = False

class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list) and isinstance(tokens[0], int)
        rng = jax.random.PRNGKey(seed)
        
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        # 1) Batch 1 prefill
        m = self.model.config
        # kv_cache_prefill = KVCache(...)
        # For simplicity in this port, let's skip complex KV cache prefill optimization 
        # and just run the model.
        # But we need KV cache for the loop.
        
        kv_cache = KVCache(
            batch_size=num_samples,
            num_layers=m.n_layer,
            seq_len=len(tokens) + (max_tokens or 1024),
            head_dim=m.n_embd // m.n_head,
            n_kv_head=m.n_kv_head
        )
        
        # Initial prefill with broadcasted tokens if num_samples > 1
        # Actually, let's just use the tokens as input for the first step
        # If num_samples > 1, we replicate the input tokens
        
        ids = jnp.array([tokens] * num_samples, dtype=jnp.int32)
        
        # Forward pass
        # We need to handle the KV cache update inside the model or manually.
        # My GPT model calls kv_cache.update.
        
        # We need to manage the RNG key
        rng, key = jax.random.split(rng)
        
        # First pass (prefill)
        logits = self.model(ids, kv_cache=kv_cache)
        # kv_cache.pos is updated? No, I need to update it.
        kv_cache.advance(ids.shape[1])
        
        logits = logits[:, -1, :]
        next_ids = sample_next_token(logits, key, temperature, top_k)
        sampled_tokens = [int(x) for x in next_ids[:, 0]] # to list of ints
        
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]
        
        num_generated = 0
        
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(state.completed for state in row_states):
                break
                
            # Process each row
            token_column = []
            token_masks = []
            for i, state in enumerate(row_states):
                is_forced = len(state.forced_tokens) > 0
                token_masks.append(0 if is_forced else 1)
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                state.current_tokens.append(next_token)
                
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                    
                # Tool logic (simplified)
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            yield token_column, token_masks
            num_generated += 1
            
            # Prepare next input
            ids = jnp.array(token_column, dtype=jnp.int32)[:, None] # (B, 1)
            
            # Forward
            rng, key = jax.random.split(rng)
            logits = self.model(ids, kv_cache=kv_cache)
            kv_cache.advance(1)
            
            logits = logits[:, -1, :]
            next_ids = sample_next_token(logits, key, temperature, top_k)
            sampled_tokens = [int(x) for x in next_ids[:, 0]]

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            if all(completed):
                break
        return results, masks

if __name__ == "__main__":
    # Test
    pass
