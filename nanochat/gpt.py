"""
GPT model in JAX using Flax NNX.
"""

import math
from functools import partial
from dataclasses import dataclass
from typing import Optional, Tuple, Any

import jax
import jax.numpy as jnp
from flax import nnx

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768

def apply_rotary_emb(x, cos, sin):
    # x: (B, H, T, D)
    # cos, sin: (1, 1, T, D/2)
    # Ensure shapes match for broadcasting
    # If x is (B, H, 1, D) (decoding step), cos/sin should be (1, 1, 1, D/2) corresponding to the position
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    out = jnp.concatenate([y1, y2], axis=3)
    return out.astype(x.dtype)

def rms_norm(x, eps=1e-6):
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)

class KVCache(nnx.Module):
    def __init__(self, batch_size, num_layers, n_kv_head, head_dim, max_len):
        self.k = nnx.Variable(jnp.zeros((num_layers, batch_size, n_kv_head, max_len, head_dim)))
        self.v = nnx.Variable(jnp.zeros((num_layers, batch_size, n_kv_head, max_len, head_dim)))
        self.pos = nnx.Variable(0)
        
    def update(self, layer_idx, k_new, v_new):
        # k_new, v_new: (B, H, T, D)
        B, H, T, D = k_new.shape
        pos = self.pos.value
        
        # Update cache using dynamic_update_slice
        # We need to handle the indices correctly.
        # k shape: (num_layers, batch_size, n_kv_head, max_len, head_dim)
        # We want to update at layer_idx, all batch, all heads, from pos to pos+T, all dim
        
        # JAX dynamic_update_slice takes (operand, slice, start_indices)
        # This is a bit tricky with multi-dimensional arrays if we only want to update a slice in one dimension.
        # We can use index_update equivalent or just concatenation for now if it's easier, 
        # but dynamic_update_slice is better for JIT.
        
        # For simplicity and correctness in JAX NNX, we can do:
        k_state = self.k.value
        v_state = self.v.value
        
        # Prepare start indices for update
        # (layer_idx, 0, 0, pos, 0)
        start_indices = (layer_idx, 0, 0, pos, 0)
        
        # k_new needs to be reshaped to match the slice shape? 
        # k_new is (B, H, T, D). Cache is (L, B, H, MaxL, D).
        # Slice to update is (1, B, H, T, D).
        # The update `k_new` should directly correspond to the slice `k_state[layer_idx, :, :, pos:pos+T, :]`
        # So `k_new` itself is the update.
        
        self.k.value = jax.lax.dynamic_update_slice(k_state, k_new, start_indices)
        self.v.value = jax.lax.dynamic_update_slice(v_state, v_new, start_indices)
        
        # Return the full sequence up to current position for attention
        # Return (B, H, pos+T, D)
        return self.k.value[layer_idx, :, :, :pos+T, :], self.v.value[layer_idx, :, :, :pos+T, :]

    def increment_pos(self, T):
        self.pos.value += T

    def get_pos(self):
        return self.pos.value

class CausalSelfAttention(nnx.Module):
    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
        self.c_q = nnx.Linear(self.n_embd, self.n_head * self.head_dim, use_bias=False, rngs=rngs)
        self.c_k = nnx.Linear(self.n_embd, self.n_kv_head * self.head_dim, use_bias=False, rngs=rngs)
        self.c_v = nnx.Linear(self.n_embd, self.n_kv_head * self.head_dim, use_bias=False, rngs=rngs)
        self.c_proj = nnx.Linear(self.n_embd, self.n_embd, use_bias=False, rngs=rngs)

    def __call__(self, x, cos_sin, mask=None, kv_cache=None, layer_idx=None):
        B, T, C = x.shape
        
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)
        
        # Rotary Embeddings
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        
        # QK Norm
        q = rms_norm(q)
        k = rms_norm(k)
        
        # Transpose to (B, H, T, D)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # KV Cache
        if kv_cache is not None and layer_idx is not None:
            # Update cache
            # k, v are (B, H, T, D)
            # cache expects (B, H, T_total, D)
            k, v = kv_cache.update(layer_idx, k, v)
            
        # GQA
        if self.n_kv_head != self.n_head:
            n_rep = self.n_head // self.n_kv_head
            k = jnp.repeat(k, n_rep, axis=1)
            v = jnp.repeat(v, n_rep, axis=1)
            
        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = jnp.einsum('bhtd,bhsd->bhts', q, k) * scale
        
        if mask is not None:
            # mask shape should match attn_logits shape or broadcast
            # attn_logits: (B, H, T_q, T_k)
            attn_logits = attn_logits + mask
        else:
            # Causal mask for training (T_q == T_k)
            if T > 1 and kv_cache is None:
                causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
                causal_mask = jnp.where(causal_mask, 0, -1e9)
                attn_logits = attn_logits + causal_mask
            
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        y = jnp.einsum('bhts,bhsd->bhtd', attn_weights, v)
        
        y = jnp.transpose(y, (0, 2, 1, 3)) # (B, T, H, D)
        y = y.reshape(B, T, self.n_embd)
        
        y = self.c_proj(y)
        return y

class MLP(nnx.Module):
    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        self.c_fc = nnx.Linear(config.n_embd, 4 * config.n_embd, use_bias=False, rngs=rngs)
        self.c_proj = nnx.Linear(4 * config.n_embd, config.n_embd, use_bias=False, rngs=rngs)

    def __call__(self, x):
        x = self.c_fc(x)
        x = jnp.square(jax.nn.relu(x))
        x = self.c_proj(x)
        return x

class Block(nnx.Module):
    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        self.attn = CausalSelfAttention(config, rngs=rngs)
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, x, cos_sin, mask=None, kv_cache=None, layer_idx=None):
        x = x + self.attn(rms_norm(x), cos_sin, mask, kv_cache, layer_idx)
        x = x + self.mlp(rms_norm(x))
        return x

class GPT(nnx.Module):
    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        self.config = config
        self.wte = nnx.Embed(config.vocab_size, config.n_embd, rngs=rngs)
        self.h = nnx.List([Block(config, rngs=rngs) for _ in range(config.n_layer)])
        self.lm_head = nnx.Linear(config.n_embd, config.vocab_size, use_bias=False, rngs=rngs)
        
        # We will compute rotary embeddings on the fly in __call__ to avoid issues with NNX state slicing
        pass

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        channel_range = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.outer(t, inv_freq)
        cos, sin = jnp.cos(freqs), jnp.sin(freqs)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        return cos, sin

    def __call__(self, idx, targets=None, kv_cache=None):
        B, T = idx.shape
        
        # Rotary embeddings
        if kv_cache is not None:
            start_pos = kv_cache.get_pos()
        else:
            start_pos = 0
            
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.config.sequence_len * 10, head_dim)
        cos = cos[:, start_pos:start_pos+T, :, :]
        sin = sin[:, start_pos:start_pos+T, :, :]
        cos_sin = (cos, sin)
        
        x = self.wte(idx)
        x = rms_norm(x)
        
        for i, block in enumerate(self.h):
            x = block(x, cos_sin, kv_cache=kv_cache, layer_idx=i)
            
        if kv_cache is not None:
            kv_cache.increment_pos(T)
            
        x = rms_norm(x)
        
        logits = self.lm_head(x)
        softcap = 15.0
        logits = softcap * jnp.tanh(logits / softcap)
        
        if targets is not None:
            import optax
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
            mask = (targets != -1)
            loss = jnp.sum(loss * mask) / jnp.sum(mask)
            return loss
        else:
            return logits
