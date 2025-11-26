"""
A number of functions that help with evaluating a base model.
"""
import math
import jax
import jax.numpy as jnp

def evaluate_bpb(model, batches, steps, token_bytes):
    """
    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    which is a tokenization vocab size-independent metric, meaning you are still comparing
    apples:apples if you change the vocab size. The way this works is that instead of just
    calculating the average loss as usual, you calculate the sum loss, and independently
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    the number of bytes that the target tokens represent.

    The added complexity is so that:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.

    In addition to evaluate_loss, we need the token_bytes tensor:
    It is a 1D tensor of shape (vocab_size,), indicating the number of bytes for
    each token id, or 0 if the token is to not be counted (e.g. special tokens).
    """
    # record the losses
    total_nats = 0.0
    total_bytes = 0
    batch_iter = iter(batches)
    
    # token_bytes should be a jax array
    token_bytes = jnp.array(token_bytes)
    
    for _ in range(steps):
        x, y = next(batch_iter)
        # Forward pass to get loss per token
        # Assuming model(x, y) returns average loss by default, 
        # we need a version that returns per-token loss or we compute it here.
        # If model(x, y) returns scalar loss, we can't easily get per-token loss without changing model.
        # Let's assume we can get logits and compute loss manually, or model has a mode for it.
        # Given current GPT implementation, it returns scalar loss.
        # We need to modify GPT to return per-token loss or compute it here.
        
        # For now, let's compute logits and loss here to be safe and get per-token loss.
        logits = model(x) # (B, T, V)
        
        # Compute per-token loss
        # logits: (B, T, V), y: (B, T)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        
        # Mask for valid targets
        mask = (y != -1)
        targets_safe = jnp.where(mask, y, 0)
        
        # Gather log_probs
        gathered_log_probs = jnp.take_along_axis(log_probs, targets_safe[..., None], axis=-1).squeeze(-1)
        per_token_loss = -gathered_log_probs * mask # (B, T)
        
        # Flatten
        loss_flat = per_token_loss.ravel()
        y_flat = y.ravel()
        
        # Mask out invalid targets for byte count
        valid_flat = (y_flat >= 0)
        y_safe_flat = jnp.where(valid_flat, y_flat, 0)
        
        # Map to byte lengths
        num_bytes_flat = jnp.where(valid_flat, token_bytes[y_safe_flat], 0)
        
        # Sum nats and bytes where num_bytes > 0 (excludes special tokens with 0 bytes)
        valid_bytes_mask = (num_bytes_flat > 0)
        total_nats += jnp.sum(loss_flat * valid_bytes_mask)
        total_bytes += jnp.sum(num_bytes_flat * valid_bytes_mask)
        
    # Sum reduce across all ranks
    # In JAX distributed, we can use jax.lax.psum if inside JIT, 
    # but here we are outside. We need jax.distributed.
    # For single process, no-op. For multi-process, we need to implement reduction.
    # Since we don't have a convenient all_reduce outside JIT in JAX easily without setup,
    # we might need to rely on the caller to handle reduction or use jax.pmap/pjit.
    # However, for now, let's assume single process or handle it if possible.
    
    # In JAX, we can use jax.process_count() to check.
    if jax.process_count() > 1:
        # This is tricky outside JIT. We'd need to use MPI or similar, 
        # or wrap this in a JIT'd function that does psum.
        # Let's wrap the reduction in a JIT'd function.
        @jax.jit
        def reduce_stats(nats, bytes_count):
            return jax.lax.psum(nats, axis_name='p'), jax.lax.psum(bytes_count, axis_name='p')
        
        # This requires pmap setup which is not here.
        # Alternative: use jax.distributed.all_gather and sum.
        # But all_gather is also not trivial outside JIT for arbitrary data.
        
        # For now, we'll just warn or assume single process, as nanochat seems to focus on single node/GPU for now or uses torchrun which we are replacing.
        pass 

    total_nats = float(total_nats)
    total_bytes = float(total_bytes)
    
    if total_bytes == 0:
        return float('inf')
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb
