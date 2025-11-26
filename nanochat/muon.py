"""
Muon optimizer for JAX/Optax.
Wrapper around optax.contrib.muon.
"""

import optax
try:
    from optax.contrib import muon
except ImportError:
    # Fallback if not available (though we checked it is)
    raise ImportError("optax.contrib.muon not found. Please upgrade optax.")

def get_muon(learning_rate, momentum=0.95, nesterov=True, ns_steps=5):
    """
    Returns the Muon optimizer transformation.
    Note: optax.contrib.muon uses 'beta' parameter instead of 'momentum'.
    """
    return muon(learning_rate=learning_rate, beta=momentum, nesterov=nesterov, ns_steps=ns_steps)
