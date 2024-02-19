"""Online convex optimization algorithms for (Exponentiated) O2NC.

update_fn for standard O2NC:
    Args:
        updates: A pytree of gradients g_t.
        state:
        params:
    
    Returns: 
        Delta_t that minimizes the regret w.r.t. the linear loss
        $$
        \ell_t(Delta) = <g_t, Delta>.
        $$

update_fn for Exponentiated O2NC:
    Args:
        updates: A pytree of tuple (g_t, beta_t, mu_t).
        state:
        params:
    
    Returns: 
        Delta_t that minimizes the regret w.r.t. the exponentiated and regularized loss
        $$
        \ell_t(Delta) = <beta_t g_t, Delta> + mu_t/2 ||Delta||^2.
        $$
"""

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
import chex
import optax
from optax import Updates, OptState, GradientTransformation
from typing import Any, Tuple, NamedTuple, Optional


class OGDState(NamedTuple):
    """Online Gradient Descent State."""
    optimizer: GradientTransformation
    opt_state: OptState


def ogd(
    learning_rate: optax.ScalarOrSchedule,
    projection: Optional[float] = None,
    exponentiated: bool=False,
) -> GradientTransformation:
    """"""

    def init_fn(params):
        optimizer = optax.chain(
            optax.sgd(learning_rate)
        )
        opt_state = optimizer.init()
        return OGDState(
            Delta=jtu.tree_map(jnp.zeros_like, params))
    
    def update_fn(updates, state, params):
        Delta = None
        return Delta, OGDState(Delta=Delta)
    
    return GradientTransformation(init_fn, update_fn)


def adagrad() -> GradientTransformation:
    return