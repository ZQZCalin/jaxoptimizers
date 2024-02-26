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
        # updates: A pytree of tuple (g_t, beta_t, mu_t).
        updates: A pytree of gradients g_t
        state:
        params:
    
    Returns: 
        Delta_t that minimizes the regret w.r.t. the exponentiated and regularized loss
        $$
        \ell_t(Delta) = <beta_t*g_t, Delta> + mu_t/2*||Delta||^2.
        $$
"""

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
import chex
import optax
from optax import Updates, OptState, ScalarOrSchedule, GradientTransformation
from typing import Any, Tuple, NamedTuple, Optional
import sys
sys.path.append('../jaxoptimizers')
from util import tree_scalar_multiply


class ScaleByOnlineLearnerState(NamedTuple):
    """Scale by Online Learner State"""
    Delta: Updates
    opt_state: OptState


# Note: this can be freely replaced with any online learner that returns Delta in each iteration.
# This only serves as an easy wrapper for popular online learners; and I can imagine for parameter-free
# algorithms, implemeting them from scratch would be easier.
def scale_by_online_learner(
    ol_optimizer: GradientTransformation,
    projection_norm: Optional[float] = None,
    exponentiated_gradient: bool = False,
) -> GradientTransformation:
    """Updates online learner and returns the updated parameter Delta_t.

    Args:
        ol_optimizer (GradientTransformation): The online learner optimizer used to update Delta (i.e., returning Delta_n-Delta_{n-1}).
        projection_norm (Optional[float]): If not None, clip the parameter to global norm <= `projection`. Defaults to None.
        exponentiated_gradient (bool): Whether it is used in standard O2NC (False) or Exponentiated O2NC (True). Defaults to False.

    Returns:
        A `GradientTransformation` object.
    """
    
    def init_fn(params):
        Delta = jtu.tree_map(jnp.zeros_like, params)
        opt_state = ol_optimizer.init(Delta)
        return ScaleByOnlineLearnerState(
            Delta=Delta, opt_state=opt_state)
    
    def update_fn(updates, state, params=None):
        # Performs a one-step update of the online learner:
        #   Updates the params (Delta) and opt_state of the online learner.
        # 
        # Here updates can be either a pytree of g_t (in standard O2NC)
        # or a pytree of tuple (g_t, beta_t, mu_t) (in Exponentiated O2NC).
        # ====================================================================================
        del params
        # Compute gradient of the exponentiated and l2-regularized loss
        def linearize(updates_, params_):
            g, beta, mu = updates_
            return beta*g + mu*params_
        if exponentiated_gradient:
            updates = jtu.tree_map(
                linearize, updates, state.Delta)
        # Update Delta.
        Delta_updates, opt_state = state.ol_optimizer.update(
            updates, state.opt_state, state.Delta)
        Delta = optax.apply_updates(state.Delta, Delta_updates)
        # Optional: project Delta into a constrained domain.
        if projection_norm:
            clip_norm = jnp.minimum(1, projection_norm/optax.global_norm(Delta))
            Delta = tree_scalar_multiply(Delta, clip_norm)
        # TODO: return state.Delta or new Delta?
        # need to check which one adheres to the notion in the references.
        return Delta, ScaleByOnlineLearnerState(
            Delta=Delta, opt_state=opt_state)
    
    return GradientTransformation(init_fn, update_fn)


def ogd(
    learning_rate: optax.ScalarOrSchedule,
    projection_norm: Optional[float] = None,
    exponentiated_gradient: bool = False,
) -> GradientTransformation:
    """Online Gradient Descent (OGD).

    Args:
        learning_rate (optax.ScalarOrSchedule): OGD learning rate.
        projection_norm (Optional[float], optional): _description_. Defaults to None.
        exponentiated_gradient (bool, optional): _description_. Defaults to False.

    Returns:
        A `GradientTransformation` object.
    """
    return scale_by_online_learner(
        ol_optimizer=optax.sgd(learning_rate),
        projection_norm=projection_norm,
        exponentiated_gradient=exponentiated_gradient
    )


class UnconstrainedOGDState(NamedTuple):
    """Unconstrained OGD State."""
    count: chex.Array
    Delta: Updates


def unconstrained_ogd(
    learning_rate: optax.ScalarOrSchedule,
    beta: float = 0.99,
    mu: float = 100.0,
) -> GradientTransformation:
    """Unconstrained OGD as implemented in [Exponentiated O2NC](XXX).

    Given learning rate eta, exponentiate constant beta, and regularzation factor mu, updates
    Delta <- beta/(1+eta*mu) * (Delta - eta*grad).

    Note that this is equivalent to weight decay, with slightly different weight_decay constant.

    Set beta = 1 and mu = 0 recovers standard OGD.

    Args:
        learning_rate (optax.ScalarOrSchedule): _description_
        beta (float, optional): _description_. Defaults to 0.99.
        mu (float, optional): _description_. Defaults to 100.

    Returns:
        A `GradientTransformation` object.
    """
    
    def init_fn(params):
        return UnconstrainedOGDState(
            count=jnp.zeros([], jnp.int32),
            Delta=jtu.tree_map(jnp.zeros_like, params))
    
    def update_fn(updates, state, params=None):
        del params
        count_inc = optax.safe_int32_increment(state.count)
        if callable(learning_rate):
            # eta = learning_rate(count_inc)
            eta = learning_rate(state.count)
        else:
            eta = learning_rate
        new_Delta = jtu.tree_map(
            lambda Delta, g: beta/(1+eta*mu) * (Delta-eta*g),
            state.Delta, updates
        )
        return new_Delta, UnconstrainedOGDState(
            count=count_inc, Delta=new_Delta)
    
    return GradientTransformation(init_fn, update_fn)


class AdaFTRLState(NamedTuple):
    """Adaptive FTRL State."""
    count: chex.Array
    mu: Updates
    nu: Updates


def ada_ftrl(
    learning_rate: ScalarOrSchedule,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    scale_exponential: bool = False,
    scale_lr: bool = False,
    scale_eps: bool = False,
) -> GradientTransformation:
    """Ada-FTRL.

    See notes for the update.

    **Note:** with scale_exponential = False, this algorithm is almost Adam, up to two differences:
        - mu and nu are not debiased;
        - eps is scaled by an exponential decay beta2**(t/2).

    Args:
        learning_rate (ScalarOrSchedule): _description_
        beta1 (float, optional): _description_. Defaults to 0.9.
        beta2 (float, optional): _description_. Defaults to 0.999.
        eps (float, optional): _description_. Defaults to 1e-8.
        scale_exponential (bool, optional): If true, scale the update by (sqrt(beta2)/beta1)**t. Defaults to False.
        scale_lr (bool, optional): If true, scale the learning rate by sqrt(1-beta2)/(1-beta1). Defaults to False.
        scale_eps (bool, optional): If true, scale eps by sqrt(1-beta2). Defaults to False.

    Returns:
        A `GradientTransformation` object.
    """

    if scale_lr:
        scale_lr_const = (1-beta2)**.5 / (1-beta1)
        if callable(learning_rate):
            learning_rate = lambda n: scale_lr_const * learning_rate(n)
        else:
            learning_rate *= scale_lr_const
    
    if scale_eps:
        eps *= (1-beta2)**.5

    def init_fn(params):
        return AdaFTRLState(
            count=jnp.zeros([], jnp.int32),
            mu=jtu.tree_map(jnp.zeros_like, params),
            nu=jtu.tree_map(jnp.zeros_like, params)
        )
    
    def update_fn(updates, state, params=None):
        del params
        mu = jtu.tree_map(
            lambda m, g: beta1*m + (1-beta1)*g, state.mu, updates)
        nu = jtu.tree_map(
            lambda v, g: beta2*v + (1-beta2)*g**2, state.nu, updates)
        if scale_exponential:
            scalar = (beta2**.5/beta1)**state.count
        else:
            scalar = 1
        if callable(learning_rate):
            eta = learning_rate(state.count)
        else:
            eta = learning_rate
        Delta = jtu.tree_map(
            lambda m, v: -scalar * eta * m / (beta2**(state.count/2)*eps + jnp.sqrt(v)),
            mu, nu)
        return Delta, AdaFTRLState(
            count=optax.safe_int32_increment(state.count), mu=mu, nu=nu)
    
    return GradientTransformation(init_fn, update_fn)


def adagrad() -> GradientTransformation:
    return


class KTBetterState(NamedTuple):
    """KT coin better state."""
    sum_grad: float
    wealth: float


def kt_better(
    eps: float = 1.0,
) -> GradientTransformation:
    
    def init_fn(params=None):
        del params
        return KTBetterState(sum_grad=.0, wealth=eps)
    
    def update_fn(updates, state, params):
        return 
    
    return 


class BlackBoxReductionState(NamedTuple):
    """Black box reduction state."""
    magnitude_state: OptState
    direction_state: OptState


def black_box_reduction(
    magnitude_learner: GradientTransformation,
    direction_learner: GradientTransformation,
) -> GradientTransformation:
    
    def init_fn(params):
        magnitude_state = magnitude_learner.init(params)
        direction_state = direction_learner.init(
            jtu.tree_map(jnp.zeros_like, params))
        return BlackBoxReductionState(
            magnitude_state=magnitude_state,
            direction_state=direction_state
        )
    
    def update_fn(updates, state, params):
        xt, magnitude_state = magnitude_learner.update(
            updates, state.magnitude_state, params)
        zt, direction_state = direction_learner.update()
        updates = jtu.tree_map()
        return updates, BlackBoxReductionState()
    
    return GradientTransformation(init_fn, update_fn)

def cocob() -> GradientTransformation:
    return


def dog() -> GradientTransformation:
    return


def pfmd() -> GradientTransformation:
    return