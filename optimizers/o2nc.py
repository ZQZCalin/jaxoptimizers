"""Online-to-non-convex Conversion."""

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
import chex
import optax
from optax import Updates, OptState, GradientTransformation
from typing import Any, Tuple, NamedTuple, Optional
from util import tree_add, tree_subtract, tree_scalar_multiply


class ScaleByRandomState(NamedTuple):
    """scale_by_random state."""
    key: chex.Array


def scale_by_random(
    sample_fn: Any,
    seed: int = 0,
) -> GradientTransformation:
    """Scales the update by a random variable.
    
    Args:
        sample_fn: A function that receives a PRNGKeyArray and returns a random number.
        seed (int): Seed for jax.random.PRNGKey.
    
    Returns:
        A `GradientTransform` object.
    """

    def init_fn(params=None):
        return ScaleByRandomState(key=jr.PRNGKey(seed))
    
    def update_fn(updates, state, params=None):
        del params
        key1, key2 = jr.split(state.key)
        scaling = sample_fn(key1)
        new_updates = tree_scalar_multiply(updates, scaling)
        return new_updates, ScaleByRandomState(key=key2)
    
    return GradientTransformation(init_fn, update_fn)


class EMAParamsState(NamedTuple):
    """Exponential moving average of params State."""
    x_bar: Updates


def ema_params(
    beta: float = 0.99,
) -> GradientTransformation:
    """Computes the (unnormalized) exponential moving average of past params:

    ..math::
        S_n = \\beta S_{n-1} + (1-\\beta) x_n

    such that :math:`S_n = (1-\\beta^n)\\times\ttext{EMA}_n`.

    **Note:** This step is totally optional for output purpose, and does not affect the training process.

    Args:
        beta (float, optional): _description_. Defaults to 0.99.

    Returns:
        A `GradientTransformation` object.
    """

    def init_fn(params):
        x_bar = jtu.tree_map(
            jnp.zeros_like, params)
        return EMAParamsState(x_bar=x_bar)
    
    def update_fn(updates, state, params):
        # Only aggregates EMA of params while passing down the original updates untouched.
        x_bar = jtu.tree_map(
            lambda x, p: beta*x + (1-beta)*p, state.x_bar, params)
        return updates, EMAParamsState(x_bar=x_bar)

    return GradientTransformation(init_fn, update_fn)


class ExponentiateLossState(NamedTuple):
    """Exponentiate loss State."""
    count: chex.Array


def exponentiate_loss(
    beta: float = 0.99,
    # Note: an alternative is to always set $\beta\mu=N$.
    mu: float = 100.0,
) -> GradientTransformation:
    """Prepares the exponentiated and regularized loss and passes down 
    a tuple of (g_t, beta_t, mu_t) to the specific online learner.

    Args:
        beta (float): Exponential constant.
        mu (float): Regularization constant.

    Returns:
        A `GradientTransformation` object.
    """

    def init_fn(params=None):
        count = jnp.zeros([], jnp.int32)
        return ExponentiateLossState(count=count)
    
    def update_fn(updates, state, params=None):
        # As the first layer of Exponentiated O2NC, `updates`` is a pytree of gradients.
        # See train.train_step for how this wrapper is involved in the training process.
        # 
        # The `new_update` should be a pytree of tuple (g_t, beta_t, mu_t).
        # ====================================================================================
        del params
        count_inc = optax.safe_int32_increment(state.count)
        # Safe incrementation that avoids (max_int + 1) -> min_int.
        # https://optax.readthedocs.io/en/latest/api/utilities.html#optax.safe_int32_increment
        beta_t = beta ** -count_inc
        mu_t = mu * beta_t
        new_updates = jtu.tree_map(
            lambda g_t: (g_t, beta_t, mu_t), updates)
        return new_updates, ExponentiateLossState(count=count_inc)
    
    return GradientTransformation(init_fn, update_fn)


class ScaleByOnlineLearnerState(NamedTuple):
    """Scale by Online Learner State"""
    Delta: Updates
    opt_state: OptState


def scale_by_online_learner(
    online_learner: GradientTransformation,
    projection: Optional[float] = None,
) -> GradientTransformation:
    """Updates online learner and returns the updated parameter Delta_t.

    Args:
        online_learner (GradientTransformation): The online learner algorithm used to update Delta.
        projection (Optional[float]): If not None, clip the parameter to global norm <= `projection`. Defaults to None.

    Returns:
        A `GradientTransformation` object.
    """
    
    def init_fn(params):
        Delta = jtu.tree_map(jnp.zeros_like, params)
        opt_state = online_learner.init(Delta)
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
        Delta_updates, opt_state = state.optimizer.update(
            updates, state.opt_state, state.Delta)
        Delta = optax.apply_updates(state.Delta, Delta_updates)
        # Optional: project Delta into a constrained domain.
        if projection:
            clip_norm = jnp.minimum(1, projection/optax.global_norm(Delta))
            Delta = tree_scalar_multiply(Delta, clip_norm)
        # TODO: return state.Delta or new Delta?
        # need to check which one adheres to the notion in the references.
        return Delta, ScaleByOnlineLearnerState(
            Delta=Delta, opt_state=opt_state)
    
    return GradientTransformation(init_fn, update_fn)


class ScaleByO2NCState(NamedTuple):
    """scale_by_o2nc state."""
    aux_params: Updates
    key: jax.Array


def scale_by_o2nc(
    seed: int = 0,
) -> GradientTransformation:
    """Stores the auxiliary parameter :math:`x_n` in standard O2NC.

    Recall that :math:`x_n = x_{n-1} + \Delta_n` and :math:`w_n = x_{n-1} + s_n\Delta_n`.
    Given params=:math:`w_{n-1}` and updates=:math:`\Delta_n`, this function stores the 
    auxiliary parameter x_n and returns :math:`w_n-w_{n-1} = x_{n-1} - w_{n-1} + s_n\Delta_n`.

    Args:
        seed (int): PRNGKey for uniform scaling. Defaults to 0.

    Returns:
        A `GradientTransformation` object.
    """

    def init_fn(params):
        return ScaleByO2NCState(
            aux_params=jtu.tree_map(jnp.copy, params),
            key=jr.PRNGKey(seed)
        )
    
    def update_fn(updates, state, params):
        x_prev = state.aux_params
        x_new = tree_add(x_prev, updates)
        # TODO: is it possible to separate random scaling from updaing aux_params?
        key1, key2 = jr.split(state.key)
        unif_scaling = jr.uniform(key1)
        w_diff = jtu.tree_map(
            lambda x, w, Delta: x - w + unif_scaling*Delta,
            x_prev, params, updates
        )
        return w_diff, ScaleByO2NCState(aux_params=x_new, key=key2)
    
    return GradientTransformation(init_fn, update_fn)


def o2nc(
    online_learner: GradientTransformation,
    projection: Optional[float] = None,
    seed: int = 0,
) -> GradientTransformation:
    """Online-to-non-convex Conversion.

    References:
        [XXX](...)

    Args:
        online_learner (GradientTransformation): Online convex optimization algorithm.
        projection (Optional[float]): If not None, clip the parameter to global norm <= `projection`. Defaults to None.
        seed (int): PRNGKey for uniform scaling. Defaults to 0.

    Returns:
        A `GradientTransformation` object.
    """
    return optax.chain(
        scale_by_online_learner(online_learner, projection),
        scale_by_o2nc(seed)
    )


def eo2nc(
    online_learner: GradientTransformation,
    projection: Optional[float] = None,
    beta: float = 0.99,
    mu: float = 100.0,
    seed: int = 0,
) -> GradientTransformation:
    """Exponentiated Online-to-non-convex Conversion.

    References:
        [XXX](...)

    Args:
        online_learner (GradientTransformation): Online convex optimization algorithm.
        projection (Optional[float]): If not None, clip the parameter to global norm <= `projection`. Defaults to None.
        beta (float): Exponential constant. Defaults to 0.99.
        mu (float): Regularization constant. Defaults to 100.0.
        seed (int): PRNGKey for exponential scaling. Defaults to 0.

    Returns:
        A `GradientTransformation` object.
    """
    return optax.chain(
        exponentiate_loss(beta, mu),
        scale_by_online_learner(online_learner, projection),
        scale_by_random(
            sample_fn=lambda key: jr.exponential(key), seed=seed),
    )