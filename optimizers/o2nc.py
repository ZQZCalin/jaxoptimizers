"""Online-to-non-convex Conversion."""

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
import chex
import optax
from optax import Updates, OptState, ScalarOrSchedule, GradientTransformation
from typing import Any, Tuple, NamedTuple, Optional, Callable
from online_learners import unconstrained_ogd
import sys
sys.path.append('../jaxoptimizers')
from util import tree_add, tree_subtract, tree_scalar_multiply


SampleFunction = Callable[[chex.Array], chex.Numeric]


class ScaleByRandomState(NamedTuple):
    """scale_by_random state."""
    key: chex.Array


def scale_by_random(
    sample_fn: SampleFunction,
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
        del params
        return ScaleByRandomState(key=jr.PRNGKey(seed))
    
    def update_fn(updates, state, params=None):
        del params
        key1, key2 = jr.split(state.key)
        scaling = sample_fn(key1)
        new_updates = tree_scalar_multiply(updates, scaling)
        return new_updates, ScaleByRandomState(key=key2)
    
    return GradientTransformation(init_fn, update_fn)


def scale_by_exponential(
    lam: float = 1.0,
    seed: int = 0,
) -> GradientTransformation:
    """Scales the update by exponential random variable with mean = lam.

    Args:
        lam (float): Mean of sampled random variable. Defaults to 1.0.
        seed (int): Seed for jax.random.PRNGKey.

    Returns:
        A `GradientTransformation` object.
    """
    sample_fn = lambda key: lam * jr.exponential(key)
    return scale_by_random(sample_fn, seed)


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
    seed: int = 0,
) -> GradientTransformation:
    """Online-to-non-convex Conversion.

    References:
        [XXX](...)

    Args:
        online_learner (GradientTransformation): Online convex optimization algorithm.
        seed (int): PRNGKey for uniform scaling. Defaults to 0.

    Returns:
        A `GradientTransformation` object.
    """
    return optax.chain(
        online_learner,
        scale_by_o2nc(seed)
    )


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
            lambda g: (g, beta_t, mu_t), updates)
        return new_updates, ExponentiateLossState(count=count_inc)
    
    return GradientTransformation(init_fn, update_fn)


def eo2nc(
    online_learner: GradientTransformation,
    # beta: float = 0.99,
    # mu: float = 100.0,
    seed: int = 0,
) -> GradientTransformation:
    """Exponentiated Online-to-non-convex Conversion.

    References:
        [XXX](...)

    Args:
        online_learner (GradientTransformation): Online convex optimization algorithm.
        # beta (float): Exponential constant. Defaults to 0.99.
        # mu (float): Regularization constant. Defaults to 100.0.
        seed (int): PRNGKey for exponential scaling. Defaults to 0.

    Returns:
        A `GradientTransformation` object.
    """
    return optax.chain(
        online_learner,
        scale_by_exponential(seed)
    )
    # optax.chain(
    #     exponentiate_loss(beta, mu),
    #     online_learner,
    #     scale_by_random(
    #         sample_fn=lambda key: jr.exponential(key), seed=seed),
    # )


# Wrapped optimizers from O2NC
def eo2nc_unconstrained_ogd(
    learning_rate: ScalarOrSchedule,
    beta: float = 0.99,
    mu: float = 100.0,
    seed: int = 0,
) -> GradientTransformation:
    return optax.chain(
        unconstrained_ogd(
            learning_rate=learning_rate,
            beta=beta,
            mu=mu
        ),
        scale_by_exponential(seed=seed)
    )


# AdamW for benchmark
class AdamWState(NamedTuple):
    """AdamW State."""
    count: chex.Array
    mu: Updates
    nu: Updates


def adamw(
    learning_rate: ScalarOrSchedule = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    debias_beta1: bool = True,
    debias_beta2: bool = True,
) -> GradientTransformation:
    """AdamW for benchmark.

    Args:
        learning_rate (ScalarOrSchedule): _description_. Defaults to 1e-4.
        beta1 (float): _description_. Defaults to 0.9.
        beta2 (float): _description_. Defaults to 0.999.
        eps (float): _description_. Defaults to 1e-8.
        weight_decay (float): _description_. Defaults to 0.0.
        debias_beta1 (bool): Defaults to True.
        debias_beta2 (bool): Defaults to True.

    Returns:
        A `GradientTransformation` object.
    """

    def init_fn(params):
        return AdamWState(
            count=jnp.zeros([], jnp.int32),
            mu=jtu.tree_map(jnp.zeros_like, params),
            nu=jtu.tree_map(jnp.zeros_like, params)
        )
    
    def update_fn(updates, state, params):
        count_inc = optax.safe_int32_increment(state.count)
        mu = jtu.tree_map(
            lambda m, g: beta1*m + (1-beta1)*g, state.mu, updates)
        nu = jtu.tree_map(
            lambda v, g: beta2*v + (1-beta2)*g**2, state.nu, updates)
        # Debias to get the true weighted average.
        if debias_beta1:
            mu_hat = tree_scalar_multiply(mu, 1/(1-beta1**count_inc))
        else:
            mu_hat = mu
        if debias_beta2:
            nu_hat = tree_scalar_multiply(nu, 1/(1-beta2**count_inc))
        else:
            nu_hat = nu
        # Unpack learning rate schedule.
        if callable(learning_rate):
            eta = learning_rate(state.count)
        else:
            eta = learning_rate
        # Compute one-step update: -eta * [mu / (eps+sqrt(nu)) + lam * params]
        new_updates = jtu.tree_map(
            lambda m, v, p: -eta * (m/(eps+jnp.sqrt(v)) + weight_decay*p),
            mu_hat, nu_hat, params
        )
        return new_updates, AdamWState(
            count=count_inc, mu=mu, nu=nu)
    
    return GradientTransformation(init_fn, update_fn)


# TESTING
def test_optimizer(
    optimizer: GradientTransformation,
):
    # Simple pytree for testing
    params = {
        'a': [jnp.array(1.), jnp.array(2.)],  # List of arrays
        'b': (jnp.array(3.), jnp.array(4.)),  # Tuple of arrays
        'c': {'d': jnp.array(5.)}  # Nested dictionary with an array
    }
    grads = jtu.tree_map(jnp.ones_like, params)
    print("initial params:\n", params)
    
    opt_state = optimizer.init(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    print("new params:\n", params)
    

if __name__ == "__main__":
    # optimizer = eo2nc_unconstrained_ogd(learning_rate=0.01)
    optimizer = adamw()
    test_optimizer(optimizer)