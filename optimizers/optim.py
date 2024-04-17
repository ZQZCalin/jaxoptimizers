"""Optimizers."""

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
import chex
import optax
from optax import Updates, Params, OptState, ScalarOrSchedule, GradientTransformation
from typing import Any, Tuple, NamedTuple, Optional, Callable
import sys
sys.path.append('../jaxoptimizers')
import util
import scheduler
import logstate
import benchmark


class PolarDescentState(NamedTuple):
    """polar_descent state."""
    count: chex.Array
    dir_mu: Updates
    dir_nu: Updates
    mag_mu: Updates
    mag_nu: Updates


def polar_descent(
    direction_lr: ScalarOrSchedule,
    magnitude_lr: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    direction_wd = 0.0,
    magnitude_wd = 0.0,
) -> GradientTransformation:
    """Decomposes direction and magnitude and updates each separately.

    Updates both direction and magnitude with AdamW. 
    Specifically, parameter x is decomposed into x = r*u where r=|x| and u=x/|x|.
    The gradient is then decomposed into df/dr = <grad, u> and df/du = grad*r.
    In the future, this should support any base optimizer (just like any blackbox reduction).

    Args:
        direction_lr: Learning rate schedule for direction adamw.
        magnitude_lr: Learning rate schedule for magnitude adamw.
        b1: First order momentum constant. Defaults to 0.9.
        b2: Second order momentum constant. Defaults to 0.999.
        weight_decay: Weight decay constant. Defaults to 0.01.

    Returns:
        A `GradientTransformation` object.
    """

    def init_fn(params):
        return PolarDescentState(
            count=jnp.zeros([], jnp.int32),
            dir_mu=util.zero_tree(params),
            dir_nu=util.zero_tree(params),
            mag_mu=jnp.zeros([]),
            mag_nu=jnp.zeros([]),
        )
    
    def update_fn(updates, state, params):
        mag = util.tree_l2_norm(params)
        dir = util.tree_scalar_multiply(params, 1/mag)
        dir_grads = util.tree_scalar_multiply(updates, mag)
        mag_grads = util.tree_inner_product(updates, dir)
        count_inc = optax.safe_int32_increment(state.count)

        # Update direction.
        dir_mu = jtu.tree_map(
            lambda m, g: b1*m + (1-b1)*g, state.dir_mu, dir_grads)
        dir_nu = jtu.tree_map(
            lambda v, g: b2*v + (1-b2)*g**2, state.dir_nu, dir_grads)
        dir_mu_hat = util.tree_scalar_multiply(dir_mu, 1/(1-b1**count_inc))
        dir_nu_hat = util.tree_scalar_multiply(dir_nu, 1/(1-b2**count_inc))
        dir_eta = scheduler.get_current_lr(direction_lr, state.count)
        dir_updates = jtu.tree_map(
            lambda m, v, p: -dir_eta * (m/(eps+jnp.sqrt(v)) + direction_wd*p),
            dir_mu_hat, dir_nu_hat, dir
        )
        new_dir = util.tree_normalize(
            optax.apply_updates(dir, dir_updates))

        # Update magnitude.
        mag_mu = jtu.tree_map(
            lambda m, g: b1*m + (1-b1)*g, state.mag_mu, mag_grads)
        mag_nu = jtu.tree_map(
            lambda v, g: b2*v + (1-b2)*g**2, state.mag_nu, mag_grads)
        mag_mu_hat = util.tree_scalar_multiply(mag_mu, 1/(1-b1**count_inc))
        mag_nu_hat = util.tree_scalar_multiply(mag_nu, 1/(1-b2**count_inc))
        mag_eta = scheduler.get_current_lr(magnitude_lr, state.count)
        mag_updates = jtu.tree_map(
            lambda m, v, p: -mag_eta * (m/(eps+jnp.sqrt(v)) + magnitude_wd*p),
            mag_mu_hat, mag_nu_hat, mag
        )
        new_mag = optax.apply_updates(mag, mag_updates)

        # Combine direction and magnitude.
        new_params = util.tree_scalar_multiply(new_dir, new_mag)
        new_updates = util.tree_subtract(new_params, params)
        return new_updates, PolarDescentState(
            count=count_inc,
            dir_mu=dir_mu,
            dir_nu=dir_nu,
            mag_mu=mag_mu,
            mag_nu=mag_nu,
        )
    
    return GradientTransformation(init_fn, update_fn)


class JumpTrajectoryState(NamedTuple):
    """jump_trajectory state."""
    count: chex.Array
    normal_state: OptState
    jump_state: OptState


# TODO: maybe we can implement something even smarter that automatically detects local minimum and performs a tangential jump.
def jump_trajectory(
    normal_optim: GradientTransformation,
    jump_optim: GradientTransformation,
    normal_steps: int = 4500,
    jump_steps: int = 500,
) -> GradientTransformation:
    
    total_steps = normal_steps + jump_steps

    def init_fn(params):
        return JumpTrajectoryState(
            count=jnp.zeros([], jnp.int32),
            normal_state=normal_optim.init(params),
            jump_state=jump_optim.init(util.zero_tree(params)),
        )

    def update_fn(updates, state, params):
        normal_phase = jnp.mod(state.count, total_steps) < normal_steps
        phase_count = jax.lax.cond(
            normal_phase,
            lambda _: jnp.mod(state.count, total_steps),
            lambda _: jnp.mod(state.count, total_steps) - normal_steps,
            operand=None
        )
        def normal_update(_):
            new_updates, dir_state = dir_optim.update(updates, state.dir_state, params)
            return
        def jump_update(_):
            return
        new_updates, dir_state, mag_state = jax.lax.cond(
            normal_phase, normal_update, jump_update, operand=None)
        return new_updates, JumpTrajectoryState(
            count=optax.safe_int32_increment(state.count),
            dir_state=dir_state,
            mag_state=mag_state,
        )
    
    return GradientTransformation(init_fn, update_fn)