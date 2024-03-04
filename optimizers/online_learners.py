"""Online convex optimization algorithms.

Please be aware that online learners have different init_fn and update_fn behavior.

update_fn(updates, state, params):
    Args:
        updates: Gradients g_t.
        state: Current state.
        params: Current parameter w_t.
            **Note:**
    
    Returns: 
        A new parameter w_{t+1} and a new state.
        **Note:** Online learners return new parameters unlike typical GradientTransformations that return updates.
        The reason is that most online learners (mirror descent, FTRL, parameter-free algorithms, etc.) do not have 
        an iterative update expression such as w_{t+1} = w_t + XXX. Hence, it would be easier to directly return the
        new parameter instead of new_params - params.

Currently the following list of functions follow this format:
    - `ada_ftrl`
    - `kt_bettor`
"""

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
import chex
import optax
from optax import Updates, Params, OptState, ScalarOrSchedule, GradientTransformation
from typing import Any, Tuple, NamedTuple, Optional, Union, Callable, Protocol
import sys
sys.path.append('../jaxoptimizers')
from util import tree_add, tree_subtract, tree_multiply, tree_scalar_multiply, tree_norm, tree_normalize, check_tree_structures_match


# Q: should we allow online learners to decide their initial parameter?
class OnlineLearnerInitFn(Protocol):
    def __call__(self, params: Params) -> Tuple[Params, OptState]:
    # def __call__(self, params: Params) -> OptState:
        """The `init` function.

        Args:
            params: The model parameter, which can be different than the online learner initial parameter.

        Returns:
            The initial parameter and the initial state of the online learner.
        """


class OnlineLearnerUpdateFn(Protocol):
    def __call__(
        self, 
        grads: Updates, 
        state: OptState, 
        params: Optional[Params]
    ) -> Tuple[Params, OptState]:
        """The `update` function.

        Args:
            grads: A tree of gradients.
            state: The state of the online learner.
            params: (Optionally) the current parameters w_t.

        Returns:
            The new parameter w_{t+1}, and the updated state.
        """


class OnlineLearner(NamedTuple):
    init: OnlineLearnerInitFn
    update: OnlineLearnerUpdateFn


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


# Next, we also provide a more general wrapper for online learners.
# If you want to use an online learner as part of a larger optimizer (say O2NC), just
# wrap the online learner with this wrapper.
class OnlineLearnerWrapperState(NamedTuple):
    """online learner wrapper state."""
    params: Updates 
    state: OptState


def online_learner_wrapper(
    online_learner: GradientTransformation,
    init_params: Params = None,
) -> GradientTransformation:
    """A wrapper for online learenr if it's used as part of the optimizer (e.g., O2NC).
    The wrapper automatically stores the params of the online learner, which may be different than the actual
    model parameter, and calls online_learner.update() in each step.

    Args:
        online_learner: _description_
        init_params: Initial parameters of the online learner. Defaults to an all-zero tree.

    Returns:
        GradientTransformation: _description_
    """

    def init_fn(params):
        new_params = jtu.tree_map(jnp.zeros_like, params) if init_params is None else init_params
        return OnlineLearnerWrapperState(
            params=new_params,
            state=online_learner.init(new_params)
        )
    
    def update_fn(updates, state, params=None):
        del params
        new_params, state = online_learner.update(updates, state.state, state.params)
        return OnlineLearnerWrapperState(params=new_params, state=state)

    return GradientTransformation(init_fn, update_fn)


# This wrapper is designed specifically for exponentiated O2NC (ER stands for exponentiated and regularized).
class EROnlineLearnerWrapperState(NamedTuple):
    params: Updates
    state: OptState
    count: chex.Array


# TODO: implement its update
def er_online_learner_wrapper(
    online_learner: GradientTransformation,
    init_params: Callable[[chex.Array], chex.Array] = jnp.zeros_like,
    beta: float = 1.0,
    mu: float = 0.0,
) -> GradientTransformation:
    """Wraps the online learner with exponentiated and regularized loss:
    <beta^-t gt, w> + mut/2 |w|^2 ==> gradient = beta^-t * (gt + mu*wt)
    """
    
    def init_fn(params):
        params = jtu.tree_map(init_params, params)
        return OnlineLearnerWrapperState(
            params=params,
            state=online_learner.init(params),
            count=jnp.ones([], jnp.int32)
        )
    
    def update_fn(updates, state, params=None):
        del params
        params, state = online_learner.update(updates, state.state, state.params)
        return OnlineLearnerWrapperState(params=params, state=state)

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


class KTBettorState(NamedTuple):
    """KT coin better state."""
    sum_grad: Updates
    wealth: Updates
    count: chex.Array


def kt_bettor(
    eps: Union[float, Any] = 1e2,
) -> GradientTransformation:
    """KT Coin Bettor.

    Note:
        By default, if dimension is higher than 1, then implements per-coordinate KT coin bettor.

    References:
        [Orabona, 2019, Alg. 9.2](https://arxiv.org/abs/1912.13213)

    Args:
        eps (float or Pytree): Initial wealth between 1 and sqrt(T). Defaults to 1.0.

    Returns:
        A `GradientTransformation` object.
    """

    def init_fn(params):
        sum_grad = jtu.tree_map(jnp.zeros_like, params)
        if type(eps) != float:
            # eps as a Pytree. Will check if eps and params have the same tree structure.
            check_tree_structures_match(eps, params)
            wealth = eps
        else:
            wealth = jtu.tree_map(lambda p: jnp.full_like(p, eps), params)
        return KTBettorState(
            sum_grad=sum_grad,
            wealth=wealth,
            count=jnp.ones([], jnp.int32)
        )
    
    def update_fn(updates, state, params):
        count_inc = optax.safe_int32_increment(state.count)
        sum_grad = tree_add(state.sum_grad, updates)
        wealth = tree_subtract(state.wealth, tree_multiply(updates, params))
        new_params = jtu.tree_map(
            lambda St, Wt: - St / count_inc * Wt, sum_grad, wealth)
        return new_params, KTBettorState(
            sum_grad=sum_grad, wealth=wealth, count=count_inc)
    
    return GradientTransformation(init_fn, update_fn)


class BlackboxERFTRLState(NamedTuple):
    """Black box FTRL state."""
    momentum: Updates


def blackbox_er_ftrl(
    beta: float = 1.0,
    mu: float = 0.0,
) -> GradientTransformation:
    
    assert beta >= 0 and beta <= 1, "beta must be between 0 and 1."

    def init_fn(params):
        return BlackboxERFTRLState(momentum=jtu.tree_map(jnp.zeros_like, params))
    
    def update_fn(updates, state, params):
        if beta == 1.0:
            momentum = jtu.tree_map(
                lambda m, g, w: m - (g+mu*w), state.momentum, updates, params)
        else:
            momentum = jtu.tree_map(
                lambda m, g, w: beta*m - (1-beta)*(g+mu*w), state.momentum, updates, params)
        return tree_normalize(momentum), BlackboxERFTRLState(momentum)

    return GradientTransformation(init_fn, update_fn)


class BlackboxReductionState(NamedTuple):
    """Black box reduction state."""
    magnitude_state: OptState
    direction_state: OptState


def blackbox_reduction(
    magnitude_learner: GradientTransformation,
    direction_learner: GradientTransformation,
    # global_scaling: bool = True
) -> GradientTransformation:
    """Black-box reduction algorithm.

    References:
        [Cutkosky & Orabona, 2018](https://arxiv.org/abs/1802.06293)

    Args:
        magnitude_learner (GradientTransformation): 
            Online learner (typically parameter-free algorithms) for magnitude; learns :math:`\|x_t\|`.
        direction_learner (GradientTransformation): 
            Online learner for direction; learns :math:`x_t / \|x_t\|`.
        global_scaling (bool, optional): 
            If true, there is one global learning rate scalar for all parameters; 
            Otherwise, each tree node has each own lr scalar. Defaults to True.
            For now, I DONT plan to implement global scaling.

    Returns:
        A `GradientTransformation` object.
    """
    
    def init_fn(params):
        params_norm = tree_norm(params)
        if params_norm == 0.:
            direction_params = params
        else:
            direction_params = tree_scalar_multiply(params, 1/params_norm)
        direction_state = direction_learner.init(direction_params)
        magnitude_state = magnitude_learner.init(params_norm)
        # Note: In the first iterate, last_iterate is initialized as params, the initial parameter 
        # of the model, where g1 is evaluated.
        return BlackboxReductionState(
            magnitude_state=magnitude_state,
            direction_state=direction_state
        )
    
    def update_fn(updates, state, params):
        st = jnp.array(
            jtu.tree_reduce(
                lambda x, y: x + y,
                jtu.tree_map(lambda g, w: jnp.dot(g, w), updates, params)))
        xt, direction_state = direction_learner.update(
            updates, state.direction_state, params)
        # TODO: change param to 1D params
        zt, magnitude_state = magnitude_learner.update(
            st, state.magnitude_state, params)
        new_params = jtu.tree_map(lambda z, x: z*x, zt, xt)
        return new_params, BlackboxReductionState(
            magnitude_state=magnitude_state,
            direction_state=direction_state
        )
    
    return GradientTransformation(init_fn, update_fn)


def cocob() -> GradientTransformation:
    return


def dog() -> GradientTransformation:
    return


def pfmd() -> GradientTransformation:
    return


# =============== TESTING ===============
def test_online_learner(
    online_learner: GradientTransformation,
):
    # Simple pytree for testing
    grads = {
        'a': [jnp.array(1.), jnp.array(2.)],  # List of arrays
        'b': (jnp.array(3.), jnp.array(4.)),  # Tuple of arrays
        'c': {'d': jnp.array(5.)}  # Nested dictionary with an array
    }
    params = jtu.tree_map(jnp.zeros_like, grads)
    print("initial params:\n", params)
    
    opt_state = online_learner.init(params)
    params, opt_state = online_learner.update(grads, opt_state, params)
    print("new params:\n", params)


if __name__ == "__main__":
    magnitude = kt_bettor()
    direction = blackbox_er_ftrl(beta=1.0, mu=0.0)
    blackbox = blackbox_reduction(magnitude, direction)

    # test_online_learner(magnitude)
    # test_online_learner(online_learner_wrapper(magnitude))
    # test_online_learner(direction)
    # test_online_learner(online_learner_wrapper(direction))
    test_online_learner(blackbox)