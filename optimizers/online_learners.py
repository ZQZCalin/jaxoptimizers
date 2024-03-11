"""Online convex optimization algorithms.

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
import util
from util import tree_add, tree_subtract, tree_multiply, tree_scalar_multiply, tree_dot, tree_norm, tree_normalize, check_tree_structures_match
from logger import RateLimitedWandbLog


class OnlineLearnerInitFn(Protocol):
    def __call__(self, params: Params) -> OptState:
        """The `init` function.

        Args:
            params: The initial parameters of the online learner. 
            Note: this can be different from the model parameter if the online learner is used as the subroutine
            of the online-to-non-convex conversion.

        Returns:
            The initial state of the online learner.
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
    """A pair of init and update functions implementing online learners.

    Unlike typical optax GradientTransformations, upon calling the update function, an OnlineLearner returns the 
    new parameter (instead of an update) together with the updated state. 
    The motivation is that most online learners (mirror descent, FTRL, parameter-free algorithms, etc.) do not have 
    an iterative update expression such as w_{t+1} = w_t + eta * g_t. Hence, it would be easier to directly return the
    new parameter instead of new_params - params.
    """
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
class WrapOnlineLearnerState(NamedTuple):
    """online learner wrapper state."""
    params: Updates 
    state: OptState


def wrap_online_learner(
    online_learner: OnlineLearner
) -> OnlineLearner:
    """Wraps an online learenr.

    Automatically stores the params of the online learner, which may be different from the
    model parameter. This wrapper can be useful if the online learner is used as a subroutine
    in the online-to-non-convex conversion.

    Args:
        online_learner: An `OnlineLearner` object to be wrapped.

    Returns:
        A wrapped `OnlineLearner` object.
    """

    def init_fn(params):
        state = online_learner.init(params)
        return WrapOnlineLearnerState(
            params=params, state=state)
    
    def update_fn(updates, state, params=None):
        del params
        new_params, state = online_learner.update(updates, state.state, state.params)
        return WrapOnlineLearnerState(params=new_params, state=state)

    return OnlineLearner(init_fn, update_fn)


# =================================================================================================
# Below implements popular online learners.
# =================================================================================================


def current_lr(
    learning_rate: ScalarOrSchedule,
    count: chex.Array,
):
    """Returns the current learning rate."""    
    if callable(learning_rate):
        return learning_rate(count)
    else:
        return learning_rate


# TODO: modify OGD accordingly
class OGDState(NamedTuple):
    """ogd state."""
    count: chex.Array


def ogd(
    learning_rate: optax.ScalarOrSchedule,
    # beta: float = 1.0,
    weight_decay: float = 0.0,
) -> OnlineLearner:
    """Online Gradient Descent (OGD).

    Updates w_{t+1} <- w_t + eta_t * g_t. 
    For now, we assume the learning always cancels the exponentiated gradient.
    Therefore, beta is always 1.0 effectively.

    Args:
        learning_rate: OGD learning rate.
        beta: Exponentiated gradient constant. Defaults to 1.0 (no exponentiation).
        weight_decay: l2 regularization constant. Defaults to 0.0 (no regularization).
    """

    def init_fn(params=None):
        del params
        return OGDState(count=jnp.zeros([], jnp.int32))
    
    def update_fn(updates, state, params):
        # l2 regularization / weight decay
        grads = jtu.tree_map(
            lambda g, w: g + weight_decay*w, updates, params)
        # gradient descent
        count_inc = optax.safe_int32_increment(state.count)
        eta = current_lr(learning_rate, state.count)
        new_params = jtu.tree_map(
            lambda w, g: w - eta*g, params, grads)
        return new_params, OGDState(count=count_inc)
    
    return OnlineLearner(init_fn, update_fn)


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


# TODO: add regularization (mu).
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


class KTBettorState(NamedTuple):
    """KT coin better state."""
    sum_grad: Updates
    wealth: Updates
    count: chex.Array


# TODO: support Pytree argument for eps
def kt_bettor(
    eps: float = 1e2,
    G: float = 1.0,
) -> OnlineLearner:
    """KT Coin Bettor.

    If dimension is higher than 1, then implements per-coordinate KT coin bettor.
    Unlike other online learners, the initial parameter should be set to zeros in most cases.

    References:
        [Orabona, 2019, Alg. 9.2](https://arxiv.org/abs/1912.13213)

    Args:
        eps (float or Pytree): Initial wealth between 1 and sqrt(T). Defaults to 100.
        G: Estimated Lipschitz constant.

    Returns:
        A `GradientTransformation` object.
    """

    def init_fn(params):
        sum_grad = jtu.tree_map(jnp.zeros_like, params)
        wealth = jtu.tree_map(lambda p: jnp.full_like(p, eps), params)
        return KTBettorState(
            sum_grad=sum_grad,
            wealth=wealth,
            count=jnp.ones([], jnp.int32)
        )
    
    def update_fn(updates, state, params):
        # NOTE: gradient is scaled down by Lipschitz constant.
        updates = tree_scalar_multiply(updates, 1/G)
        count_inc = optax.safe_int32_increment(state.count)
        sum_grad = tree_add(state.sum_grad, updates)
        wealth = tree_subtract(state.wealth, tree_multiply(updates, params))
        new_params = jtu.tree_map(
            lambda St, Wt: - St / count_inc * Wt, sum_grad, wealth)
        return new_params, KTBettorState(
            sum_grad=sum_grad, wealth=wealth, count=count_inc)
    
    return OnlineLearner(init_fn, update_fn)


class BlackboxFTRLState(NamedTuple):
    """FTRL (for blackbox reduction) state."""
    momentum: Updates


def blackbox_ftrl(
    beta: float = 1.0
) -> OnlineLearner:
    """Implements FTRL projected on a unit ball with exponentiated gradients
    equal to beta**-t * gt. (Note that gt is automatically regularized by blackbox reduction.) 

    Args:
        beta: Exponentiation constant between 0 and 1. Defaults to 1.0 (no exponentiation).
    """
    
    assert beta >= 0 and beta <= 1, "beta must be between 0 and 1."

    def init_fn(params):
        return BlackboxFTRLState(momentum=jtu.tree_map(jnp.zeros_like, params))
    
    def update_fn(updates, state, params=None):
        del params
        if beta == 1.0:
            momentum = jtu.tree_map(
                lambda m, g: m - g, state.momentum, updates)
        else:
            momentum = jtu.tree_map(
                lambda m, g: beta*m - (1-beta)*g, state.momentum, updates)
        return tree_normalize(momentum), BlackboxFTRLState(momentum)

    return OnlineLearner(init_fn, update_fn)


# TODO: add sign(s1:t) for output.
class BlackboxReductionState(NamedTuple):
    """Black box reduction state."""
    magnitude_params: Params
    direction_params: Params
    magnitude_state: OptState
    direction_state: OptState


# TODO: implement different scaling for each parameter.
def blackbox_reduction(
    magnitude_learner: OnlineLearner,
    direction_learner: OnlineLearner,
    weight_decay: float = 0.0,
) -> OnlineLearner:
    """Black-box reduction algorithm.

    References:
        [Cutkosky & Orabona, 2018](https://arxiv.org/abs/1802.06293)

    To adapt to exponentiated and regularized loss, we slightly modify the blackbox reduction algorithm.
    Specifically, given exp-reg gradient gt_tilde = beta**-t * (gt + mu*wt), we send gt_tilde to the direction learner,
    but we send the non-exponentiated scalar gradient, namely <gt + mu*wt, wt>, to the parameter-free 1d learner.
    The intuition is that we want the 1d learner to learn the optimal sequence of beta**t * lr. A theoretical support for 
    this modification is yet to be established.

    For computation efficiency, blackbox reduction is automatically wrapped and params are stored.

    Args:
        magnitude_learner: Online learner (typically 1D parameter-free algorithms) for magnitude; learns |xt|.
        direction_learner: Online learner for direction; learns xt/|xt|.
        weight_decay: Regularization constant. Defaults to 0.0 (no regularization).
    """
    
    def init_fn(params):
        zt, xt = util.tree_norm_direction_decomposition(params)
        magnitude_state = magnitude_learner.init(zt)
        direction_state = direction_learner.init(xt)
        return BlackboxReductionState(
            magnitude_params=zt,
            direction_params=xt,
            magnitude_state=magnitude_state,
            direction_state=direction_state
        )
    
    def update_fn(updates, state, params=None):
        del params
        zt, xt = state.magnitude_params, state.direction_params
        params = tree_scalar_multiply(xt, zt)
        st = util.tree_inner_product(updates, xt) + weight_decay * zt
        gt_tilde = jtu.tree_map(
            lambda g, w: g + weight_decay*w, updates, params)
        new_zt, magnitude_state = magnitude_learner.update(
            st, state.magnitude_state, zt)
        new_xt, direction_state = direction_learner.update(
            gt_tilde, state.direction_state, xt)
        new_params = tree_scalar_multiply(new_xt, new_zt)

        return new_params, BlackboxReductionState(
            magnitude_params=new_zt,
            direction_params=new_xt,
            magnitude_state=magnitude_state,
            direction_state=direction_state
        )
    
    return OnlineLearner(init_fn, update_fn)


class NormalizedBlackboxState(NamedTuple):
    """normalized 1d to dimension-free reduction state."""
    sum_st: Updates
    sum_gt: Updates
    base_params: Params
    base_state: OptState
    key: jax.Array


def normalized_blackbox(
    base_learner: OnlineLearner,
    beta: float = 1.0,
    weight_decay: float = 0.0,
    seed: int = 0,
) -> OnlineLearner:
    """One-dimension to dimension-free reduction.

    Args:
        base_learner: 1d learner.
        bet: Exponentiated gradient constant. Defaults to 1.0.
        weight_decay: l2 regularization constant. Defaults to 0.0.
        seed: PRNGKey seed. Defaults to 0.
    """

    def init_fn(params):
        # For now, always initialize the 1d learner with zt=0.
        zero = jnp.zeros([], jnp.float32)
        base_state = base_learner.init(zero)
        return NormalizedBlackboxState(
            sum_st=zero,
            sum_gt=jtu.tree_map(jnp.zeros_like, params),
            base_params=zero,
            base_state=base_state,
            key=jr.PRNGKey(seed),
        )
    
    def update_fn(updates, state, params):
        gt = jtu.tree_map(
            lambda g, w: g + weight_decay*w, updates, params)
        def true_fun(_):
            key, v = util.random_unit_vector(state.key, gt)
            # jax.debug.print("  (debug) random vector v = {v}", v=v)
            st = util.tree_inner_product(gt, v)
            return key, st
        def false_fun(_):
            st = jnp.sign(state.sum_st) * util.tree_inner_product(gt, tree_normalize(state.sum_gt))
            return state.key, st
        key, st = jax.lax.cond(
            util.is_zero_tree(state.sum_gt), true_fun, false_fun, operand=None)
        # jax.debug.print(">>>gt = {g}, normalized gt = {x}", g=gt, x=tree_normalize(state.sum_gt))
        # jax.debug.print(">>>inner = {x}", x=util.tree_inner_product(gt, tree_normalize(state.sum_gt)))
        # jax.debug.print("  (debug) gt = {g}, st = {st}", g=gt, st=st)
        zt, base_state = base_learner.update(st, state.base_state, state.base_params)
        sum_st = tree_add(state.sum_st, st)
        if beta == 1.0:
            sum_gt = tree_add(state.sum_gt, gt)
        else:
            # Since we only use normalized sum_gt, it's ok to use the biased aggregation.
            sum_gt = jtu.tree_map(
                lambda m, g: beta*m + (1-beta)*g, state.sum_gt, gt)
        # jax.debug.print(">>>sum_gt = {s}", s=sum_gt)
        xt = tree_scalar_multiply(tree_normalize(sum_gt), zt*jnp.sign(sum_st))
        return xt, NormalizedBlackboxState(
            sum_st=sum_st,
            sum_gt=sum_gt,
            base_params=zt,
            base_state=base_state,
            key=key
        )
    
    return OnlineLearner(init_fn, update_fn)


# =============== TESTING ===============
def test_online_learner(
    online_learner: OnlineLearner
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

    for i in range(3):
        # print(">>>updating optimizer")
        updates, opt_state = online_learner.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        print(f"round{i+1} new params:\n", params)


if __name__ == "__main__":
    magnitude = kt_bettor(eps=100)
    direction = blackbox_ftrl(beta=1.0)
    blackbox = blackbox_reduction(magnitude, direction, weight_decay=0.0)

    # test_online_learner(magnitude)
    # test_online_learner(wrap_online_learner(magnitude))
    # test_online_learner(direction)
    # test_online_learner(wrap_online_learner(direction))
    # test_online_learner(blackbox)
    # test_online_learner(normalized_blackbox(
    #     base_learner=magnitude, beta=1.0, weight_decay=0.0
    # ))

    ol = normalized_blackbox(
        base_learner=magnitude, beta=1.0, weight_decay=0.0
    )
    tree = {
        'a': [jnp.array(1.), jnp.array(2.)],  # List of arrays
        'b': (jnp.array(3.), jnp.array(4.)),  # Tuple of arrays
        'c': {'d': jnp.array(5.)}  # Nested dictionary with an array
    }
    # state = ol.init(tree)
    # params, new_state = ol.update(tree, state, tree)
    # print(">>>init_state:")
    # print(state)
    # print(">>>update_state")
    # print(new_state)

    import time

    t = time.time()
    print(util.is_zero_tree(tree))
    print(time.time() - t)
    t = time.time()
    print(tree_norm(tree) == 0)
    print(time.time() - t)