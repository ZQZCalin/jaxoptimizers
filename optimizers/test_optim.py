import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
import optax
from optax import Updates, Params, OptState, ScalarOrSchedule, GradientTransformation
from typing import Any, Tuple, NamedTuple, Optional, Union, Callable, Protocol
from tqdm import tqdm
import sys
sys.path.append('../jaxoptimizers')
import util
import online_learners as ol
from o2nc import online_nonconvex
import wandb


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
    # print(">>>optimizer initialized")
    
    for i in range(3):
        # print(">>>updating optimizer")
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        print(f"round{i+1} new params:\n", params)
    

if __name__ == "__main__":
    # optimizer = eo2nc_unconstrained_ogd(learning_rate=0.01)
    # optimizer = adamw()
        
    # online_learner = ol.ogd(learning_rate=0.01)

    # online_learner = ol.blackbox_reduction(
    #     magnitude_learner=ol.kt_bettor(eps=10),
    #     direction_learner=ol.blackbox_ftrl(beta=1.0),
    #     weight_decay=0.01,
    # )

    learning_rate = optax.linear_schedule(0.1, 0.01, 10000)

    online_learner = ol.ogd_mirror_descent(
        learning_rate=learning_rate,
        beta=1.0,
        mu=0.0
    )

    optimizer = online_nonconvex(
        online_learner=online_learner,
        random_scaling=lambda key: jr.exponential(key),
        seed=0,
    )
    
    grad_clip = optax.clip_by_global_norm(10.0)
    optimizer = optax.chain(
        grad_clip,
        optax.apply_if_finite(optimizer, 15)
        # optimizer
    )
    # The issue occurs in optax.apply_if_finite()

    test_optimizer(optimizer)