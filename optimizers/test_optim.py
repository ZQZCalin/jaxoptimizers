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
import benchmark
import scheduler
import optim


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


def test_sgdm():
    learning_rate = optax.linear_schedule(0.1, 0.01, 10000)

    optimizer = benchmark.sgdm(learning_rate, beta=1.0, weight_decay=0.0)
    
    grad_clip = optax.clip_by_global_norm(10.0)
    optimizer = optax.chain(
        grad_clip,
        optax.apply_if_finite(optimizer, 15)
    )

    test_optimizer(optimizer)
    

def test_jump():
    learning_rate = scheduler.warmup_linear_decay_schedule(0.0, 3e-4, 500, 5000)


if __name__ == "__main__":
    test_sgdm()