import chex
import jax
import jax.numpy as jnp
from jax import tree_util as jtu
import optax
from typing import Any, Callable, NamedTuple, Optional, Protocol, Sequence, Tuple, Union


# Family of Adaptive Learning Rates

