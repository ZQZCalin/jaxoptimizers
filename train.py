# from jax.random import PRNGKey, PRNGKeyArray
from util import softmax_cross_entropy, tree_norm, log_optax
from model.gpt import GPT
from loader.c4_loader import get_c4_loader_next_token
import logging
import transformers
import ml_dtypes
from opt import optimizers, transforms, tuners
import mechanic

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu

import optax
import equinox as eqx

from jaxtyping import Array
from jax.random import PRNGKey
# a temporary workaround: will be systematically fixed later
from jax import Array as PRNGKeyArray
from jaxamp import amp, DynamicScalerState, dynamic_scale_value_and_grad
from typing import Tuple, Any, Optional, Sequence, Union, NamedTuple, Callable
from collections import defaultdict
from omegaconf import OmegaConf, DictConfig

import time
import tqdm
import wandb
import hydra





def init_optimizer(
    model: eqx.Module,
    config: DictConfig,
    logger: None,
):
    """Construct optimizer from model and training config.

    Args:
        model (eqx.Module): _description_
        config (DictConfig): _description_
        logger (None): _description_

    Returns:
        _type_: _description_
        _type_: _description_
    """
    optimizer = None
    opt_state = None
    return optimizer, opt_state