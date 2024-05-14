# Train gpt2 model on c4 dataset.
# 
# We will fix our model and dataset and test the 
# performance of different optimizers on this task.
# ===========================================================================


import logging
import transformers

import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu

import optax
import equinox as eqx

from jax import Array
from jaxamp import amp, DynamicScalerState, dynamic_scale_value_and_grad
from typing import Tuple, Any, Optional, Sequence, Union, NamedTuple, Callable
import copy

import tqdm
import wandb
import hydra
from omegaconf import OmegaConf, DictConfig
import argparse

import util
from util import softmax_cross_entropy, tree_norm, get_accuracy, get_dtype
import logstate
from logger import TimeKeeper, RateLimitedWandbLog
from model.gpt import GPT
from loader.lm_loader import get_lm_loader_next_token

import os
import shutil
import json
import sys
sys.path.append('./optimizers')
from optimizers.o2nc import deterministic_online_nonconvex, wrap_random_scaling
import optimizers.online_learners as ol
import optimizers.benchmark as benchmark
import optimizers.scheduler as scheduler
import optimizers.optim as optim


divider = "="*100


def alert_message(msg):
    print(f">>> Alert!: {msg}")


class AuxState(NamedTuple):
    """Auxiliary states stored for additional loggings."""
    # High memory costs.
    params_diff: Optional[optax.Updates]        # x_n - x_{n-1} = s_n * Delta_n
    last_grads: Optional[optax.Updates]         # grad_{n-1}
    sum_grads: Optional[optax.Updates]          # sum_{i=1}^{n-1} grad_i
    # Low memory costs.
    random_scalar: Optional[Array]              # s_n
    cumulative_loss_ol: Optional[Array]         # sum_{i=1}^n <grad_i, Delta_i>
    cumulative_loss_optim: Optional[Array]      # sum_{i=1}^n f(x_i, z_i) - f(x_{i-1}, z_i)
    num_inf_grads: Optional[Array]              # sum_{i=1}^n one(grad_i = nan)


class TrainState(NamedTuple):
    model: eqx.Module
    opt_state: optax.OptState
    dynamic_scaler_state: Optional[DynamicScalerState]
    iteration: Array
    train_key: Array
    aux_state: AuxState


def load_lm_data(config: DictConfig, tokenizer: Any, split: str = "train"):
    """Wrapper for Pile dataset.

    Returns:
        torch.utils.data.DataLoader.
    """
    loader = get_lm_loader_next_token(
        tokenizer,
        split=split,
        batch_size=config.train.batch_size,
        max_length=config.model.context_length,
        shuffle_buffer_size=config.train.shuffle_buffer_size,
        pad_to_multiple_of=config.model.context_length,
        num_workers=config.train.dataloader_workers,
        dataset=config.train.dataset,
    )
    return loader


def loss_fn(model: eqx.Module, batch: Tuple[Array, Array], key: Array):
    """Wrapper for cross entropy loss.

    Args:
        model: equinox module
        batch: _description_
        key: PRNGKeyArray

    Returns:
        Loss value and logits (model outputs).
    """
    def single_example_loss_fn(input, target):
        logits = model(input, key=key)
        loss = softmax_cross_entropy(logits, target)
        return loss, logits

    vmapped_loss_fn = jax.vmap(single_example_loss_fn, in_axes=(0, 0), out_axes=(0, 0))
    input, target = batch
    loss, logits = vmapped_loss_fn(input, target)

    return jnp.mean(loss), logits


def init_scheduler(
    lr_config: DictConfig,
    **kwargs
) -> optax.ScalarOrSchedule:
    """Parses the config and initializes a learning rate scheduler.

    Args:
        lr_config: The learning rate config.
        kargs: Additional arguments to overwrite learning rate config.

    Returns:
        A `optax.ScalarOrSchedule` object.
    """
    # Overwrites default config.
    config = {
        'lr': 0,
        'schedule': 'constant',
        'warmup': 0,
        'max_steps': 0
    }
    config.update(lr_config),
    config.update(kwargs)
    config = OmegaConf.create(config)

    use_warmup = type(config.warmup)==int and (config.warmup > 0)
    if config.schedule == "constant":
        learning_rate = config.lr
    elif config.schedule == "cosine":
        if use_warmup:
            learning_rate = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=config.lr,
                warmup_steps=config.warmup,
                decay_steps=config.max_steps,
            )
        else:
            learning_rate = optax.cosine_decay_schedule(
                init_value=config.lr,
                decay_steps=config.max_steps,
            )
    elif config.schedule == "linear":
        if use_warmup:
            learning_rate = scheduler.warmup_linear_decay_schedule(
                init_value=0.0,
                peak_value=config.lr,
                warmup_steps=config.warmup,
                decay_steps=config.max_steps,
            )
        else:
            learning_rate = scheduler.linear_decay_schedule(
                init_value=config.lr,
                decay_steps=config.max_steps,
            )
    return learning_rate


def wrap_scheduler(
    learning_rate: optax.ScalarOrSchedule,
    logger: None,
    schedule_title: str="schedule",
):
    """Returns a wrapped scheduler that logs current learning rate."""
    def wrapper(schedule, count, logger, title):
        if callable(schedule):
            lr = schedule(count)
        else:
            lr = schedule
        if logger is not None:
            jax.experimental.io_callback(logger, None, {f"lr/{title}": lr}, commit=False)
        return lr
    return jtu.Partial(
        wrapper, learning_rate, logger=logger, title=schedule_title)


def init_optimizer(
    model: eqx.Module,
    config: DictConfig,
    logger: None,
):
    """Construct optimizer from model and training config.

    Returns:
        Initial optimizer and opt_state.
    """
    # Initialize base optimizer / online learner.
    name = config.train.optimizer
    max_steps = config.train.max_steps

    def init_adamw(config: DictConfig, **kwargs):
        """use kwargs to pass down optional arguments (e.g., schedule_title)"""
        learning_rate = wrap_scheduler(
            init_scheduler(config.lr_config), logger=logger, **kwargs)
        return benchmark.adamw(
            learning_rate=learning_rate,
            beta1=config.beta1,
            beta2=config.beta2,
            weight_decay=config.weight_decay,
            debias_beta1=config.debias_beta1,
            debias_beta2=config.debias_beta2,
        )

    def init_sgdm(config: DictConfig, **kwargs):
        learning_rate = wrap_scheduler(
            init_scheduler(config.lr_config), logger=logger, **kwargs)
        return benchmark.sgdm(
            learning_rate=learning_rate,
            beta=config.beta,
            weight_decay=config.weight_decay
        )

    if name == "adamw":
        optimizer = init_adamw(config=config.adamw)
    elif name == "sgdm":
        optimizer = init_sgdm(config=config.sgdm)
    elif name == "polar":
        optimizer = optim.polar_descent(
            direction_optim=init_adamw(config=config.polar.direction),
            magnitude_optim=init_adamw(config=config.polar.magnitude, schedule_title="schedule_2"),
        )
    elif name == "jump":
        optimizer = optim.jump_trajectory(
            normal_optim=init_adamw(config=config.jump.normal),
            jump_optim=init_adamw(config=config.jump.jump, schedule_title="schedule_2"),
            normal_steps=config.jump.normal_steps,
            jump_steps=config.jump.jump_steps,
        )
    elif name == "ogd_md":
        learning_rate = wrap_scheduler(
            init_scheduler(config.ogd_md.lr_config, max_steps=max_steps), logger=logger)
        optimizer = ol.ogd_mirror_descent(
            learning_rate=learning_rate,
            beta=config.ogd_md.beta,
            mu=config.ogd_md.mu,
        )

    # Wrap online to non-convex.
    if name in ["ogd_md"]:
        wrap_o2nc = True
    elif name in ["adamw", "sgdm", "polar", "jump"]:
        wrap_o2nc = False
    else:
        wrap_o2nc = config.train.wrap_o2nc
    if wrap_o2nc:
        optimizer = deterministic_online_nonconvex(optimizer)

    # Wrap random scaling.
    optimizer = wrap_random_scaling(
        gradient_transformation=optimizer,
        random_scaling=config.train.random_scaling,
        seed=config.train.random_scaling_seed
    )
    
    # Gradient clipping and finite gradient wrapper.
    grad_clip = optax.clip_by_global_norm(config.train.gradient_clip_val)
    optimizer = optax.chain(
        grad_clip,
        optax.apply_if_finite(optimizer, 15)
    )

    # Initialize opt_state
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    return optimizer, opt_state


def train_step(
    train_state: TrainState,
    batch: Tuple[Array, Array],
    optimizer: optax.GradientTransformation,
    config: DictConfig,
):
    # Apply auto mixed precision.
    if config.use_amp:
        amp_loss_fn = amp(loss_fn, compute_dtype=get_dtype(config.precision))
        value_and_grad_fn = dynamic_scale_value_and_grad(
            amp_loss_fn, filter=True, has_aux=True, redo_on_nan=0
        )
    else:
        value_and_grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)

    model = train_state.model
    opt_state = train_state.opt_state
    dynamic_scaler_state = train_state.dynamic_scaler_state
    key, new_key = jr.split(train_state.train_key)

    # Apply one-step update.
    if config.use_amp:
        dynamic_scaler_state, ((loss, logits), grads) = value_and_grad_fn(
            model, batch, key, dynamic_scaler_state=dynamic_scaler_state
        )
    else:
        (loss, logits), grads = value_and_grad_fn(model, batch, key)

    # NOTE: it seems that all JAX updates are "immutable", so it's ok to just make a shallow copy as follows.
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    new_model = eqx.apply_updates(model, updates)

    # Compute train accuracy.
    accuracy = get_accuracy(logits, batch)

    # Log basic statistics.
    log_data = {
        "grads/norm": tree_norm(grads),
        "grads/l1-norm": util.tree_l1_norm(grads),
    }
    optim_logs = util.merge_dicts(*logstate.list_of_logs(opt_state))
    log_data.update(optim_logs)

    # Log additional training data.
    log_grads = config.log_config.grads
    log_loss = config.log_config.loss
    log_loss_ol = (config.wrap_o2nc) and log_loss

    # TODO: right now we need a dummy copy of extra_logs, which is kinda dumb.
    # need a more clever way to deal with logs.
    extra_logs_template = {
        "update/online_learner:instant_loss": jnp.zeros([]),
        "update/online_learner:cumulative_loss": jnp.zeros([]),
        "update/optim:loss_diff": jnp.zeros([]),
        "update/optim:cumulative_loss": jnp.zeros([]),
        "grads/<gn, g(n-1)>": jnp.zeros([]),
        "grads/<gn, g(1:n-1)>": jnp.zeros([]),
        "grads/cos(gn, g(n-1))": jnp.zeros([]),
        "grads/cos(gn, g(1:n-1))": jnp.zeros([]),
        "grads/<gn, xn-x(n-1)>": jnp.zeros([]),
        "grads/inf_grads": jnp.zeros([], jnp.int32),
    }

    def log_if_nan(_):
        """Returns a pair of aux_state and extra_logs when grads=nan."""
        aux_state = train_state.aux_state
        extra_logs = copy.deepcopy(extra_logs_template)

        aux_state = aux_state._replace(
            num_inf_grads = optax.safe_int32_increment(aux_state.num_inf_grads)
        )
        return aux_state, extra_logs
    
    def log_if_finite(_):
        """Returns a pair of aux_state and extra_logs when grads is finite."""
        aux_state = train_state.aux_state
        extra_logs = copy.deepcopy(extra_logs_template)

        # Compute online learner instantaneous loss: <grad_n, Delta_n>
        if log_loss_ol:
            last_delta = util.tree_scalar_multiply(
                aux_state.params_diff, 1/aux_state.random_scalar)
            loss_ol = util.tree_inner_product(grads, last_delta)
            cumulative_loss_ol = aux_state.cumulative_loss_ol + loss_ol
            new_random_scalar = optim_logs['update/random_scaling']

            extra_logs.update({
                "update/online_learner:instant_loss": loss_ol,
                "update/online_learner:cumulative_loss": cumulative_loss_ol,
            })
        else:
            cumulative_loss_ol = None
            new_random_scalar = None

        # Compute optimization loss gap f(x_n, z_n) - f(x_{n-1}, z_n)
        if log_loss:
            last_model = eqx.apply_updates(
                model, util.negative_tree(aux_state.params_diff))
            last_loss, _ = loss_fn(last_model, batch, key)
            loss_diff = loss - last_loss
            cumulative_loss_optim = aux_state.cumulative_loss_optim + loss_diff

            extra_logs.update({
                "update/optim:loss_diff": loss_diff,
                "update/optim:cumulative_loss": cumulative_loss_optim,
            })
        else:
            cumulative_loss_optim = None

        # Gradient measures: <g_n, g_{n-1}> and <g_n, g_{1:n-1}>
        if log_grads:
            new_sum_grads = util.tree_add(aux_state.sum_grads, grads)
            extra_logs.update({
                "grads/<gn, g(n-1)>": util.tree_inner_product(grads, aux_state.last_grads),
                "grads/<gn, g(1:n-1)>": util.tree_inner_product(grads, aux_state.sum_grads),
                "grads/cos(gn, g(n-1))": util.tree_cosine_similarity(grads, aux_state.last_grads),
                "grads/cos(gn, g(1:n-1))": util.tree_cosine_similarity(grads, aux_state.sum_grads),
                "grads/<gn, xn-x(n-1)>": util.tree_inner_product(grads, aux_state.params_diff),
                "grads/inf_grads": aux_state.num_inf_grads,
            })
        else:
            new_sum_grads = None

        # Update aux_state.
        aux_state = aux_state._replace(
            params_diff=updates if (log_grads or log_loss) else None,
            last_grads=grads if log_grads else None,
            sum_grads=new_sum_grads,
            random_scalar=new_random_scalar,
            cumulative_loss_ol=cumulative_loss_ol,
            cumulative_loss_optim=cumulative_loss_optim,
        )
        return aux_state, extra_logs
    
    # aux_state, extra_logs = jax.lax.cond(
    #     util.is_finite_tree(grads), log_if_finite, log_if_nan, operand=None)
    aux_state = train_state.aux_state
    extra_logs = {}
    
    # Update new train_state.
    train_state = TrainState(
        model=new_model,
        opt_state=opt_state,
        dynamic_scaler_state=dynamic_scaler_state,
        iteration=train_state.iteration + 1,
        train_key=new_key,
        aux_state=aux_state,
    )
    
    # Merge effective additional_log into log_data
    effective_keys = []
    if log_grads:
        effective_keys += [
            "grads/<gn, g(n-1)>",
            "grads/<gn, g(1:n-1)>",
            "grads/cos(gn, g(n-1))",
            "grads/cos(gn, g(1:n-1))",
            "grads/<gn, xn-x(n-1)>",
            "grads/inf_grads",
        ]
    if log_loss:
        effective_keys += [
            "update/optim:loss_diff",
            "update/optim:cumulative_loss",
        ]
    if log_loss_ol:
        effective_keys += [
            "update/online_learner:instant_loss",
            "update/online_learner:cumulative_loss",
        ]
    log_data.update({
        key: extra_logs[key] for key in effective_keys if key in extra_logs
    })

    return loss, accuracy, log_data, train_state


def save_checkpoint(path: str, train_state: TrainState) -> None:
    """Stores train_state to path."""
    with open(path, 'wb') as f:
        eqx.tree_serialise_leaves(f, train_state)
    print(f"Successfully saved checkpoint to {path}")


def load_checkpoint(path: str, structure: TrainState) -> TrainState:
    """Loads and returns train_state from path."""
    with open(path, 'rb') as f:
        train_state = eqx.tree_deserialise_leaves(f, structure)
    print(f"Successfully loaded checkpoint from {path}")
    return train_state


def train_loop(
    train_state: TrainState,
    optimizer: Any,
    dataloader: Any,
    config: DictConfig,
    time_keeper: TimeKeeper,
    logger: RateLimitedWandbLog,
) -> TrainState:
    pbar = tqdm.tqdm(enumerate(dataloader), total=config.train.max_steps)

    running_loss, running_accuracy, total_tokens = 0, 0, 0
    
    train_step_jit = eqx.filter_jit(
        jtu.Partial(train_step, config=config.train),
    )
    # Just-in-time compilation of the train_step(..., config=config.train) function.
    # [jax.jit](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html)
    # [eqx.filter_jit](https://docs.kidger.site/equinox/api/transformations/#equinox.filter_jit)
    # [jtu.Partial](https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.Partial.html)
    
    # Initialize Wandb Logger
    beta = 1.0 - 1.0 / config.train.running_stats_window
    iteration_timing_events = ["iteration", "dataloader", "train_step"]
    time_keeper.mark(start_events=["dataloader", "iteration", "tokens", "samples"])

    num_steps = config.checkpoint.num_steps if config.checkpoint.path else config.train.max_steps
    start_steps = train_state.iteration     # 0 if not loading from checkpoint
    max_steps = start_steps + num_steps

    for it, batch in pbar:
        if it < start_steps:
            # A dumb way to skip trained data points.
            # TODO: import loadit and for more efficient data loading.
            # right now it takes 50 iter/s to load data (so ~5hr to load 10^6 data).
            continue
        if it > max_steps:
            break
        # Load training batch.
        input_ids = jnp.asarray(batch["input_ids"])
        labels = jnp.asarray(batch["labels"])
        tokens = jnp.sum(jnp.asarray(batch["attention_mask"]))
        samples = labels.shape[0]
        time_keeper.mark(end_events={"dataloader": 1}, start_events=["train_step"])

        # Apply one-step train_step.
        loss, accuracy, log_data, train_state = train_step_jit(
            train_state, (input_ids, labels), optimizer
        )
        time_keeper.mark(
            end_events={"train_step": 1},
        )

        # Update loss and accuracy.
        running_loss = beta * running_loss + (1.0 - beta) * loss
        total_tokens += tokens
        running_accuracy = beta * running_accuracy + (1 - beta) * accuracy
        pbar.set_description(
            f"train iter: {it}, tokens: {total_tokens}, loss: {loss:.2f}, accuracy: {accuracy:.4f}, running_loss: {running_loss/(1.0-beta**(it+1)):.2f}, running_accuracy: {running_accuracy/(1.0-beta**(it+1)):.4f}"
        )

        # ======================================================================
        # BELOW UPDATES ADDITIONAL LOG MESSAGES...
        # Basic states.
        metrics = {
            "iterations": train_state.iteration,
            "loss": loss,
            "total_tokens": total_tokens,
            "accuracy": accuracy,
        }
        metrics.update(log_data)

        # Time complexity related statistics.
        time_keeper.mark(
            start_events=["dataloader", "iteration", "tokens", "samples"],
            end_events={"iteration": 1, "tokens": tokens, "samples": samples},
        )
        durations = time_keeper.get_durations()
        proportions = time_keeper.get_proportions()
        metrics.update(
            {
                f"time/secs_per/{k}": durations[k]
                for k in iteration_timing_events
                if k in durations
            }
        )
        metrics.update(
            {
                f"time/fraction_spent/{k}": proportions[k]
                for k in iteration_timing_events
                if k in proportions
            }
        )

        if "iteration" in durations:
            throughput = {
                "throughput/iteration_per_sec": 1.0 / durations["iteration"],
                "throughput/samples_per_sec": 1.0 / durations["samples"],
                "throughput/tokens_per_sec": 1.0 / durations["tokens"],
            }
            metrics.update(throughput)

        if config.train.wandb_project is not None:
            logger(
                metrics,
                step=train_state.iteration,
            )

        # ======================================================================
        # Saving Checkpoint.
        if config.checkpoint.path and train_state.iteration % config.checkpoint.save_steps == 0:
            checkpoint_path = os.path.join(os.getcwd(), 'checkpoint', config.checkpoint.path, f'{train_state.iteration}.json')
            save_checkpoint(checkpoint_path, train_state)

    return train_state


def init_aux_state(aux: DictConfig, model: eqx.Module) -> AuxState:
    """Initializes aux_state from confg."""
    zeros = util.zero_tree(eqx.filter(model, eqx.is_array))
    return AuxState(
        params_diff = zeros if aux.store_last_params else None,
        last_grads = zeros if aux.store_last_grads else None,
        sum_grads = zeros if aux.store_past_grads else None,
        random_scalar = jnp.ones([]),
        cumulative_loss_ol = jnp.zeros([]),
        cumulative_loss_optim = jnp.zeros([]),
        num_inf_grads = jnp.zeros([], jnp.int32),
    )


def train(config: DictConfig):
    # Initialize C4 dataloader for gpt2.
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    train_loader = load_lm_data(config, tokenizer)

    # Initialize Wandb logging.
    if config.train.wandb_project is not None:
        limited_log = RateLimitedWandbLog(config.train.wandb_logs_per_sec)
        wandb.init(project=config.train.wandb_project)
        wandb.config.update(OmegaConf.to_container(config))
    else:
        limited_log = None

    # Initialize optimizer and train state.
    model = GPT(tokenizer.vocab_size, config.model, key=jr.PRNGKey(42))
    optimizer, opt_state = init_optimizer(model, config, logger=limited_log)
    train_state = TrainState(
        model=model,
        opt_state=opt_state,
        dynamic_scaler_state=DynamicScalerState() if config.train.use_amp else None,
        iteration=jnp.array(0),
        train_key=jr.PRNGKey(0),
        aux_state=init_aux_state(config.train.log_config, model)
    )

    # Load train state from checkpoint.
    if config.checkpoint.path and config.checkpoint.load_model:
        checkpoint_path = os.path.join(os.getcwd(), 'checkpoint', config.checkpoint.path, str(config.checkpoint.load_model)+'.json')
        if not os.path.exists(checkpoint_path):
            raise(f"Error: Checkpoint file {checkpoint_path} does not exist.")
        train_state = load_checkpoint(checkpoint_path, train_state)

    time_keeper = TimeKeeper()

    train_state = train_loop(
        train_state,
        optimizer,
        train_loader,
        config,
        logger=limited_log,
        time_keeper=time_keeper
    )


@hydra.main(version_base=None, config_path="conf", config_name="config_gpt2")
def main(config: DictConfig) -> None:
    """Main training process integrated with checkpoing saving and loading.
    
    Upon loading config, if checkpoint.path!=null, enter checkpoint mode:
        - A new checkpoint directory will be created if checkpoint.path doesn't exist.
        - If config.yaml already exists, it will be loaded and will overwrite the config template beside the checkpoint section.
        - If config.yaml already exists and you want to overwrite it, you can turn on checkpoint.overwrite=True. Use this with caution
          as this overwrites the existing config file.
        - The updated config will be saved to config.yaml.
    If checkpoint.path==null, train the model in the standard way without checkpointing.
    """
    # Config handling.
    if config.checkpoint.path:
        checkpoint_dir = os.path.join(os.getcwd(), 'checkpoint', config.checkpoint.path)
        config_path = os.path.join(checkpoint_dir, 'config.yaml')
        # Create checkpoint directory.
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            print(f"Directory {checkpoint_dir} created.")
        # Load existing config file if exists.
        if os.path.exists(config_path) and not config.checkpoint.overwrite:
            checkpoint_config = config.checkpoint
            config = OmegaConf.load(config_path)
            config.checkpoint = checkpoint_config
        config.checkpoint.overwrite = False     # manually turn off checkpoint.overwrite
        # Update config file.
        with open(config_path, 'w') as f:
            OmegaConf.save(config, f)
        print(f"Config file {config_path} updated.")
        logging.info(OmegaConf.to_yaml(config))
    # Main train function.
    train(config)


# TODO: use loadit chunk_shuffle to load the checkpoint datapoints (accelerated data loading process).
if __name__ == "__main__":
    main()