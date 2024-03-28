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

import tqdm
import wandb
import hydra
from omegaconf import OmegaConf, DictConfig

import util
from util import softmax_cross_entropy, tree_norm, get_accuracy, get_dtype
import logstate
from logger import TimeKeeper, RateLimitedWandbLog
from model.gpt import GPT
from loader.c4_loader import get_c4_loader_next_token

import sys
sys.path.append('./optimizers')
from optimizers.o2nc import adamw, online_nonconvex
import optimizers.online_learners as ol


divider = "="*100


class TrainState(NamedTuple):
    model: eqx.Module
    opt_state: optax.OptState
    dynamic_scaler_state: Optional[DynamicScalerState]
    iteration: Array


class ExtraTrainState(NamedTuple):
    """Additional train states stored for additional loggings."""
    # High memory costs.
    params_diff: Optional[optax.Updates]        # x_n - x_{n-1} = s_n * Delta_n
    last_grads: Optional[optax.Updates]         # grad_{n-1}
    sum_grads: Optional[optax.Updates]          # sum_{i=1}^{n-1} grad_i
    # Low memory costs.
    random_scalar: Optional[Array]              # s_n
    cumulative_loss_ol: Optional[Array]         # sum_{i=1}^n <grad_i, Delta_i>
    cumulative_loss_optim: Optional[Array]      # sum_{i=1}^n f(x_i, z_i) - f(x_{i-1}, z_i)


def load_c4_data(config: DictConfig, tokenizer: Any, split: str = "train"):
    """Wrapper for C4 dataset.

    Args:
        config (DictConfig): _description_
        tokenizer (Any): _description_
        split (str, optional): _description_. Defaults to "train".

    Returns:
        torch.utils.data.DataLoader: C4 train/test dataloader.
    """
    loader = get_c4_loader_next_token(
        tokenizer,
        split=split,
        batch_size=config.train.batch_size,
        max_length=config.model.context_length,
        pad_to_multiple_of=config.model.context_length,
        num_workers=config.train.dataloader_workers,
        ds_path=config.train.data_path,
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
    max_steps: int,
    config: DictConfig
) -> optax.ScalarOrSchedule:
    """Parses the config and initializes the learning rate scheduler.

    Args:
        max_steps (int): _description_
        config (DictConfig): optimizer / benchmark config.

    Returns:
        optax.ScalarOrSchedule.
    """
    if config.lr_schedule == "constant":
        learning_rate = config.lr
    elif config.lr_schedule == "cosine":
        if config.lr_warmup:
            learning_rate = optax.warmup_cosine_decay_schedule(
                init_value=.0,
                peak_value=config.lr,
                warmup_steps=config.lr_warmup,
                decay_steps=max_steps
            )
        else:
            learning_rate = optax.cosine_decay_schedule(
                init_value=config.lr,
                decay_steps=max_steps
            )
    return learning_rate


def lr_wrapper(
    learning_rate: optax.ScalarOrSchedule,
    count: int,
    logger: None
):
    if callable(learning_rate):
        lr = learning_rate(count)
    else:
        lr = learning_rate
    if logger is not None:
        jax.experimental.io_callback(logger, None, {"lr/schedule": lr}, commit=False)
    return lr


def init_optimizer(
    model: eqx.Module,
    config: DictConfig,
    logger: None,
):
    """Construct optimizer from model and training config.

    Args:
        model: _description_
        config: _description_
        logger: _description_

    Returns:
        _type_: _description_
        _type_: _description_
    """
    max_steps = config.train.max_steps
    gradient_clip_val = config.train.gradient_clip_val
    run_benchmark = config.benchmark.run_benchmark
    
    # Learning rate scheduler.
    lr_config = config.benchmark if run_benchmark else config.optimizer
    learning_rate = init_scheduler(max_steps, lr_config)

    # Wrap scheduler to log learning rate to wandb.
    learning_rate = jtu.Partial(
        lr_wrapper, learning_rate, logger=logger)
    
    if run_benchmark:
        # Run Adamw as the benchmark
        print("====================== NOTE: RUNNING BENCHMARK ======================")
        benchmark_config = config.benchmark
        if benchmark_config.use_default:
            optimizer = optax.adamw(
                learning_rate=learning_rate,
                b1=benchmark_config.beta1,
                b2=benchmark_config.beta2,
                weight_decay=benchmark_config.weight_decay,
            )
        else:
            optimizer = adamw(
                learning_rate=learning_rate,
                beta1=benchmark_config.beta1,
                beta2=benchmark_config.beta2,
                weight_decay=benchmark_config.weight_decay,
                debias_beta1=benchmark_config.debias_beta1,
                debias_beta2=benchmark_config.debias_beta2,
            )
    else:
        # Run online-to-nonconvex conversion.
        print("====================== NOTE: NORMAL O2NC ======================")
        optim_config = config.optimizer
        ol_config = config.optimizer.ol_config
        print(f">>> Online learner: {optim_config.online_learner}")
        if optim_config.online_learner == "ogd_mirror_descent":
            online_learner = ol.ogd_mirror_descent(
                learning_rate=learning_rate,
                beta=ol_config.beta,
                mu=ol_config.mu,
            )
        elif optim_config.online_learner == "ada_ftrl":
            online_learner = ol.ada_ftrl(
                learning_rate=learning_rate,
                **ol_config
            )
        elif optim_config.online_learner == "blackbox_ftrl":
            online_learner = ol.blackbox_reduction(
                magnitude_learner=ol.kt_bettor(eps=ol_config.eps),
                direction_learner=ol.blackbox_ftrl(beta=ol_config.beta),
                weight_decay=ol_config.weight_decay,
            )
        elif optim_config.online_learner == "normalized_blackbox":
            online_learner = ol.normalized_blackbox(
                base_learner=ol.kt_bettor(
                    eps=ol_config.eps, 
                    G=ol_config.Lipschitz,
                    log_reward=True,
                ),
                beta=ol_config.beta,
                weight_decay=ol_config.weight_decay,
                per_layer=ol_config.per_layer,
            )
        else:
            online_learner = ol.ogd(
                learning_rate=learning_rate,
                weight_decay=ol_config.weight_decay
            )

        # print(divider+"\ntesting online learner.......")
        # ol.test_online_learner(online_learner)
        # print("\ntest finished.\n"+divider)

        # Random scaling function.
        exponential_scaling = lambda key: jr.exponential(key)
        uniform_scaling = lambda key: jr.uniform(key, minval=0, maxval=2)

        if optim_config.random_scaling == "exponential":
            random_scaling = exponential_scaling
        elif optim_config.random_scaling == "uniform":
            random_scaling = uniform_scaling
        else:
            random_scaling = exponential_scaling

        # Wrap base online learner with (Exponentiated) O2NC.
        optimizer = online_nonconvex(
            online_learner=online_learner,
            random_scaling=random_scaling,
            seed=optim_config.seed,
        )
        
        # NOTE: for now, we always wrap with the above more specific wrapper.
        # if optim_config.wrap_o2nc == "o2nc":
        #     optimizer = online_nonconvex(
        #         online_learner=online_learner,
        #         random_scaling=random_scaling,
        #         seed=optim_config.optimizer.seed,
        #     )
        # elif config.wrap_o2nc == "eo2nc":
        #     optimizer = eo2nc(online_learner, config.seed)

        # print(divider+"\ntesting wrapped o2nc......")
        # test_optimizer(optimizer)
        # print("finish testing.\n"+divider)

    # Gradient clipping and NaN wrapper.
    grad_clip = optax.clip_by_global_norm(gradient_clip_val)
    optimizer = optax.chain(
        grad_clip,
        optax.apply_if_finite(optimizer, 15)
    )

    # print(divider+"\ntesting final optimizer......")
    # test_optimizer(optimizer)
    # print("finish testing.\n"+divider)

    # Initialize opt_state
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    return optimizer, opt_state


def train_step(
    train_state: TrainState,
    extra_train_state: Optional[ExtraTrainState],
    batch: Tuple[Array, Array],
    optimizer: optax.GradientTransformation,
    key: Array,
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
    extra_logs = util.merge_dicts(*logstate.list_of_logs(opt_state))
    log_data.update(extra_logs)

    # Update new train_state.
    new_train_state = TrainState(
        model=new_model,
        opt_state=opt_state,
        dynamic_scaler_state=dynamic_scaler_state,
        iteration=train_state.iteration + 1
    )

    # Compute additional log statistics.
    if config.log_extra:
        # Compute online learner instantaneous loss: <grad_n, Delta_n>
        last_delta = util.tree_scalar_multiply(
            extra_train_state.params_diff, 1/extra_train_state.random_scalar)
        loss_ol = util.tree_inner_product(grads, last_delta)
        cumulative_loss_ol = extra_train_state.cumulative_loss_ol + loss_ol

        log_data.update({
            "update/online_learner:instant_loss": loss_ol,
            "update/online_learner:cumulative_loss": cumulative_loss_ol,
        })

        # Compute optimization loss gap f(x_n, z_n) - f(x_{n-1}, z_n)
        last_model = eqx.apply_updates(
            model, util.negative_tree(extra_train_state.params_diff))
        last_loss, _ = loss_fn(last_model, batch, key)
        loss_diff = loss - last_loss
        cumulative_loss_optim = extra_train_state.cumulative_loss_optim + loss_diff

        log_data.update({
            "update/optim:loss_diff": loss_diff,
            "update/optim:cumulative_loss": cumulative_loss_optim,
        })

        # Gradient measures: <g_n, g_{n-1}> and <g_n, g_{1:n-1}>
        log_data.update({
            "grads/<gn, g(n-1)>": util.tree_inner_product(grads, extra_train_state.last_grads),
            "grads/<gn, g(1:n-1)>": util.tree_inner_product(grads, extra_train_state.sum_grads),
            "grads/cos(gn, g(n-1))": util.tree_cosine_similarity(grads, extra_train_state.last_grads),
            "grads/cos(gn, g(1:n-1))": util.tree_cosine_similarity(grads, extra_train_state.sum_grads),
        })

        # Update extra_train_state.
        random_scalar = extra_logs['update/random_scaling']
        sum_grads = util.tree_add(extra_train_state.sum_grads, grads)
        new_extra_train_state = ExtraTrainState(
            params_diff=updates,
            last_grads=grads,
            sum_grads=sum_grads,
            random_scalar=random_scalar,
            cumulative_loss_ol=cumulative_loss_ol,
            cumulative_loss_optim=cumulative_loss_optim   
        )
    else:
        new_extra_train_state = None

    return loss, accuracy, log_data, new_train_state, new_extra_train_state


def train_loop(
    train_state: TrainState,
    extra_train_state: Optional[ExtraTrainState],
    optimizer: Any,
    dataloader: Any,
    config: DictConfig,
    time_keeper: TimeKeeper,
    logger: RateLimitedWandbLog,
    key: Array,
):
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

    for it, batch in pbar:
        if it > config.train.max_steps:
            break
        # Load training batch.
        input_ids = jnp.asarray(batch["input_ids"])
        labels = jnp.asarray(batch["labels"])
        to_use, key = jr.split(key)
        tokens = jnp.sum(jnp.asarray(batch["attention_mask"]))
        samples = labels.shape[0]
        time_keeper.mark(end_events={"dataloader": 1}, start_events=["train_step"])

        # Apply one-step train_step.
        loss, accuracy, log_data, train_state, extra_train_state = train_step_jit(
            train_state, extra_train_state, (input_ids, labels), optimizer, key=key
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

    return train_state, key


@hydra.main(version_base=None, config_path="conf", config_name="config_gpt2")
def train(config: DictConfig) -> None:
    logging.info(OmegaConf.to_yaml(config))

    # Initialize C4 dataloader for gpt2.
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    train_loader = load_c4_data(config, tokenizer)

    # Initialize Wandb logging.
    if config.train.wandb_project is not None:
        limited_log = RateLimitedWandbLog(config.train.wandb_logs_per_sec)
        wandb.init(project=config.train.wandb_project)
        wandb.config.update(OmegaConf.to_container(config))
    else:
        limited_log = None

    # Initialize model, optimizer, and loss.
    model = GPT(tokenizer.vocab_size, config.model, key=jr.PRNGKey(42))
    optimizer, opt_state = init_optimizer(model, config, logger=limited_log)

    if config.train.use_amp:
        dynamic_scaler_state = DynamicScalerState()
    else:
        dynamic_scaler_state = None

    train_state = TrainState(
        model=model,
        opt_state=opt_state,
        dynamic_scaler_state=dynamic_scaler_state,
        iteration=jnp.array(0)
    )

    # Extra train state for logging purpose.
    zeros = util.zero_tree(eqx.filter(model, eqx.is_array))
    extra_train_state = None if not config.train.log_extra else ExtraTrainState(
        params_diff=zeros,
        last_grads=zeros,
        sum_grads=zeros,
        random_scalar=jnp.ones([]),
        cumulative_loss_ol=jnp.zeros([]),
        cumulative_loss_optim=jnp.zeros([]),
    )

    key = jr.PRNGKey(0)

    time_keeper = TimeKeeper()

    train_loop(
        train_state,
        extra_train_state,
        optimizer,
        train_loader,
        config,
        logger=limited_log,
        time_keeper=time_keeper,
        key=key,
    )


@hydra.main(version_base=None, config_path="conf", config_name="config_gpt2")
def test(config: DictConfig) -> None:
    # logging.info(OmegaConf.to_yaml(config))

    # Initialize C4 dataloader for gpt2.
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    train_loader = load_c4_data(config, tokenizer)

    # Initialize model, optimizer, and loss.
    model = GPT(tokenizer.vocab_size, config.model, key=jr.PRNGKey(42))
    optimizer, opt_state = init_optimizer(model, config, logger=None)

    params = eqx.filter(model, eqx.is_array)
    updates = jtu.tree_map(jnp.ones_like, params)

    # print(optimizer)
    optimizer.update(updates, opt_state, params)


if __name__ == "__main__":
    train()
    # test()