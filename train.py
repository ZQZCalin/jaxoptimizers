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

import util
from util import softmax_cross_entropy, tree_norm, get_accuracy, get_dtype
import logstate
from logger import TimeKeeper, RateLimitedWandbLog
from model.gpt import GPT
from loader.c4_loader import get_c4_loader_next_token
from loader.pile_loader import get_pile_loader_next_token
from loader.lm_loader import get_lm_loader_next_token

import sys
sys.path.append('./optimizers')
from optimizers.o2nc import online_nonconvex, deterministic_online_nonconvex, wrap_random_scaling
import optimizers.online_learners as ol
import optimizers.benchmark as benchmark
import optimizers.scheduler as scheduler
import optimizers.optim as optim


divider = "="*100


def alert_message(msg):
    print(f">>> Alert!: {msg}")


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
    num_inf_grads: Optional[Array]              # sum_{i=1}^n one(grad_i = nan)


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
        # Note: currently ds_path is NOT passed in the actual data_loader
        # ds_path=config.train.data_path,
    )
    return loader


def load_pile_data(config: DictConfig, tokenizer: Any, split: str = "train"):
    """Wrapper for Pile dataset.

    Args:
        config (DictConfig): _description_
        tokenizer (Any): _description_
        split (str, optional): _description_. Defaults to "train".

    Returns:
        torch.utils.data.DataLoader: Pile train/test dataloader.
    """
    loader = get_pile_loader_next_token(
        tokenizer,
        split=split,
        batch_size=config.train.batch_size,
        max_length=config.model.context_length,
        pad_to_multiple_of=config.model.context_length,
        num_workers=config.train.dataloader_workers,
        # ds_path=config.train.data_path,
    )
    return loader


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
    use_warmup = type(config.warmup)==int
    if config.schedule == "constant":
        learning_rate = config.lr
    elif config.schedule == "cosine":
        if use_warmup:
            learning_rate = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=config.lr,
                warmup_steps=config.warmup,
                decay_steps=max_steps,
            )
        else:
            learning_rate = optax.cosine_decay_schedule(
                init_value=config.lr,
                decay_steps=max_steps,
            )
    elif config.schedule == "linear":
        if use_warmup:
            learning_rate = scheduler.warmup_linear_decay_schedule(
                init_value=0.0,
                peak_value=config.lr,
                warmup_steps=config.warmup,
                decay_steps=max_steps,
            )
        else:
            learning_rate = scheduler.linear_decay_schedule(
                init_value=config.lr,
                decay_steps=max_steps,
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

    Returns:
        Initial optimizer and opt_state.
    """
    max_steps = config.train.max_steps
    gradient_clip_val = config.train.gradient_clip_val
    run_benchmark = config.train.run_benchmark
    
    # Learning rate scheduler.
    lr_config = config.benchmark.lr_config if run_benchmark else config.optimizer.lr_config
    learning_rate = init_scheduler(max_steps, lr_config)

    # Wrap scheduler to log learning rate to wandb.
    learning_rate = jtu.Partial(
        lr_wrapper, learning_rate, logger=logger)
    
    if run_benchmark:
        # Run Adamw as the benchmark
        print("====================== NOTE: RUNNING BENCHMARK ======================")
        benchmark_config = config.benchmark
        if benchmark_config.optim == "adamw":
            adamw_config = benchmark_config.adamw_config
            optimizer = benchmark.adamw(
                learning_rate=learning_rate,
                beta1=adamw_config.beta1,
                beta2=adamw_config.beta2,
                weight_decay=adamw_config.weight_decay,
                debias_beta1=adamw_config.debias_beta1,
                debias_beta2=adamw_config.debias_beta2,
            )
        elif benchmark_config.optim == "sgdm":
            sgdm_config = benchmark_config.sgdm_config
            optimizer = benchmark.sgdm(
                learning_rate=learning_rate,
                beta=sgdm_config.beta,
                weight_decay=sgdm_config.weight_decay,
            )
        elif benchmark_config.optim == "polar":
            polar_config = benchmark_config.polar
            optimizer = optim.polar_descent(
                direction_lr=learning_rate,
                magnitude_lr=learning_rate,
                b1=polar_config.b1,
                b2=polar_config.b2,
                eps=polar_config.eps,
                direction_wd=polar_config.direction_wd,
                magnitude_wd=polar_config.magnitude_wd,
            )
        elif benchmark_config.optim == "jump":
            jump_config = benchmark_config.jump
            normal_lr = jtu.Partial(
                lr_wrapper, 
                scheduler.warmup_linear_decay_schedule(
                    init_value=0.0,
                    peak_value=jump_config.lr1,
                    warmup_steps=0.1*jump_config.normal_steps,
                    decay_steps=jump_config.normal_steps
                ),
                logger=logger
            )
            jump_lr = scheduler.warmup_linear_decay_schedule(
                init_value=0.0,
                peak_value=jump_config.lr2,
                warmup_steps=0.1*jump_config.jump_steps,
                decay_steps=jump_config.jump_steps
            )
            normal_optim = benchmark.adamw(
                normal_lr, 0.9, 0.999, 1e-8, jump_config.wd1
            )
            jump_optim = benchmark.adamw(
                jump_lr, 0.9, 0.999, 1e-8, jump_config.wd2
            )
            optimizer = optim.jump_trajectory(
                normal_optim=normal_optim,
                jump_optim=jump_optim,
                normal_steps=jump_config.normal_steps,
                jump_steps=jump_config.jump_steps
            )
        else:
            # Default optax adamw optimizer.
            alert_message("Benchmark optimzier is not specified. Defaults to optim.adamw.")
            optimizer = optax.adamw(
                learning_rate=learning_rate,
                b1=benchmark_config.beta1,
                b2=benchmark_config.beta2,
                weight_decay=benchmark_config.weight_decay,
            )
    else:
        # Run online-to-nonconvex conversion.
        print("====================== NOTE: RUNNING O2NC ======================")
        optim_config = config.optimizer
        print(f">>> Online learner: {optim_config.online_learner}")
        if optim_config.online_learner == "ogd_mirror_descent":
            ol_config = optim_config.ogd_md_config
            online_learner = ol.ogd_mirror_descent(
                learning_rate=learning_rate,
                beta=ol_config.beta,
                mu=ol_config.mu,
            )
        elif optim_config.online_learner == "ada_ftrl":
            ol_config = optim_config.ada_ftrl_config
            online_learner = ol.ada_ftrl(
                learning_rate=learning_rate,
                **ol_config
            )
        # elif optim_config.online_learner == "blackbox_ftrl":
        #     ol_config = config.optimizer.ol_config
        #     online_learner = ol.blackbox_reduction(
        #         magnitude_learner=ol.kt_bettor(eps=ol_config.eps),
        #         direction_learner=ol.blackbox_ftrl(beta=ol_config.beta),
        #         weight_decay=ol_config.weight_decay,
        #     )
        elif optim_config.online_learner == "normalized_blackbox":
            ol_config = optim_config.kt_blackbox_config
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
            # Default online learner set to OGD.
            alert_message("Online learner is not specified. Defaults to OGD.")
            online_learner = ol.ogd(learning_rate=learning_rate)

        # TODO: deprecate and use separated o2nc and random scaling
        # Random scaling function.
        # exponential_scaling = lambda key: jr.exponential(key)
        # uniform_scaling = lambda key: jr.uniform(key, minval=0, maxval=2)
        # no_scaling = lambda key: jnp.ones([])

        # if optim_config.random_scaling == "none":
        #     random_scaling = no_scaling
        # elif optim_config.random_scaling == "exponential":
        #     random_scaling = exponential_scaling
        # elif optim_config.random_scaling == "uniform":
        #     random_scaling = uniform_scaling
        # else:
        #     print("*** Alert: no randomized scaling is applied!")
        #     random_scaling = no_scaling

        # Wrap base online learner with (Exponentiated) O2NC.
        # optimizer = online_nonconvex(
        #     online_learner=online_learner,
        #     random_scaling=random_scaling,
        #     seed=optim_config.random_scaling_seed,
        # )
        optimizer = deterministic_online_nonconvex(online_learner)

    # Gradient clipping and NaN wrapper.
    optimizer = wrap_random_scaling(
        gradient_transformation=optimizer,
        random_scaling=config.train.random_scaling,
        seed=config.train.random_scaling_seed
    )
    
    grad_clip = optax.clip_by_global_norm(gradient_clip_val)
    optimizer = optax.chain(
        grad_clip,
        optax.apply_if_finite(optimizer, 15)
    )

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
    optim_logs = util.merge_dicts(*logstate.list_of_logs(opt_state))
    log_data.update(optim_logs)

    # Update new train_state.
    new_train_state = TrainState(
        model=new_model,
        opt_state=opt_state,
        dynamic_scaler_state=dynamic_scaler_state,
        iteration=train_state.iteration + 1
    )

    # Log additional training data.
    log_grads = config.log_config.grads
    log_loss = config.log_config.loss
    log_loss_ol = (not config.run_benchmark) and log_loss

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
        """Returns a pair of new_extra_train_state and extra_logs when grads=nan."""
        new_extra_train_state = extra_train_state._replace(
            num_inf_grads = optax.safe_int32_increment(extra_train_state.num_inf_grads)
        )
        # TODO: right now we need a dummy copy of extra_logs, which is kinda dumb.
        # need a more clever way to deal with logs.
        extra_logs = copy.deepcopy(extra_logs_template)
        return new_extra_train_state, extra_logs
    
    def log_if_finite(_):
        """Returns a pair of new_extra_train_state and extra_logs when grads is finite."""
        extra_logs = copy.deepcopy(extra_logs_template)

        # Compute online learner instantaneous loss: <grad_n, Delta_n>
        if log_loss_ol:
            last_delta = util.tree_scalar_multiply(
                extra_train_state.params_diff, 1/extra_train_state.random_scalar)
            loss_ol = util.tree_inner_product(grads, last_delta)
            cumulative_loss_ol = extra_train_state.cumulative_loss_ol + loss_ol
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
                model, util.negative_tree(extra_train_state.params_diff))
            last_loss, _ = loss_fn(last_model, batch, key)
            loss_diff = loss - last_loss
            cumulative_loss_optim = extra_train_state.cumulative_loss_optim + loss_diff

            extra_logs.update({
                "update/optim:loss_diff": loss_diff,
                "update/optim:cumulative_loss": cumulative_loss_optim,
            })
        else:
            cumulative_loss_optim = None

        # Gradient measures: <g_n, g_{n-1}> and <g_n, g_{1:n-1}>
        if log_grads:
            new_sum_grads = util.tree_add(extra_train_state.sum_grads, grads)
            extra_logs.update({
                "grads/<gn, g(n-1)>": util.tree_inner_product(grads, extra_train_state.last_grads),
                "grads/<gn, g(1:n-1)>": util.tree_inner_product(grads, extra_train_state.sum_grads),
                "grads/cos(gn, g(n-1))": util.tree_cosine_similarity(grads, extra_train_state.last_grads),
                "grads/cos(gn, g(1:n-1))": util.tree_cosine_similarity(grads, extra_train_state.sum_grads),
                "grads/<gn, xn-x(n-1)>": util.tree_inner_product(grads, extra_train_state.params_diff),
                "grads/inf_grads": extra_train_state.num_inf_grads,
            })
        else:
            new_sum_grads = None

        # Update extra_train_state.
        new_extra_train_state = extra_train_state._replace(
            params_diff=updates if (log_grads or log_loss) else None,
            last_grads=grads if log_grads else None,
            sum_grads=new_sum_grads,
            random_scalar=new_random_scalar,
            cumulative_loss_ol=cumulative_loss_ol,
            cumulative_loss_optim=cumulative_loss_optim,
        )
        return new_extra_train_state, extra_logs
    
    new_extra_train_state, extra_logs = jax.lax.cond(
        util.is_finite_tree(grads), log_if_finite, log_if_nan, operand=None)
    
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

    # if config.train.dataset == "c4":
    #     train_loader = load_c4_data(config, tokenizer)
    # elif config.train.dataset == "pile":
    #     train_loader = load_pile_data(config, tokenizer)
    train_loader = load_lm_data(config, tokenizer)

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
    # extra_train_state = None if not config.train.log_extra else ExtraTrainState(
    #     params_diff=zeros,
    #     last_grads=zeros,
    #     sum_grads=zeros,
    #     random_scalar=jnp.ones([]),
    #     cumulative_loss_ol=jnp.zeros([]),
    #     cumulative_loss_optim=jnp.zeros([]),
    #     num_inf_grads=jnp.zeros([], jnp.int32),
    # )
    log_grads = config.train.log_config.grads
    log_loss = config.train.log_config.loss
    log_loss_ol = (not config.train.run_benchmark) and log_loss
    zeros = util.zero_tree(eqx.filter(model, eqx.is_array))
    extra_train_state = ExtraTrainState(
        params_diff = zeros if (log_grads or log_loss) else None,
        last_grads = zeros if log_grads else None,
        sum_grads = zeros if log_grads else None,
        random_scalar = jnp.ones([]) if log_loss_ol else None,
        cumulative_loss_ol = jnp.zeros([]) if log_loss_ol else None,
        cumulative_loss_optim = jnp.zeros([]) if log_loss else None,
        num_inf_grads = jnp.zeros([], jnp.int32),
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


if __name__ == "__main__":
    train()