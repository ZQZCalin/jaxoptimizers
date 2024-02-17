import jax
from jax import numpy as jnp
import equinox as eqx
from jax import random as jr
# from jax.random import PRNGKey, PRNGKeyArray
# also, we are not using PRNGKey directly (but instead use jr.PRNGKey)
from jax.random import PRNGKey
# a temporary workaround: will be systematically fixed later
from jax import Array as PRNGKeyArray
from jaxamp import amp, DynamicScalerState, dynamic_scale_value_and_grad
# Q: why do we import jaxtyping? currently, we don't use jax annotation?
from jaxtyping import Array
from typing import Tuple, Any, Optional, Sequence, Union, NamedTuple, Callable
import hydra
from omegaconf import OmegaConf, DictConfig
import equinox as eqx
import tqdm
from jax import tree_util as jtu
import optax
from util import softmax_cross_entropy, tree_norm, log_optax
from model.gpt import GPT
from loader.c4_loader import get_c4_loader_next_token
import logging
import transformers
import ml_dtypes
import wandb
from collections import defaultdict
import time
from opt import optimizers, transforms, tuners
import mechanic


class TrainState(NamedTuple):
    model: eqx.Module
    opt_state: Any
    dynamic_scaler_state: Optional[DynamicScalerState]
    iteration: Array


def get_accuracy(logits: Array, batch: Tuple[Array], ignore_index: int = -100):
    input, target = batch # [N, L],  [N, L]
    predictions = jnp.argmax(logits, axis=2) # [N, L, C] -> [N, L]
    return jnp.sum(predictions == target) / jnp.sum(target != ignore_index)


def loss_fn(model: eqx.Module, batch: Tuple[Array, Array], key: PRNGKeyArray):
    def single_example_loss_fn(input, target):
        logits = model(input, key=key)
        loss = softmax_cross_entropy(logits, target)
        return loss, logits

    vmapped_loss_fn = jax.vmap(single_example_loss_fn, in_axes=(0, 0), out_axes=(0, 0))
    input, target = batch
    loss, logits = vmapped_loss_fn(input, target)

    return jnp.mean(loss), logits


def get_dtype(dtype: str):
    registry = {
        "bfloat16": ml_dtypes.bfloat16,
        "float16": jnp.float16,
    }

    return registry[dtype.lower()]


def train_step(
    train_state: TrainState,
    batch: Tuple[Array, Array],
    optimizer: optax.GradientTransformation,
    key: PRNGKeyArray,
    config: Any,
):
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

    if config.use_amp:
        dynamic_scaler_state, ((loss, logits), grads) = value_and_grad_fn(
            model, batch, key, dynamic_scaler_state=dynamic_scaler_state
        )
    else:
        (loss, logits), grads = value_and_grad_fn(model, batch, key)
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)

    new_train_state = TrainState(
        model=model,
        opt_state=opt_state,
        dynamic_scaler_state=dynamic_scaler_state,
        iteration=train_state.iteration + 1,
    )

    accuracy = get_accuracy(logits, batch)

    log_data = {"grads/norm": tree_norm(grads)}

    if isinstance(opt_state, optax.contrib.MechanicState):
        log_data["mechanic/s"] = jnp.sum(opt_state.s)

    return loss, accuracy, log_data, new_train_state


class ExpAvg:
    def __init__(self, window_size):
        self.A = 0.0
        self.beta = 1.0 - 1.0 / window_size
        self.count = 0

    def update(self, value):
        self.A = self.A * self.beta + (1.0 - self.beta) * value
        self.count += 1

    @property
    def value(self):
        return self.A / (1.0 - self.beta ** (self.count + 1))


class TimeKeeper:
    def __init__(self, window_size=100):
        self.timestamps = {}
        self.average_durations = defaultdict(lambda: ExpAvg(window_size))
        self.periods = defaultdict(lambda: ExpAvg(window_size))

    def mark(self, start_events=[], end_events={}):
        cur_time = time.time()
        for e, c in end_events.items():
            if c > 0:
                delta = (cur_time - self.timestamps[e]) / c
                self.average_durations[e].update(delta)
        for s in start_events:
            if s in self.timestamps:
                delta = cur_time - self.timestamps[s]
                self.periods[s].update(delta)
            self.timestamps[s] = cur_time

        return cur_time

    def get_durations(self):
        return {k: v.value for k, v in self.average_durations.items()}

    def get_proportions(self):
        return {
            k: self.average_durations[k].value / self.periods[k].value
            for k in self.periods
        }


class RateLimitedWandbLog:
    def __init__(self, max_frequency=1.0):
        self.max_frequency = 1.0
        self.last_time = time.time() - 1.0 / self.max_frequency
        self.metrics = {}

    def __call__(self, metrics, *args, commit=True, **kwargs):
        self.metrics.update(metrics)
        if commit:
            cur_time = time.time()
            if cur_time >= self.last_time + 1.0 / self.max_frequency:
                wandb.log(self.metrics, *args, **kwargs)
                self.last_time = cur_time
                self.metrics = {}


def train_loop(
    train_state: TrainState,
    optimizer: Any,
    dataloader: Any,
    config: DictConfig,
    time_keeper: TimeKeeper,
    logger: RateLimitedWandbLog,
    key: PRNGKeyArray,
):
    pbar = tqdm.tqdm(enumerate(dataloader), total=config.train.max_steps)

    running_loss = 0
    running_accuracy = 0
    total_tokens = 0
    train_step_jit = eqx.filter_jit(
        jtu.Partial(train_step, config=config.train),
    )
    beta = 1.0 - 1.0 / config.train.running_stats_window
    iteration_timing_events = ["iteration", "dataloader", "train_step"]
    time_keeper.mark(start_events=["dataloader", "iteration", "tokens", "samples"])
    for it, batch in pbar:
        if it > config.train.max_steps:
            break
        tokens = jnp.sum(jnp.asarray(batch["attention_mask"]))
        input_ids = jnp.asarray(batch["input_ids"])
        labels = jnp.asarray(batch["labels"])
        samples = labels.shape[0]
        time_keeper.mark(end_events={"dataloader": 1}, start_events=["train_step"])
        to_use, key = jr.split(key)
        loss, accuracy, log_data, train_state = train_step_jit(
            train_state, (input_ids, labels), optimizer, key=key
        )
        time_keeper.mark(
            end_events={"train_step": 1},
        )
        running_loss = beta * running_loss + (1.0 - beta) * loss
        total_tokens += tokens
        running_accuracy = beta * running_accuracy + (1 - beta) * accuracy
        pbar.set_description(
            f"train iter: {it}, tokens: {total_tokens}, loss: {loss}, accuracy: {accuracy}, running_loss: {running_loss/(1.0-beta**(it+1))}, running_accuracy: {running_accuracy/(1.0-beta**(it+1))}"
        )

        metrics = {
            "iterations": train_state.iteration,
            "loss": loss,
            "total_tokens": total_tokens,
            "accuracy": accuracy,
        }
        metrics.update(log_data)

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


def schedule_fn(
    max_iter: int,
    warmup_iter: int,
    peak: float,
    count: int,
    logger: Optional[RateLimitedWandbLog] = None,
):
    result = peak * jax.lax.select(
        count < warmup_iter,
        count / warmup_iter,
        (max_iter - count) / (max_iter - warmup_iter),
    )
    if logger is not None:
        jax.experimental.io_callback(logger, None, {"lr/schedule": result}, commit=False)
    return result


def init_optimizer(
    model: eqx.Module,
    config: DictConfig,
    logger: Optional[RateLimitedWandbLog] = None,
):
    if not config.log_callback_data:
        logger = None
    total_steps = config.max_steps
    schedule = jtu.Partial(
        schedule_fn, total_steps, config.lr_warmup, 1.0, logger=logger
    )

    # bake_schedule = True: set lr schedule before mechanic
    if config.bake_schedule:
        base_schedule = schedule
    else:
        base_schedule = 1.0

    # initialize base optimizer
    # - if bake_momentum = True:
    #    no action needed; we use the original optimizer
    # - if bake_momentum = False: 
    #    we need to remove momentum from base optmizer and add it back after mechanic
    if config.optimizer == "sgd":
        optimizer = optax.chain(
            optax.add_decayed_weights(config.wd),
            optax.sgd(learning_rate=base_schedule, momentum=config.mom),
        )
    elif config.optimizer == "adamw":
        # if config.bake_momentum:
        #     optimizer = optax.adamw(
        #         learning_rate=base_schedule,
        #         weight_decay=config.weight_decay
        #     )
        # else:
        # See opt/adamw.py for more details
        # disable momentum update for mu if bake_momentum is diabled or use_momentum is disabled
        _use_momentum = config.bake_momentum and config.use_momentum
        optimizer = optimizers.adamw(
            learning_rate=base_schedule, 
            weight_decay=config.weight_decay,
            use_momentum=_use_momentum,
            correct_bias=config.correct_bias
        )
            
    # apply mechanic
    if config.mechanize:
        if config.tuner.name == "default": 
            optimizer = optax.contrib.mechanize(optimizer, weight_decay=config.mech_lambda)
        elif config.tuner.name == "centered_mirror_descent":
            tuner = tuners.centered_mirror_descent(**config.tuner.props)
            optimizer = mechanic.mechanize(optimizer, tuner)
        else:
            optimizer = optax.contrib.mechanize(optimizer, weight_decay=config.mech_lambda)

        if logger is not None:

            def log_fn(updates, state, params):
                jax.experimental.io_callback(logger, None, {"mechanic/s": jnp.sum(state.s)}, commit=False)

            optimizer = log_optax(optimizer, log_fn)

    else:
        optimizer = optax.chain(optimizer, optax.scale(config.lr))

    # bake_schedule = False: set lr schedule after mechanic
    if not config.bake_schedule:
        optimizer = optax.chain(optimizer, optax.scale_by_schedule(schedule))
        # if not config.mechanize:
        #     optimizer = optax.chain(optimizer, optax.scale(config.lr))

    # bake_momentum = False: add momentum aggregation after mechanic
    if (not config.bake_momentum) and config.use_momentum:
        optimizer = optax.chain(optimizer, transforms.scale_by_momentum(b=0.9))


    optimizer = optax.apply_if_finite(optimizer, 15)

    # we do gradient clipping before anything else
    grad_clip = optax.clip_by_global_norm(config.gradient_clip_val)
    optimizer = optax.chain(grad_clip, optimizer)

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    return optimizer, opt_state


def load_c4_data(config: DictConfig, tokenizer: Any, split: str = "train"):
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


@hydra.main(version_base=None, config_path="conf", config_name="config_gpt2")
def train(config: DictConfig) -> None:
    logging.info(OmegaConf.to_yaml(config))

    tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    train_loader = load_c4_data(config, tokenizer)

    if config.train.wandb_project is not None:
        limited_log = RateLimitedWandbLog(config.train.wandb_logs_per_sec)
        wandb.init(project=config.train.wandb_project)
        wandb.config.update(OmegaConf.to_container(config))
    else:
        limited_log = None

    model = GPT(tokenizer.vocab_size, config.model, key=jr.PRNGKey(42))
    optimizer, opt_state = init_optimizer(model, config.train, logger=limited_log)

    if config.train.use_amp:
        dynamic_scaler_state = DynamicScalerState()
    else:
        dynamic_scaler_state = None

    train_state = TrainState(
        model=model,
        opt_state=opt_state,
        dynamic_scaler_state=dynamic_scaler_state,
        iteration=jnp.array(0),
    )

    key = jr.PRNGKey(0)

    time_keeper = TimeKeeper()

    train_loop(
        train_state,
        optimizer,
        train_loader,
        config,
        logger=limited_log,
        time_keeper=time_keeper,
        key=key,
    )


if __name__ == "__main__":
    train()