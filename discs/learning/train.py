"""Training tools."""

import abc
import argparse
import os
from typing import Any
from absl import logging
from clu import metric_writers
from discs.common import utils
import discs.learning.raw_torch_dataset as torch_data
import flax
from flax import jax_utils
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import numpy as np
import optax


@flax.struct.dataclass
class TrainState:
  step: int
  params: Any
  opt_state: Any
  ema_params: Any


def init_host_train_state(params, optimizer):
  state = TrainState(
      step=0,
      params=params,
      opt_state=optimizer.init(params),
      ema_params=utils.copy_pytree(params),
  )
  return jax.device_get(state)


def load_numpyds(data_name, data_dir):
  args = argparse.Namespace()
  args.dataset_name = data_name
  args.data_dir = data_dir
  train_data, _, _ = torch_data.load_raw_dataset(args)
  return train_data


def get_data_mean(dataset):
  data_mean = 0
  num_samples = 0
  for x in dataset:
    img = x['image']._numpy().astype(np.float32)  # pylint: disable=protected-access
    data_mean = data_mean + img
    num_samples += 1
  data_mean = np.array(
      np.reshape(data_mean, [-1]), dtype=np.float32) / num_samples / 255.0
  data_mean = np.clip(data_mean, a_min=0.01, a_max=0.99)
  return data_mean.tolist()


def build_lr_schedule(config):
  """Build lr schedule."""
  if config.lr_schedule == 'constant':
    lr_schedule = lambda step: step * 0 + config.learning_rate
  elif config.lr_schedule == 'updown':
    warmup_steps = int(config.warmup_frac * config.total_train_steps)
    lr_schedule = optax.join_schedules([
        optax.linear_schedule(0, config.learning_rate, warmup_steps),
        optax.linear_schedule(config.learning_rate, config.learning_rate * 0.01,
                              config.total_train_steps - warmup_steps)
    ], [warmup_steps])
  elif config.lr_schedule == 'up_exp_down':
    warmup_steps = int(config.warmup_frac * config.total_train_steps)
    lr_schedule = optax.warmup_exponential_decay_schedule(
        init_value=0.0, peak_value=config.learning_rate,
        warmup_steps=warmup_steps, transition_steps=20000,
        decay_rate=0.9, end_value=1e-6
    )
  else:
    raise ValueError('Unknown lr schedule %s' % config.lr_schedule)
  return lr_schedule


def build_optimizer(config):
  """Build optimizer."""
  lr_schedule = build_lr_schedule(config)
  optimizer_name = config.get('optimizer', 'adamw')
  optims = []
  grad_norm = config.get('grad_norm', 0.0)
  if grad_norm > 0.0:
    optims.append(optax.clip_by_global_norm(grad_norm))
  opt_args = {}
  if optimizer_name in ['adamw', 'lamb']:
    opt_args['weight_decay'] = config.get('weight_decay', 0.0)
  optims.append(
      getattr(optax, optimizer_name)(lr_schedule, **opt_args)
  )
  optim = optax.chain(*optims)
  return optim


class Trainer(abc.ABC):
  """Common trainer."""

  def __init__(self, config):
    self.config = config
    self.optimizer = build_optimizer(self.config.experiment)

  def init_states(self, rng):
    model_key, sampler_key = jax.random.split(rng)
    params = self.model.make_init_params(model_key)
    global_state = init_host_train_state(params, self.optimizer)
    num_samples = utils.get_per_process_batch_size(
        self.config.experiment.batch_size)
    local_state = utils.create_sharded_sampler_state(
        sampler_key, self.model, self.sampler, num_samples)
    return global_state, local_state

  @abc.abstractmethod
  def build_loss_func(self, rng, batch):
    pass

  @abc.abstractmethod
  def batch_processing(self, local_state, shared_state, batch_rng, batch):
    pass

  def global_update_fn(self, state, rng, batch):
    """Single gradient update step."""
    params, opt_state = state.params, state.opt_state
    loss_fn = self.build_loss_func(rng, batch)
    (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    grads = jax.lax.pmean(grads, axis_name='shard')
    aux = jax.lax.pmean(aux, axis_name='shard')
    updates, opt_state = self.optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    ema_params = utils.apply_ema(
        decay=jnp.where(state.step == 0, 0.0,
                        self.config.experiment.ema_decay),
        avg=state.ema_params,
        new=params,
    )
    new_state = state.replace(
        step=state.step + 1, params=params, opt_state=opt_state,
        ema_params=ema_params)
    return new_state, aux

  def train_loop(self, logger, global_key,
                 shared_state, local_state, train_loader, fn_plot=None):
    """Train loop."""
    global_update_fn = jax.pmap(self.global_update_fn, axis_name='shard')
    exp_config = self.config.experiment
    lr_schedule = build_lr_schedule(exp_config)
    ckpt_folder = os.path.join(exp_config.save_root, 'ckpts')
    init_step = shared_state.step
    shared_state = jax_utils.replicate(shared_state)
    process_rng_key = jax.random.fold_in(global_key, jax.process_index())

    def save_model(state, step, prefix='checkpoint_', overwrite=False):
      if jax.process_index() == 0:
        host_state = jax.device_get(jax_utils.unreplicate(state))
        checkpoints.save_checkpoint(ckpt_folder, host_state, step, keep=10,
                                    prefix=prefix, overwrite=overwrite)

    with metric_writers.ensure_flushes(logger):
      for step in range(init_step + 1, exp_config.total_train_steps + 1):
        try:
          batch = next(train_loader)
        except StopIteration:
          break
        process_rng_key = jax.random.fold_in(process_rng_key, step)
        process_rng_key, batch_rng_key = jax.random.split(process_rng_key)
        batch, local_state = self.batch_processing(
            local_state, shared_state, batch_rng_key, batch)
        step_rng_keys = utils.shard_prng_key(process_rng_key)
        shared_state, aux = global_update_fn(shared_state, step_rng_keys, batch)
        if step % exp_config.log_every_steps == 0:
          aux = jax.device_get(jax_utils.unreplicate(aux))
          aux['train/lr'] = lr_schedule(step)
          logger.write_scalars(step, aux)
        if step % exp_config.plot_every_steps == 0 and fn_plot is not None:
          fn_plot(step, shared_state, local_state)
        if step % exp_config.save_every_steps == 0:
          save_model(shared_state, step)
    logging.info('done training!')
    return shared_state
