"""Utilities."""

import functools
import os
from typing import Any
from absl import logging
from clu import metric_writers
import flax
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class SamplerState:
  step: int
  samples: Any
  sampler_state: Any


def apply_ema(decay, avg, new):
  return jax.tree_map(lambda a, b: decay * a + (1. - decay) * b, avg, new)


def copy_pytree(pytree):
  return jax.tree_map(jnp.array, pytree)


@functools.partial(jax.pmap, axis_name='shard')
def all_gather(x):
  return jax.lax.all_gather(x, 'shard', tiled=True)


def tree_stack(trees):
  """https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75 ."""
  leaves_list = []
  treedef_list = []
  for tree in trees:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    leaves_list.append(leaves)
    treedef_list.append(treedef)

  grouped_leaves = zip(*leaves_list)
  result_leaves = [jnp.stack(l) for l in grouped_leaves]
  return treedef_list[0].unflatten(result_leaves)


def setup_logging(config):
  """Setup logging and writer."""
  if jax.process_index() == 0:
    logging.info(config)
    logging.info('process count: %d', jax.process_count())
    logging.info('device count: %d', jax.device_count())
    logging.info('device/host: %d', jax.local_device_count())
  logdir = config.experiment.save_root
  writer = metric_writers.create_default_writer(
      logdir, just_logging=jax.process_index() > 0)
  fig_folder = os.path.join(logdir, 'figures')
  if jax.process_index() == 0:
    if not os.path.exists(fig_folder):
      os.makedirs(fig_folder)
  config.experiment.fig_folder = fig_folder
  return writer


def shard_to_local_devices(pytree):
  n_local = jax.local_device_count()
  def shard_tensor(x):
    assert x.shape[0] % n_local == 0
    return jnp.reshape(x, [n_local, x.shape[0] // n_local] + list(x.shape[1:]))
  return jax.tree_map(shard_tensor, pytree)


def get_per_process_batch_size(batch_size):
  num_devices = jax.device_count()
  assert (batch_size // num_devices * num_devices == batch_size), (
      'Batch size %d must be divisible by num_devices %d', batch_size,
      num_devices)
  batch_size = batch_size // jax.process_count()
  return batch_size


def shard_prng_key(prng_key):
  # PRNG keys can used at train time to drive stochastic modules
  # e.g. DropOut. We would like a different PRNG key for each local
  # device so that we end up with different random numbers on each one,
  # hence we split our PRNG key and put the resulting keys into the batch
  return jax.random.split(prng_key, num=jax.local_device_count())


class RngGen(object):
  """Random number generator state utility for Jax."""

  def __init__(self, init_rng):
    self._base_rng = init_rng
    self._counter = 0

  def __iter__(self):
    return self

  def __next__(self):
    return self.advance(1)

  def advance(self, count: int):
    self._counter += count
    return jax.random.fold_in(self._base_rng, self._counter)


def create_sharded_sampler_state(sampler_key, model, sampler, num_samples):
  """Create sampler state and shard into local devices."""
  init_rng, sampler_rng = jax.random.split(sampler_key)
  init_samples = model.get_init_samples(init_rng, num_samples=num_samples)
  sampler_rngs = jax.random.split(sampler_rng, jax.local_device_count())
  sampler_states = [sampler.make_init_state(sampler_rng)
                    for sampler_rng in sampler_rngs]
  local_state = SamplerState(
      step=jnp.zeros((jax.local_device_count(),), dtype=jnp.int32),
      samples=shard_to_local_devices(init_samples),
      sampler_state=tree_stack(sampler_states),
  )
  return local_state
