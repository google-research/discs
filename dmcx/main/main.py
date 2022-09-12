"""Run Sampler to generate chains."""

from collections.abc import Sequence
from absl import app
import dmcx.model.bernouli as bernouli_model
import dmcx.sampler.randomwalk as randomwalk_sampler
import dmcx.sampler.blockgibbs as blockgibbs_sampler

import os

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
from jax import random
from ml_collections import config_dict
import jax.numpy as jnp
import jax


def load_configs():
  """Loading config vals for main, model and sampler."""

  config_main = config_dict.ConfigDict(
      initial_dictionary=dict(
          parallel=False,
          model='bernouli',
          sampler='gibbs',
          num_samples=100,
          chain_length=100000,
          chain_burnin_length=95000))
  config_model = config_dict.ConfigDict(
      initial_dictionary=dict(dimension=4, init_sigma=1.0))
  config_sampler = config_dict.ConfigDict(
      initial_dictionary=dict(
          adaptive=False,
          target_acceptance_rate=0.234,
          sample_dimension=4,
          num_categories=2,
          random_order=False,
          block_size=2))
  if config_model.dimension != config_sampler.sample_dimension:
    config_model.dimension = config_sampler.sample_dimension
  return config_main, config_model, config_sampler


def get_model(config_main, config_model):
  if config_main.model == 'bernouli':
    return bernouli_model.Bernouli(config_model)
  raise Exception('Please provide a correct model name.')


def get_sampler(config_main, config_sampler):
  if config_main.sampler == 'random_walk':
    return randomwalk_sampler.RandomWalkSampler(config_sampler)
  elif config_main.sampler == 'gibbs':
    return blockgibbs_sampler.BlockGibbsSampler(config_sampler)
  raise Exception('Please provide a correct sampler name.')


def split(arr, n_devices):
  return arr.reshape(n_devices, arr.shape[0] // n_devices, *arr.shape[1:])


def compute_chain(model, chain_length, chain_burnin_lengthgth, sampler_step,
                  state, params, rng_sampler_step, x, n_devices):
  chain = []
  for i in range(chain_length - 1):
    rng_sampler_step_p = jax.random.split(rng_sampler_step, num=n_devices)
    x, state = sampler_step(model, rng_sampler_step_p, x, params, state)
    del rng_sampler_step_p
    rng_sampler_step, _ = jax.random.split(rng_sampler_step)
    if chain_burnin_lengthgth <= i + 1:
      chain.append(x)
  return jnp.array(chain)


def get_sample_mean(samples):
  mean_over_samples = jnp.mean(samples, axis=0)
  return mean_over_samples


def get_sample_variance_unbiased(samples):
  sample_mean = jnp.mean(samples, axis=0, keepdims=True)
  var_over_samples = jnp.sum(
      (samples - sample_mean)**2, axis=0) / (
          samples.shape[0] - 1)
  return var_over_samples


def get_population_mean_and_var(model, params):
  mean_p = model.get_expected_val(params)
  var_p = model.get_var(params)
  return mean_p, var_p


def get_mse(pred, target):
  return jnp.mean((pred - target)**2)


def get_max_error(pred, target):
  return jnp.max((pred - target)**2)


def compute_error(model, params, samples):

  mean_p, var_p = get_population_mean_and_var(model, params)
  mean_s_batch = get_sample_mean(samples)
  avg_mean_error = get_mse(mean_s_batch, mean_p)
  max_mean_error = get_max_error(mean_s_batch, mean_p)
  var_unbiased_s = get_sample_variance_unbiased(samples)
  avg_var_error = get_mse(var_p, var_unbiased_s)
  max_var_error = get_max_error(var_p, var_unbiased_s)

  return avg_mean_error, max_mean_error, avg_var_error, max_var_error


def compute_error_across_chain_and_batch(model, params, samples):

  avg_mean_error, max_mean_error, avg_var_error, max_var_error = compute_error(
      model, params, samples)
  print('Average of mean error over samples of chains: ', avg_mean_error)
  print('Max of mean error over samples of chains: ', max_mean_error)
  print('Average of var error over samples of chains: ', avg_var_error)
  print('Max of var error over samples of chains: ', max_var_error)

  last_samples = samples[:, -1:, :]
  avg_mean_error_last_samples, max_mean_error_last_samples, avg_var_error_last_samples, max_var_error_last_samples = compute_error(
      model, params, last_samples)
  print('Average of mean error of last samples of chains: ',
        avg_mean_error_last_samples)
  print('Max of mean error of last samples of chains: ',
        max_mean_error_last_samples)
  print('Average of var error of last samples of chains: ',
        avg_var_error_last_samples)
  print('Max of var error of last samples of chains: ',
        max_var_error_last_samples)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  rnd = random.PRNGKey(0)
  config_main, config_model, config_sampler = load_configs()
  model = get_model(config_main, config_model)
  sampler = get_sampler(config_main, config_sampler)

  rng_param, rng_x0, rng_sampler, rng_sampler_step = jax.random.split(
      rnd, num=4)
  del rnd
  params = model.make_init_params(rng_param)
  x = model.get_init_samples(rng_x0, config_main.num_samples)
  state = sampler.make_init_state(rng_sampler)

  if not config_main.parallel:
    n_devices = 2
    step_jit = jax.jit(sampler.step, static_argnums=0)
    chain = compute_chain(model, config_main.chain_length,
                          config_main.chain_burnin_length, step_jit, state,
                          params, rng_sampler_step, x, n_devices)
  else:
    n_devices = jax.local_device_count()
    step_pmap = jax.pmap(sampler.step, static_broadcasted_argnums=[0])
    params_pmap = jnp.stack([params] * n_devices)
    state_pmap = jnp.stack([state] * n_devices)
    x_pmap = split(x, n_devices)
    print('Num devices: ', n_devices, ',X shape: ', x_pmap.shape,
          ',Params shape: ', params_pmap.shape)
    chain = compute_chain(model, config_main.chain_length,
                          config_main.chain_burnin_length, step_pmap,
                          state_pmap, params_pmap, rng_sampler_step, x_pmap,
                          n_devices)
    chain = chain.reshape(chain.shape[0], -1, chain.shape[-1])
  print('Samples Shape [Num of Samples, Num of Batch, Sample Dimension]: ',
        jnp.shape(chain))
  compute_error_across_chain_and_batch(model, params, chain)


if __name__ == '__main__':
  app.run(main)
