"""Run Sampler to generate chains."""

from collections.abc import Sequence
from absl import app
import dmcx.model.bernouli as bernouli_model
import dmcx.model.ising as ising_model
import dmcx.sampler.randomwalk as randomwalk_sampler
import dmcx.sampler.blockgibbs as blockgibbs_sampler
import dmcx.sampler.locallybalanced as locallybalanced_sampler
import os

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
from jax import random
from ml_collections import config_dict
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb


def load_configs():
  """Loading config vals for main, model and sampler."""

  config_main = config_dict.ConfigDict(
      initial_dictionary=dict(
          parallel=False,
          model='ising',
          sampler='locally_balanced',
          num_samples=50,
          chain_length=1000,
          chain_burnin_length=950,
          window_size=10,
          window_stride=10))
  config_model = config_dict.ConfigDict(
      initial_dictionary=dict(
          shape=(5, 5),
          init_sigma=1.0,
          lamda=0.1,
          external_field_type=0,
          parallel_sampling=False))
  config_sampler = config_dict.ConfigDict(
      initial_dictionary=dict(
          adaptive=False,
          target_acceptance_rate=0.234,
          sample_shape=(5, 5),
          num_categories=2,
          random_order=False,
          block_size=3,
          balancing_fn_type=1))
  if config_model.shape != config_sampler.sample_shape:
    config_model.shape = config_sampler.sample_shape
  return config_main, config_model, config_sampler


def get_model(config_main, config_model):
  if config_main.model == 'bernouli':
    return bernouli_model.Bernouli(config_model)
  elif config_main.model == 'ising':
    return ising_model.Ising(config_model)
  raise Exception('Please provide a correct model name.')


def get_sampler(config_main, config_sampler):
  if config_main.sampler == 'random_walk':
    return randomwalk_sampler.RandomWalkSampler(config_sampler)
  elif config_main.sampler == 'gibbs':
    return blockgibbs_sampler.BlockGibbsSampler(config_sampler)
  elif config_main.sampler == 'locally_balanced':
    return locallybalanced_sampler.LocallyBalancedSampler(config_sampler)
  raise Exception('Please provide a correct sampler name.')


def split(arr, n_devices):
  return arr.reshape(n_devices, arr.shape[0] // n_devices, *arr.shape[1:])


def compute_chain(model, chain_length, chain_burnin_length, sampler_step, state,
                  params, rng_sampler_step, x, n_devices):
  chain = []
  for _ in tqdm(range(chain_length)):
    rng_sampler_step_p = jax.random.split(rng_sampler_step, num=n_devices)
    x, state = sampler_step(model, rng_sampler_step_p, x, params, state)
    del rng_sampler_step_p
    rng_sampler_step, _ = jax.random.split(rng_sampler_step)
    chain.append(x)
  return jnp.array(chain), jnp.array(chain[chain_burnin_length:])


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

  last_samples = jnp.expand_dims(samples[-1], axis=1)
  print('Last Sample Shape: ', last_samples.shape)
  avg_mean_error_last_samples, _, avg_var_error_last_samples, _ = compute_error(
      model, params, last_samples)
  print('mean error of chain of last samples: ', avg_mean_error_last_samples)
  print('var error of chain of last samples: ', avg_var_error_last_samples)


def get_mixing_time_graph_over_chain(model, params, chain, window_size,
                                     window_stride, config_main):

  mean_errors = []
  max_mean_errors = []
  for start in range(0, len(chain), window_stride):
    if (len(chain) - start) < window_size:
      break
    samples = chain[start:start + window_size]
    # print(jnp.shape(samples))
    avg_mean_error, max_mean_error, _, _ = compute_error(model, params, samples)
    # print(avg_mean_error)
    mean_errors.append(avg_mean_error)
    max_mean_errors.append(max_mean_error)
  plt.plot(jnp.arange(1, 1 + len(mean_errors)), mean_errors, '--bo')
  plt.xlabel('Iteration Step Over Chain')
  plt.ylabel('Avg Mean Error')
  plt.title('Avg Mean Error Over Chains for {}!'.format(config_main.sampler))
  plt.savefig('MixingTimeAvgMean_{}'.format(config_main.sampler))
  plt.clf()
  plt.plot(jnp.arange(1, 1 + len(max_mean_errors)), max_mean_errors, '--bo')
  plt.xlabel('Iteration Step Over Chain')
  plt.ylabel('Max Mean Error')
  plt.title('Max Mean Error Over Chains for {}!'.format(config_main.sampler))
  plt.savefig('MixingTimeMaxMean_{}'.format(config_main.sampler))


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
    chain, samples = compute_chain(model, config_main.chain_length,
                                   config_main.chain_burnin_length, step_jit,
                                   state, params, rng_sampler_step, x,
                                   n_devices)
  else:
    n_devices = jax.local_device_count()
    step_pmap = jax.pmap(sampler.step, static_broadcasted_argnums=[0])
    params_pmap = jnp.stack([params] * n_devices)
    state_pmap = jnp.stack([state] * n_devices)
    x_pmap = split(x, n_devices)
    print('Num devices: ', n_devices, ',X shape: ', x_pmap.shape,
          ',Params shape: ', params_pmap.shape)
    chain, samples = compute_chain(model, config_main.chain_length,
                                   config_main.chain_burnin_length, step_pmap,
                                   state_pmap, params_pmap, rng_sampler_step,
                                   x_pmap, n_devices)
    if isinstance(config_sampler.sample_shape, int):
      sample_shape = (config_sampler.sample_shape,)
    else:
      sample_shape = config_sampler.sample_shape
    chain = chain.reshape((config_main.chain_length, config_main.num_samples) +
                          sample_shape)
    samples = samples.reshape((samples.shape[0], config_main.num_samples) +
                              sample_shape)

  print('Sampler: ', config_main.sampler, '! Chain Length: ',
        config_main.chain_length, '! Burn-in Length: ',
        config_main.chain_burnin_length)
  print(
      'Samples Shape [Num of samples in each chain, Batch Size, Sample shape]: ',
      samples.shape)
  compute_error_across_chain_and_batch(model, params, chain)
  get_mixing_time_graph_over_chain(model, params, chain,
                                   config_main.window_size,
                                   config_main.window_stride, config_main)


if __name__ == '__main__':
  app.run(main)
