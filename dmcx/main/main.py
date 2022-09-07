"""Run Sampler to generate chains."""

from collections.abc import Sequence
from absl import app
import dmcx.model.bernouli as bernouli_model
import dmcx.sampler.randomwalk as randomwalk_sampler
import os

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
from jax import random
from ml_collections import config_dict
import jax.numpy as jnp
import jax


def load_configs():
  """Loading config vals for main, model and sampler."""

  config_main = config_dict.ConfigDict(
      initial_dictionary=dict(
          parallel=False,
          model='ber',
          sampler='RW',
          num_samples=100,
          chain_lenght=5000,
          chain_burnin_len=4500))
  config_model = config_dict.ConfigDict(
      initial_dictionary=dict(dimension=5, init_sigma=1.0))
  config_sampler = config_dict.ConfigDict(
      initial_dictionary=dict(
          adaptive=False,
          ber_target_accept_rate=0.234,
          sample_dimension=5,
          num_categ=2))
  if config_model.dimension != config_sampler.sample_dimension:
    config_model.dimension = config_sampler.sample_dimension
  return config_main, config_model, config_sampler


def get_model(config_main, config_model):
  if config_main.model == 'ber':
    return bernouli_model.Bernouli(config_model)
  raise Exception('Please provide a correct model name.')


def get_sampler(config_main, config_sampler):
  if config_main.sampler == 'RW':
    return randomwalk_sampler.RandomWalkSampler(config_sampler)
  raise Exception('Please provide a correct sampler name.')


def split(arr, n_devices):
  return arr.reshape(n_devices, arr.shape[0] // n_devices, *arr.shape[1:])


def compute_chain(model, chain_lenght, chain_burnin_len, step_jit, state,
                  params, rng_sampler_step, x):
  chain = []
  for i in range(chain_lenght - 1):
    x, state = step_jit(model, rng_sampler_step, x, params, state)
    rng_sampler_step, _ = random.split(rng_sampler_step)
    if chain_burnin_len <= i + 1:
      chain.append(x)
  chain = jnp.swapaxes(jnp.array(chain), axis1=0, axis2=1)
  return chain


def get_sample_mean(samples):
  mean_over_samples = jnp.mean(samples, axis=1)
  return mean_over_samples


def get_sample_variance_unbiased(samples):
  sample_mean = jnp.mean(samples, axis=1, keepdims=True)
  var_over_samples = jnp.sum(
      (samples - sample_mean)**2, axis=1) / (
          samples.shape[1] - 1)
  return var_over_samples


def compute_error(model, params, samples):

  mean_s_batch = get_sample_mean(samples)
  mean_p = model.get_expected_val(params)
  mean_mean_error = jnp.mean((mean_s_batch - mean_p)**2)
  max_mean_error = jnp.max((mean_s_batch - mean_p)**2)
  print('Mean of mean error over all chains: ', mean_mean_error)
  print('Max of mean error over all chains: ', max_mean_error)
  var_unbiased_s = get_sample_variance_unbiased(samples)
  var_p = model.get_var(params)
  mean_var_error = jnp.mean((var_p - var_unbiased_s)**2)
  max_var_error = jnp.max((var_p - var_unbiased_s)**2)
  print('Mean of var error over all chains: ', mean_var_error)
  print('Max of var error over all chains: ', max_var_error)


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
  step_jit = jax.jit(sampler.step, static_argnums=0)
  n_devices = jax.local_device_count()

  if not config_main.parallel:
    chain = compute_chain(model, config_main.chain_lenght,
                          config_main.chain_burnin_len, step_jit, state, params,
                          rng_sampler_step, x)
  else:
    params = jnp.stack([params] * n_devices)
    rng_sampler_step = jax.random.split(rng_sampler_step, num=n_devices)
    x = split(x, n_devices)
    print('Num devices: ', n_devices, ',X shape: ', x.shape, ',Params shape: ',
          params.shape, ',Random Key shape: ', rng_sampler_step.shape)
    compute_chain_p = jax.pmap(
        compute_chain, static_broadcasted_argnums=[0, 1, 2, 3, 4])
    chain = compute_chain_p(model, config_main.chain_lenght,
                            config_main.chain_burnin_len, step_jit, state,
                            params, rng_sampler_step, x)
    chain = chain.reshape(x.shape[0], -1, x.shape[-1])
  print('Samples Shape=', jnp.shape(chain))
  compute_error(model, params, chain)


if __name__ == '__main__':
  app.run(main)
