"""Run Sampler to generate chains."""

from collections.abc import Sequence
from absl import app
import dmcx.model.bernouli as bernouli_model
import dmcx.sampler.randomwalk as randomwalk_sampler
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
import jax
from jax import random
import jax.numpy as jnp
from ml_collections import config_dict


def load_configs():
  """Loading config vals for main, model and sampler."""

  config_main = config_dict.ConfigDict(
      initial_dictionary=dict(
          parallel=False,
          model='ber',
          sampler='RW',
          num_samples=100,
          chain_lenght=5000,
          chain_burn_in_len=4500))
  config_model = config_dict.ConfigDict(
      initial_dictionary=dict(dimension=5, init_sigma=1.0))
  config_sampler = config_dict.ConfigDict(
      initial_dictionary=dict(
          adaptive=False,
          target_accept_ratio=0.234,
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


def get_sample_mean(samples):
  mean_over_samples = jnp.mean(samples, axis=1)
  mean_over_batch = jnp.mean(mean_over_samples, axis=0)
  return mean_over_batch


def get_sample_variance(samples):
  var_over_samples = jnp.var(samples, axis=1)
  mean_var_over_batch = jnp.mean(var_over_samples, axis=0)
  return mean_var_over_batch


def compute_chain(model, chain_lenght, chain_burn_in_len, step_jit, state,
                  params, rng_sampler_step, x):
  chain = []
  for i in range(chain_lenght - 1):
    x, state = step_jit(model, rng_sampler_step, x, params, state)
    rng_sampler_step, _ = random.split(rng_sampler_step)
    if chain_burn_in_len <= i + 1:
      chain.append(x)
  chain = jnp.swapaxes(jnp.array(chain), axis1=0, axis2=1)
  return chain


def split(arr, n_devices):
  """Splits the first axis of `arr` evenly across the number of devices."""
  return arr.reshape(n_devices, arr.shape[0] // n_devices, *arr.shape[1:])


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
                          config_main.chain_burn_in_len, step_jit, state,
                          params, rng_sampler_step, x)
  else:
    params = jnp.stack([params] * n_devices)
    rng_sampler_step = jax.random.split(rng_sampler_step, num=n_devices)
    x = split(x, n_devices)
    print('Num devices: ', n_devices, ',X shape: ', x.shape, ',Params shape: ',
          params.shape, ',Random Key shape: ', rng_sampler_step.shape)
    compute_chain_p = jax.pmap(
        compute_chain, static_broadcasted_argnums=[0, 1, 2, 3, 4])
    chain = compute_chain_p(model, config_main.chain_lenght,
                            config_main.chain_burn_in_len, step_jit, state,
                            params, rng_sampler_step, x)
    chain = chain.reshape(x.shape[0], -1, x.shape[-1])
  print('Samples Shape=', jnp.shape(chain))
  mean = get_sample_mean(chain)
  var = get_sample_variance(chain)
  print('Sample Mean: ', mean)
  print('Population Mean: ', model.get_expected_val(params))
  print('Sample Var: ', var)
  print('Population Var: ', model.get_var(params))


if __name__ == '__main__':
  app.run(main)
