"""Run Sampler to generate chains."""

from collections.abc import Sequence
from absl import app
import dmcx.model.bernouli as bernouli_model
import dmcx.sampler.randomwalk as randomwalk_sampler
import jax
from jax import random
import jax.numpy as jnp
from ml_collections import config_dict


def load_configs():
  """Loading config vals for main, model and sampler."""

  config_main = config_dict.ConfigDict(
      initial_dictionary=dict(
          model='ber',
          sampler='RW',
          num_samples=100,
          chain_lenght=1000,
          sampler_state=3))
  config_model = config_dict.ConfigDict(
      initial_dictionary=dict(dimension=10, init_sigma=1.0))
  config_sampler = config_dict.ConfigDict(
      initial_dictionary=dict(
          adaptive=False, target_accept_ratio=0.234, sample_dimension=10))

  return config_main, config_model, config_sampler


def get_model(config_main, config_model):
  if config_main.model == 'ber':
    return bernouli_model.Bernouli(config_model)
  raise Exception('Please provide a correct model name.')


def get_sampler(config_main, config_sampler):
  if config_main.sampler == 'RW':
    return randomwalk_sampler.RandomWalkSampler(config_sampler)
  raise Exception('Please provide a correct sampler name.')


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
  chain = []
  chain.append(x)
  for _ in range(config_main.chain_lenght - 1):
    x, state = step_jit(model, rng_sampler_step, x, params, state)
    rng_sampler_step, _ = random.split(rng_sampler_step)
    chain.append(x)
  chain = jnp.swapaxes(jnp.array(chain), axis1=0, axis2=1)
  print(jnp.shape(chain))


if __name__ == '__main__':
  app.run(main)
