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
          chain_lenght=50,
          sampler_state=3))
  config_model = config_dict.ConfigDict(
      initial_dictionary=dict(dimension=10, init_sigma=1.0))
  config_sampler = config_dict.ConfigDict(
      initial_dictionary=dict(
          adaptive=False, target_accept_ratio=0.234, num_flips=3))

  return config_main, config_model, config_sampler

def get_model(config_main, config_model):
  if config_main.model == 'ber':
    return bernouli_model.Bernouli(config_model)
  raise Exception('Please provide a correct model name.')


def get_sampler(config_main, config_sampler):
  if config_main.sampler == 'RW':
    return randomwalk_sampler.RandomWalkSampler(config_sampler)
  raise Exception('Please provide a correct sampler name.')


@jax.jit
def generate_chains(rnd):
  """Generates chain of samples."""

  config_main, config_model, config_sampler = load_configs()
  model = get_model(config_main, config_model)
  sampler = get_sampler(config_main, config_sampler)

  state = config_main.sampler_state
  rng_param, rng_x0, rng_sampler = random.split(rnd, num=3)
  del rnd
  params = model.make_init_params(rng_param)
  x = model.get_init_samples(rng_x0, config_main.num_samples)
  chain = jnp.expand_dims(x, axis=1)
  print('initial sample shape \n', x.shape)
  for _ in range(config_main.chain_lenght - 1):
    x, state = sampler.step(rng_sampler, x, model, params, state)
    rng_sampler, _ = random.split(rng_sampler)
    chain = jnp.concatenate((chain, jnp.expand_dims(x, axis=1)), axis=1)
  return chain





def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  chain = generate_chains(random.PRNGKey(0))
  print(jnp.shape(chain))


if __name__ == '__main__':
  app.run(main)
