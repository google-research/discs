"""Run Sampler to generate chains."""

from collections.abc import Sequence
from absl import app
from ml_collections import config_dict
import dmcx.model.bernouli as bernouli_model
import dmcx.model.ising as ising_model
import dmcx.model.potts as potts_model
import dmcx.model.categorical as categorical_model
import dmcx.sampler.randomwalk as randomwalk_sampler
import dmcx.sampler.blockgibbs as blockgibbs_sampler
import dmcx.sampler.locallybalanced as locallybalanced_sampler
import dmcx.sampler.gibbswithgrad as gibbswithgrad_sampler
import dmcx.evaluation.evaluator as metric_evaluator
import os

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
import jax.numpy as jnp
import jax
import time

from tqdm import tqdm
import pdb


def load_configs():
  """Loading config vals for main, model and sampler."""

  sample_shape = (10, 10)
  num_categories = 2
  one_hot_rep = True
  if isinstance(sample_shape, int):
    sample_shape = (sample_shape,)
  config_main = config_dict.ConfigDict(
      initial_dictionary=dict(
          parallel=False,
          model='bernouli',
          sampler='random_walk',
          num_samples=48,
          chain_length=1000,
          chain_burnin_length=900,
          sample_shape=sample_shape,
      )
  )
  if config_main.sampler == 'gibbs_with_grad' and num_categories > 2:
    one_hot_rep = True
  else:
    one_hot_rep = False
  config_model = config_dict.ConfigDict(
      initial_dictionary=dict(
          shape=sample_shape,
          init_sigma=1.0,
          lambdaa=0.1,
          external_field_type=0,
          num_categories=num_categories,
          one_hot_representation=one_hot_rep,
      )
  )
  config_sampler = config_dict.ConfigDict(
      initial_dictionary=dict(
          adaptive=False,
          target_acceptance_rate=0.234,
          sample_shape=sample_shape,
          num_categories=num_categories,
          random_order=False,
          block_size=3,
          balancing_fn_type=0,
      )
  )
  config_evaluator = config_dict.ConfigDict(
      initial_dictionary=dict(
          window_size=10,
          window_stride=10,
      )
  )

  return config_main, config_model, config_sampler, config_evaluator


def get_model(config_main, config_model):
  if config_main.model == 'bernouli':
    return bernouli_model.Bernouli(config_model)
  elif config_main.model == 'ising':
    return ising_model.Ising(config_model)
  elif config_main.model == 'categorical':
    return categorical_model.Categorical(config_model)
  elif config_main.model == 'potts':
    return potts_model.Potts(config_model)
  raise Exception('Please provide a correct model name.')


def get_sampler(config_main, config_sampler):
  if config_main.sampler == 'random_walk':
    return randomwalk_sampler.RandomWalkSampler(config_sampler)
  elif config_main.sampler == 'gibbs':
    return blockgibbs_sampler.BlockGibbsSampler(config_sampler)
  elif config_main.sampler == 'locally_balanced':
    return locallybalanced_sampler.LocallyBalancedSampler(config_sampler)
  elif config_main.sampler == 'gibbs_with_grad':
    return gibbswithgrad_sampler.GibbsWithGradSampler(config_sampler)
  raise Exception('Please provide a correct sampler name.')


def compute_chain(
    model,
    chain_length,
    chain_burnin_length,
    sampler_step,
    state,
    params,
    rng_sampler_step,
    x,
    n_rand_split,
):
  """Generates the chain of samples."""
  chain = []
  for _ in tqdm(range(chain_length)):
    rng_sampler_step_p = jax.random.split(rng_sampler_step, num=n_rand_split)
    x, state = sampler_step(model, rng_sampler_step_p, x, params, state)
    del rng_sampler_step_p
    rng_sampler_step, _ = jax.random.split(rng_sampler_step)
    chain.append(x)
  return (
      jnp.array(chain),
      jnp.array(chain[chain_burnin_length:]),
      jnp.array(state),
  )


def split(arr, n_devices):
  return arr.reshape(n_devices, arr.shape[0] // n_devices, *arr.shape[1:])


def initialize_model_and_sampler(rnd, model, sampler, config_main):
  rng_param, rng_x0, rng_sampler = jax.random.split(rnd, num=3)
  del rnd
  params = model.make_init_params(rng_param)
  x0 = model.get_init_samples(rng_x0, config_main.num_samples)
  state = sampler.make_init_state(rng_sampler)
  return params, x0, state


def get_batch_of_chains(rnd, model, sampler, config_main):
  """Sets up the model and the samlping alg and gets the chain of samples."""
  params, x, state = initialize_model_and_sampler(
      rnd, model, sampler, config_main
  )
  model_params = params
  rng_sampler_step, _ = jax.random.split(rnd)
  if not config_main.parallel:
    n_rand_split = 2
    compiled_step = jax.jit(sampler.step, static_argnums=0)
  else:
    n_devices = jax.local_device_count()
    n_rand_split = n_devices
    compiled_step = jax.pmap(sampler.step, static_broadcasted_argnums=[0])
    params = jnp.stack([params] * n_devices)
    state = jnp.stack([state] * n_devices)
    x = split(x, n_devices)
  chain, samples, state = compute_chain(
      model,
      config_main.chain_length,
      config_main.chain_burnin_length,
      compiled_step,
      state,
      params,
      rng_sampler_step,
      x,
      n_rand_split,
  )

  if config_main.parallel:
    chain = chain.reshape(
        (config_main.chain_length, config_main.num_samples)
        + config_main.sample_shape
    )
    samples = samples.reshape(
        (samples.shape[0], config_main.num_samples) + config_main.sample_shape
    )
    state = state[0]

  return chain, samples, state[1], model_params


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # getting the configs
  config_main, config_model, config_sampler, config_evaluator = load_configs()
  model = get_model(config_main, config_model)
  sampler = get_sampler(config_main, config_sampler)
  evaluator = metric_evaluator.Evaluator(config_evaluator)

  # generating samples
  rnd = jax.random.PRNGKey(0)
  start_time = time.time()
  chain, samples, num_loglike_calls, params = get_batch_of_chains(
      rnd, model, sampler, config_main
  )
  running_time = time.time() - start_time

  print('Time took to generate the chain: ', running_time)
  print(
      'Sampler: ',
      config_main.sampler,
      '! Chain Length: ',
      config_main.chain_length,
      '! Burn-in Length: ',
      config_main.chain_burnin_length,
  )
  print(
      (
          'Samples Shape [Num of samples in each chain, Batch Size, Sample'
          ' shape]: '
      ),
      samples.shape,
  )

  # evaluating the samples
  if config_main.model == 'bernouli':
    errors = evaluator.compute_error_across_chain_and_batch(
        model, params, chain
    )
    print(
        'Error over samples: avg mean: {}, max mean: {}, avg var: {}, max var:'
        ' {}! Error over last samples: avg mean: {}, avg var {}'.format(
            errors[0], errors[1], errors[2], errors[3], errors[4], errors[5]
        )
    )
    evaluator.plot_mixing_time_graph_over_chain(
        model, params, chain, config_main
    )
  rnd_ess, _ = jax.random.split(rnd)
  del rnd
  ess_metrcis = evaluator.get_effective_sample_size_metrics(
      rnd_ess,
      samples,
      config_main.chain_length,
      running_time,
      num_loglike_calls,
  )
  print(
      'mean ESS: {}, ESS metrics over: M_H: {}, time: {}, loglike: {}'.format(
          ess_metrcis[0], ess_metrcis[1], ess_metrcis[2], ess_metrcis[3]
      )
  )


if __name__ == '__main__':
  app.run(main)
