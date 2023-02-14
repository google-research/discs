"""Experiment class that runs sampler on the model to generate chains."""
import jax
import jax.numpy as jnp
from ml_collections import config_dict
import tqdm
import pdb


class Experiment:
  """Experiment class that generates chains of samples."""

  def __init__(self, config):
    self.config = config

  def _initialize_model_and_sampler(self, rnd, model, sampler):
    rng_param, rng_x0, rng_x0_ess, rng_state = jax.random.split(rnd, num=4)
    params = model.make_init_params(rng_param)
    x0 = model.get_init_samples(rng_x0, self.config.batch_size)
    x0_ess = model.get_init_samples(rng_x0_ess, 1)
    state = sampler.make_init_state(rng_state)
    return params, x0, state, x0_ess

  def _split(self, arr, n_devices):
    return arr.reshape(n_devices, arr.shape[0] // n_devices, *arr.shape[1:])

  def _prepare_state(self, state, n_devices):
    for key in state:
      state[key] = jnp.hstack([state[key]] * n_devices)
    return state

  def _prepare_data_for_parallel(self, params, x, state, n_devices):
    params = jnp.stack([params] * n_devices)
    state = self._prepare_state(state, n_devices)
    x = self._split(x, n_devices)
    return params, x, state

  def _compile_sampler_step(self, sampler):
    if not self.config.run_parallel:
      compiled_step = jax.jit(sampler.step, static_argnums=0)
    else:
      compiled_step = jax.pmap(sampler.step, static_broadcasted_argnums=[0])
    return sampler.step #compiled_step

  def _setup_num_devices(self):
    if not self.config.run_parallel:
      n_rand_split = 2
    else:
      n_rand_split = jax.local_device_count()
    return n_rand_split

  def get_batch_of_chains(self, model, sampler):
    """Sets up the model and the samlping alg and gets the chain of samples."""
    rnd = jax.random.PRNGKey(0)
    params, x, state, x0_ess = self._initialize_model_and_sampler(
        rnd, model, sampler
    )
    model_params = params
    sample_shape = x.shape[1:]
    n_rand_split = self._setup_num_devices()
    if self.config.run_parallel:
      params, x, state = self._prepare_data_for_parallel(
          params, x, state, n_rand_split
      )
    compiled_step = self._compile_sampler_step(sampler)
    chain, state, acc_ratios, hops = self._compute_chain(
        model,
        compiled_step,
        state,
        params,
        rnd,
        x,
        n_rand_split,
        x0_ess,
    )
    if self.config.run_parallel:
      num_ll_calls = state['num_ll_calls'][0]
    else:
      num_ll_calls = state['num_ll_calls']
    return chain, num_ll_calls, acc_ratios, hops, model_params

  def _compute_chain(
      self,
      model,
      sampler_step,
      state,
      params,
      rng_sampler_step,
      x,
      n_rand_split,
      x0_ess,
  ):
    """Generates the chain of samples."""
    chain = []
    acc_ratios = []
    hops = []
    for _ in tqdm.tqdm(range(self.config.chain_length)):
      if self.config.run_parallel:
        rng_sampler_step_p = jax.random.split(
            rng_sampler_step, num=n_rand_split
        )
      else:
        rng_sampler_step_p = rng_sampler_step
      new_x, state, acc = sampler_step(
          model, rng_sampler_step_p, x, params, state
      )
      acc_ratios.append(acc)
      hop = jnp.sum(abs(x - new_x)) / self.config.batch_size
      hops.append(hop)
      x = new_x
      del rng_sampler_step_p
      rng_sampler_step, _ = jax.random.split(rng_sampler_step)
      if self.config.run_parallel:
        new_x = new_x.reshape((self.config.batch_size,) + params[0].shape)
      mapped_x = self._get_mapped_samples(new_x, x0_ess)
      chain.append(mapped_x)
    return (jnp.array(chain), state, jnp.array(acc_ratios), jnp.array(hops))

  def _get_mapped_samples(self, samples, x0_ess):
    samples = samples.reshape(samples.shape[0], -1)
    x0_ess = x0_ess.reshape(x0_ess.shape[0], -1)
    return jnp.sum(jnp.abs(samples - x0_ess), -1)


def build_experiment(config: config_dict):
  return Experiment(config.experiment)
