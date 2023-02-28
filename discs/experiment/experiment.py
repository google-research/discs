"""Experiment class that runs sampler on the model to generate chains."""
import jax
import jax.numpy as jnp
from ml_collections import config_dict
import tqdm
import pdb
import flax
import time


class Experiment:
  """Experiment class that generates chains of samples."""

  def __init__(self, config):
    self.config = config.experiment
    self.config_model = config.model

  def _initialize_model_and_sampler(self, rnd, model, sampler):
    rng_param, rng_x0, rng_x0_ess, rng_state = jax.random.split(rnd, num=4)
    if not self.config_model.get('data_path', None):
      params = model.make_init_params(rng_param)
    else:
      params = flax.core.frozen_dict.freeze(self.config_model.params)
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
    return compiled_step

  def _compile_evaluator(self, evaluator):
    if not self.config.run_parallel:
      compiled_eval_step = jax.jit(evaluator.evaluate_step, static_argnums=1)
      compiled_eval_chain = jax.jit(evaluator.evaluate_chain)

    else:
      compiled_eval_step = jax.pmap(evaluator.evaluate_step)
      compiled_eval_chain = jax.pmap(evaluator.evaluate_chain, static_broadcasted_argnums=[1])
    return compiled_eval_step, compiled_eval_chain

  def _setup_num_devices(self):
    if not self.config.run_parallel:
      n_rand_split = 2
    else:
      n_rand_split = jax.local_device_count()
    return n_rand_split

  def get_results(self, model, sampler, evaluator):
    num_ll_calls, acc_ratios, hops, evals, running_time, _ = (
        self._get_chains_and_evaluations(model, sampler, evaluator)
    )
    metrcis = evaluator.get_eval_metrics(evals[-1], running_time, num_ll_calls)
    return metrcis, running_time, acc_ratios, hops

  def _get_chains_and_evaluations(self, model, sampler, evaluator):
    """Sets up the model and the samlping alg and gets the chain of samples."""
    rnd = jax.random.PRNGKey(0)
    params, x, state, x0_ess = self._initialize_model_and_sampler(
        rnd, model, sampler
    )
    model_params = params
    n_rand_split = self._setup_num_devices()
    if self.config.run_parallel:
      params, x, state = self._prepare_data_for_parallel(
          params, x, state, n_rand_split
      )
    compiled_step = self._compile_sampler_step(sampler)
    compiled_eval_step, compiled_eval_chain = self._compile_evaluator(evaluator)
    state, acc_ratios, hops, evals, running_time = self._compute_chain(
        model,
        compiled_step,
        compiled_eval_step,
        compiled_eval_chain,
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
    return num_ll_calls, acc_ratios, hops, evals, running_time, model_params

  def _compute_chain(
      self,
      model,
      sampler_step,
      eval_step_fn,
      eval_chain_fn,
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
    evaluations = []
    running_time = 0
    for _ in tqdm.tqdm(range(self.config.chain_length)):
      if self.config.run_parallel:
        rng_sampler_step_p = jax.random.split(
            rng_sampler_step, num=n_rand_split
        )
      else:
        rng_sampler_step_p = rng_sampler_step

      start = time.time()
      new_x, state, acc = sampler_step(
          model, rng_sampler_step_p, x, params, state
      )
      running_time += time.time() - start
      del rng_sampler_step_p
      rng_sampler_step, _ = jax.random.split(rng_sampler_step)
      eval_val = eval_step_fn(new_x, model, params)
      if eval_val:
        evaluations.append(eval_val)
      acc_ratios.append(acc)
      hops.append(self._get_hop(x, new_x))
      chain.append(self._get_mapped_samples(new_x, x0_ess))
      x = new_x
    chain = chain[int(self.config.chain_length * self.config.ess_ratio) :]
    chain = jnp.array(chain)
    eval_val = eval_chain_fn(chain, rng_sampler_step)
    if eval_val:
      evaluations.append(eval_val)
    return (
        state,
        jnp.array(acc_ratios),
        jnp.array(hops),
        jnp.array(evaluations),
        running_time,
    )

  def _get_mapped_samples(self, samples, x0_ess):
    samples = samples.reshape((self.config.batch_size, -1))
    x0_ess = x0_ess.reshape(x0_ess.shape[0], -1)
    return jnp.sum(jnp.abs(samples - x0_ess), -1)

  def _get_hop(self, x, new_x):
    return jnp.sum(abs(x - new_x)) / self.config.batch_size


def build_experiment(config: config_dict):
  return Experiment(config)
