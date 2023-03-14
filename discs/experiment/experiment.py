"""Experiment class that runs sampler on the model to generate chains."""
import jax
import jax.numpy as jnp
from ml_collections import config_dict
from discs.common import math
import tqdm
import pdb
import time
import functools
import optax
import flax


class Experiment:
  """Experiment class that generates chains of samples."""

  def __init__(self, config):
    self.config = config.experiment
    self.config_model = config.model
    self.parallel = False
    if jax.local_device_count() != 1 and self.config.run_parallel:
      self.parallel = True

  def _build_temperature_schedule(self, config):
    """Temperature schedule."""

    if config.t_schedule == 'constant':
      schedule = lambda step: step * 0 + config.init_temperature
    elif config.t_schedule == 'linear':
      schedule = optax.linear_schedule(
          config.init_temperature, config.final_temperature, config.chain_length
      )
    elif config.t_schedule == 'exp_decay':
      schedule = optax.exponential_decay(
          config.init_temperature,
          config.chain_length,
          config.decay_rate,
          end_value=config.final_temperature,
      )
    else:
      raise ValueError('Unknown schedule %s' % config.t_schedule)
    return schedule

  def _initialize_model_and_sampler(
      self, rnd, model, sampler_init_state_fn, model_init_params_fn
  ):
    """Initializes model params, sampler state and gets the initial samples."""
    rng_param, rng_x0, rng_x0_ess, rng_state = jax.random.split(rnd, num=4)
    if self.config_model.get('data_path', None):
      params = self.config_model.params
    elif self.config_model.get('cfg_str', None):
      params = flax.core.frozen_dict.unfreeze(self.config_model.params)
    else:
      params = model_init_params_fn(jax.random.split(rng_param, self.config.num_models))
    if 'mask' not in params:
      params['mask'] = None
    num_samples = self.config.batch_size * self.config.num_models
    x0 = model.get_init_samples(rng_x0, num_samples)
    x0_ess = model.get_init_samples(rng_x0_ess, 1)
    state = sampler_init_state_fn(
        jax.random.split(rng_state, self.config.num_models)
    )
    return params, x0, state, x0_ess


  def _prepare_data(self, params, x, state):
    pdb.set_trace()
    if self.parallel:
        if self.config.num_models > jax.local_device_count():
            assert self.config.num_models % jax.local_device_count() == 0
            num_models_per_device = self.config.num_models // jax.local_device_count()
            bshape = (jax.local_device_count(), num_models_per_device)
        else:
            num_models_per_device = 1
            bshape = (self.config.num_models, num_models_per_device)
    else:
        bshape = (self.config.num_models,)

    x_shape = bshape + (self.config.batch_size,) + self.config_model.shape
    fn_breshape = lambda x: jnp.reshape(x, bshape + x.shape[1:])
    state = jax.tree_map(fn_breshape, state)
    params = jax.tree_map(fn_breshape, params)
    x = jnp.reshape(x, x_shape)

    return params, x, state, fn_breshape, bshape

  def _compile_sampler_step(self, step_fn):
    if not self.parallel:
      compiled_step = jax.jit(step_fn)
    else:
      compiled_step = jax.pmap(step_fn)
    return compiled_step

  def _compile_evaluator(self, obj_fn_step, obj_fn_chain):
    if not self.parallel:
      compiled_eval_step = jax.jit(obj_fn_step)
      compiled_eval_chain = jax.jit(obj_fn_chain)

    else:
      compiled_eval_step = jax.pmap(obj_fn_step)
      compiled_eval_chain = jax.pmap(obj_fn_chain)
    return compiled_eval_step, compiled_eval_chain

  def get_results(self, model, sampler, evaluator, saver):
    num_ll_calls, acc_ratios, hops, evals, running_time, _ = (
        self._get_chains_and_evaluations(model, sampler, evaluator, saver)
    )
    metrcis = evaluator.get_eval_metrics(evals[-1], running_time, num_ll_calls)
    return metrcis, running_time, acc_ratios, hops

  def _get_vmapped_functions(self, sampler, model, evaluator):
    model_init_params_fn = jax.vmap(model.make_init_params)
    sampler_init_state_fn = jax.vmap(sampler.make_init_state)
    step_fn = jax.vmap(functools.partial(sampler.step, model=model))
    obj_fn_step = jax.vmap(
        functools.partial(evaluator.evaluate_step, model=model)
    )
    obj_fn_chain = jax.vmap(evaluator.evaluate_chain)
    return (
        model_init_params_fn,
        sampler_init_state_fn,
        step_fn,
        obj_fn_step,
        obj_fn_chain,
    )

  def _get_chains_and_evaluations(self, model, sampler, evaluator, saver):
    """Sets up the model and the samlping alg and gets the chain of samples."""
    (
        model_init_params_fn,
        sampler_init_state_fn,
        step_fn,
        obj_fn_step,
        obj_fn_chain,
    ) = self._get_vmapped_functions(sampler, model, evaluator)
    rnd = jax.random.PRNGKey(0)
    params, x, state, x0_ess = self._initialize_model_and_sampler(
        rnd, model, sampler_init_state_fn, model_init_params_fn
    )
    model_params = params
    params, x, state, fn_reshape, breshape = self._prepare_data(params, x, state)
    compiled_step = self._compile_sampler_step(step_fn)
    compiled_eval_step, compiled_eval_chain = self._compile_evaluator(
        obj_fn_step, obj_fn_chain
    )
    t_schedule = self._build_temperature_schedule(self.config)
    ref_obj = self.config_model.get('ref_obj', None)
    state, acc_ratios, hops, evals, running_time = self._compute_chain(
        model,
        compiled_step,
        compiled_eval_step,
        compiled_eval_chain,
        state,
        params,
        rnd,
        x,
        x0_ess,
        saver,
        t_schedule,
        fn_reshape,
        breshape,
        ref_obj
    )
    if self.run_parallel:
      num_ll_calls = state['num_ll_calls'][0]
    else:
      num_ll_calls = state['num_ll_calls']
    return num_ll_calls, acc_ratios, hops, evals, running_time, model_params

  def _compute_chain(
      self,
      model,
      step_fn,
      eval_step_fn,
      eval_chain_fn,
      state,
      params,
      rng,
      x,
      x0_ess,
      saver,
      t_schedule,
      fn_reshape,
      bshape,
      ref_obj = None
  ):
    """Generates the chain of samples."""
    chain = []
    acc_ratios = []
    hops = []
    evaluations = []
    trajectory = []
    running_time = 0
    pdb.set_trace()
    init_temperature = jnp.ones(bshape, dtype=jnp.float32)
    for step in tqdm.tqdm(range(self.config.chain_length)):
      cur_temp = t_schedule(step)
      params['temperature'] = init_temperature * cur_temp
      rng = jax.random.fold_in(rng, step)
      step_rng = fn_reshape(jax.random.split(rng, math.prod(bshape)))
      start = time.time()
      new_x, state, acc = step_fn(
          rng=step_rng,
          x=x,
          model_param=params,
          state=state,
          x_mask=params['mask'],
      )
      running_time += time.time() - start
      eval_val = eval_step_fn(samples=new_x, params=params)
      if eval_val != None:
        best_ratio = jnp.max(eval_val, axis = -1).reshape(-1)/ref_obj
        print(jnp.mean(best_ratio))
        eval_val = jnp.mean(eval_val)
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
    samples = samples.reshape((-1,) + self.config_model.shape)
    x0_ess = x0_ess.reshape(x0_ess.shape[0], -1)
    return jnp.sum(jnp.abs(samples - x0_ess), -1)

  def _get_hop(self, x, new_x):
    return jnp.sum(abs(x - new_x)) / self.config.batch_size


def build_experiment(config: config_dict):
  return Experiment(config)
