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
      params = model_init_params_fn(
          jax.random.split(rng_param, self.config.num_models)
      )
    if 'mask' not in params:
      params['mask'] = None
    num_samples = self.config.batch_size * self.config.num_models
    x0 = model.get_init_samples(rng_x0, num_samples)
    x0_ess = model.get_init_samples(rng_x0_ess, 1)
    state = sampler_init_state_fn(
        jax.random.split(rng_state, self.config.num_models)
    )
    return params, x0, state, x0_ess

  def _prepare_dict(self, state, n_devices):
    for key in state:
      if state[key] is None:
        continue
      state[key] = jnp.squeeze(jnp.stack([state[key]] * n_devices), axis=1)
    return state

  def _prepare_data(self, params, x, state):
    if self.parallel:
      if self.config.num_models >= jax.local_device_count():
        assert self.config.num_models % jax.local_device_count() == 0
        num_models_per_device = (
            self.config.num_models // jax.local_device_count()
        )
        bshape = (jax.local_device_count(), num_models_per_device)
        x_shape = bshape + (self.config.batch_size,) + self.config_model.shape
      else:
        assert self.config.batch_size % jax.local_device_count() == 0
        batch_size_per_device = (
            self.config.batch_size // jax.local_device_count()
        )
        params = self._prepare_dict(params, jax.local_device_count())
        state = self._prepare_dict(state, jax.local_device_count())
        bshape = (jax.local_device_count(), self.config.num_models)
        x_shape = bshape + (batch_size_per_device,) + self.config_model.shape
    else:
      bshape = (self.config.num_models,)
      x_shape = bshape + (self.config.batch_size,) + self.config_model.shape
    fn_breshape = lambda x: jnp.reshape(x, bshape + x.shape[1:])
    state = jax.tree_map(fn_breshape, state)
    params = jax.tree_map(fn_breshape, params)
    x = jnp.reshape(x, x_shape)

    print('x shape: ', x.shape)
    print('state shape: ', state['steps'].shape)
    key = list(params.keys())[1]
    print('params shape: ', params[key].shape)
    return params, x, state, fn_breshape, bshape

  def _compile_sampler_step(self, step_fn):
    if not self.parallel:
      compiled_step = jax.jit(step_fn)
    else:
      compiled_step = jax.pmap(step_fn)
    return compiled_step

  def _compile_evaluator(self, obj_fn_step):
    if not self.parallel:
      compiled_eval_step = jax.jit(obj_fn_step)
    else:
      compiled_eval_step = jax.pmap(obj_fn_step)
    return compiled_eval_step

  def _get_vmapped_functions(self, sampler, model, evaluator):
    model_init_params_fn = jax.vmap(model.make_init_params)
    sampler_init_state_fn = jax.vmap(sampler.make_init_state)
    step_fn = jax.vmap(functools.partial(sampler.step, model=model))
    obj_fn_step = jax.vmap(
        functools.partial(evaluator.evaluate_step, model=model)
    )
    return (
        model_init_params_fn,
        sampler_init_state_fn,
        step_fn,
        obj_fn_step,
    )

  def _compile_fns(self, step_fn, obj_fn_step, evaluator):
    compiled_step = self._compile_sampler_step(step_fn)
    compiled_eval_step = self._compile_evaluator(obj_fn_step)
    compiled_eval_chain = jax.jit(evaluator.evaluate_chain)
    get_hop = jax.jit(self._get_hop)
    get_mapped_samples = jax.jit(self._get_mapped_samples)
    eval_metric = jax.jit(evaluator.get_eval_metrics)
    return (
        compiled_step,
        compiled_eval_step,
        compiled_eval_chain,
        eval_metric,
        get_hop,
        get_mapped_samples,
    )

  def get_results(self, model, sampler, evaluator, saver):
    self._get_chains_and_evaluations(model, sampler, evaluator, saver)

  def _get_chains_and_evaluations(self, model, sampler, evaluator, saver):
    """Sets up the model and the samlping alg and gets the chain of samples."""
    (
        model_init_params_fn,
        sampler_init_state_fn,
        step_fn,
        obj_fn_step,
    ) = self._get_vmapped_functions(sampler, model, evaluator)
    rnd = jax.random.PRNGKey(0)
    params, x, state, x0_ess = self._initialize_model_and_sampler(
        rnd, model, sampler_init_state_fn, model_init_params_fn
    )
    params, x, state, fn_reshape, breshape = self._prepare_data(
        params, x, state
    )
    compiled_fns = self._compile_fns(
        step_fn, obj_fn_step, evaluator
    )
    self._compute_chain(
        compiled_fns,
        state,
        params,
        rnd,
        x,
        x0_ess,
        saver,
        fn_reshape,
        breshape,
    )

  def _compute_chain(
      self,
      compiled_fns,
      state,
      params,
      rng,
      x,
      x0_ess,
      saver,
      fn_reshape,
      bshape,
  ):
    """Generates the chain of samples."""

    (
        chain,
        acc_ratios,
        hops,
        running_time,
        ess,
        metrics,
        best_ratio,
        init_temperature,
        t_schedule,
        ref_obj,
        sample_mask,
    ) = self._initialize_chain_vars(bshape)

    (
        step_fn,
        eval_step_fn,
        eval_chain_fn,
        eval_metric,
        get_hop,
        get_mapped_samples,
    ) = compiled_fns
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
      if step % self.config.log_every_steps == 0:
        if self.config.evaluator == 'co_eval':
          eval_val = eval_step_fn(samples=new_x, params=params)
          ratio = jnp.max(eval_val, axis=-1).reshape(-1) / ref_obj
          best_ratio = jnp.maximum(ratio, best_ratio)
          sample = best_ratio[sample_mask]
        else:
          sample = get_mapped_samples(new_x, x0_ess)
        chain.append(sample)

      if self.config.get_additional_metrics:
        # avg over all models
        acc = jnp.mean(acc)
        acc_ratios.append(acc)
        # hop avg over batch size and num models
        hops.append(get_hop(x, new_x))
      x = new_x

    if self.config.evaluator == 'ess_eval':
      chain = chain[int(self.config.chain_length * self.config.ess_ratio) :]
      chain = jnp.array(chain)
      ess = eval_chain_fn(chain, rng)
      num_ll_calls = int(state['num_ll_calls'][0])
      metrics = eval_metric(ess, running_time, num_ll_calls)
    else:
      saver.dump_results(best_ratio[sample_mask])

    saver.save_results(acc_ratios, hops, metrics, running_time)

  def _initialize_chain_vars(self, bshape):
    t_schedule = self._build_temperature_schedule(self.config)
    ref_obj = self.config_model.get('ref_obj', None)
    sample_idx = self.config_model.get('sample_idx', None)
    sample_mask = None
    if sample_idx is not None:
        sample_mask = sample_idx >= 0

    chain = []
    acc_ratios = []
    hops = []
    running_time = 0
    ess = None
    metrcis = None
    best_ratio = jnp.ones(self.config.num_models, dtype=jnp.float32) * -1e9
    init_temperature = jnp.ones(bshape, dtype=jnp.float32)

    return (
        chain,
        acc_ratios,
        hops,
        running_time,
        ess,
        metrcis,
        best_ratio,
        init_temperature,
        t_schedule,
        ref_obj,
        sample_mask,
    )

  def _get_mapped_samples(self, samples, x0_ess):
    samples = samples.reshape((-1,) + self.config_model.shape)
    x0_ess = x0_ess.reshape((-1,) + self.config_model.shape)
    return jnp.sum(jnp.abs(samples - x0_ess), -1)

  def _get_hop(self, x, new_x):
    return jnp.sum(abs(x - new_x)) / self.config.batch_size / self.config.num_models


def build_experiment(config: config_dict):
    return Experiment(config)
