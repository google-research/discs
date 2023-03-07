"""Sampling loops."""

import functools
import os
from clu import metric_writers
from discs.common import math
from discs.common import utils as discs_utils
from flax import jax_utils

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pickle5 as pickle
import tqdm
import pdb


def build_temperature_schedule(config):
  """Temperature schedule."""

  if config.t_schedule == 'constant':
    schedule = lambda step: step * 0 + config.init_temperature
  elif config.t_schedule == 'linear':
    schedule = optax.linear_schedule(config.init_temperature,
                                     config.final_temperature,
                                     config.chain_length)
  elif config.t_schedule == 'exp_decay':
    schedule = optax.exponential_decay(
        config.init_temperature, config.chain_length, config.decay_rate,
        end_value=config.final_temperature)
  else:
    raise ValueError('Unknown schedule %s' % config.t_schedule)
  return schedule


def solve_dataset(config, global_key, logger,
                  datagen, model, sampler):
  """Do sampling over each instance of the dataset."""
  pdb.set_trace()
  proc_key = jax.random.fold_in(global_key, jax.process_index())
  is_local = jax.process_count() == 1
  assert is_local  # TODO(x): test it for multi-process case
  exp_config = config.experiment

  if exp_config.num_models > jax.local_device_count():
    assert exp_config.num_models % jax.local_device_count() == 0
    num_instances_per_device = exp_config.num_models // jax.local_device_count()
    batch_repeat = 1
  else:
    assert jax.local_device_count() % exp_config.num_models == 0
    batch_repeat = jax.local_device_count() // exp_config.num_models
    num_instances_per_device = 1

  sampler_init_fn = jax.vmap(sampler.make_init_state)
  step_fn = jax.vmap(functools.partial(sampler.step, model=model))
  obj_fn = jax.vmap(model.evaluate)
  bshape = (num_instances_per_device,)
  if jax.local_device_count() == 1:
    step_fn = jax.jit(step_fn)
    obj_fn = jax.jit(obj_fn)
  else:
    bshape = (jax.local_device_count(),) + bshape
    step_fn = jax.pmap(step_fn, axis_name='shard')
    obj_fn = jax.pmap(obj_fn, axis_name='shard')
  if hasattr(model, 'step0') and exp_config.get('temp0_steps', 0):
    step0_fn = jax.vmap(model.step0)
    if jax.local_device_count() == 1:
      step0_fn = jax.jit(step0_fn)
    else:
      step0_fn = jax.pmap(step0_fn, axis_name='shard')
  fn_breshape = lambda x: jnp.reshape(x, bshape + x.shape[1:])
  num_samples = exp_config.num_models * exp_config.batch_size
  sample_bshape = bshape + (num_samples // math.prod(bshape),)

  init_temperature = jnp.ones(bshape, dtype=jnp.float32)
  t_schedule = build_temperature_schedule(exp_config)
  results = {'best_ratio': np.zeros((0,), dtype=np.float32),
             'sample_idx': np.zeros((0,), dtype=np.int32),
             'reference_obj': np.zeros((0,), dtype=np.float32)}
  with metric_writers.ensure_flushes(logger):
    for batch_idx, data_list in enumerate(datagen):
      sample_idx, params, reference_obj = zip(*data_list)
      reference_obj = np.array(reference_obj, dtype=np.float32)
      sample_idx = np.array(sample_idx, dtype=np.int32)
      params = discs_utils.tree_stack(params)
      params = jax.tree_map(
          lambda x: jnp.tile(x, [batch_repeat] + [1] * (x.ndim - 1)), params)
      params = jax.tree_map(fn_breshape, params)
      sample_mask = sample_idx >= 0

      proc_key, _ = jax.random.split(proc_key)
      model_key, sampler_key, rng = jax.random.split(proc_key, 3)
      sampler_state = sampler_init_fn(
          jax.random.split(sampler_key, math.prod(bshape)))
      sampler_state = jax.tree_map(fn_breshape, sampler_state)
      samples = model.get_init_samples(model_key, num_samples=num_samples)
      samples = jnp.reshape(samples, sample_bshape + samples.shape[1:])

      pbar = range(config.experiment.chain_length)
      if is_local and exp_config.use_tqdm:
        pbar = tqdm.tqdm(pbar)
      flush_every = max(1, config.experiment.chain_length // 1000)
      best_ratio = np.ones(exp_config.num_models, dtype=np.float32) * -1e9
      best_samples = np.zeros(
          (exp_config.num_models, samples.shape[-1]), dtype=np.int32)
      trajectory = []
      for step in pbar:
        cur_temp = t_schedule(step)
        params['temperature'] = init_temperature * cur_temp
        rng = jax.random.fold_in(rng, step)
        step_rng = fn_breshape(jax.random.split(rng, math.prod(bshape)))
        samples, sampler_state = step_fn(
            rng=step_rng, x=samples, model_param=params, state=sampler_state,
            x_mask=params['mask'])

        if step % exp_config.log_every_steps == 0:
          step_obj = obj_fn(params, samples)
          step_chosen = jnp.argmax(step_obj, axis=-1, keepdims=True)
          obj = jnp.squeeze(jnp.take_along_axis(step_obj, step_chosen, -1), -1)
          chosen_samples = jnp.squeeze(jnp.take_along_axis(
              samples, jnp.expand_dims(step_chosen, -1), -2), -2)
          obj = jnp.reshape(obj, (batch_repeat, -1,))
          chosen_samples = jnp.reshape(
              chosen_samples, (batch_repeat, -1, samples.shape[-1]))
          rep_chosen = jnp.argmax(obj, axis=0, keepdims=True)
          obj = jnp.take_along_axis(obj, rep_chosen, 0)
          chosen_samples = jnp.take_along_axis(
              chosen_samples, jnp.expand_dims(rep_chosen, -1), 0)
          obj = discs_utils.all_gather(obj)
          chosen_samples = discs_utils.all_gather(chosen_samples)
          if jax.process_index() == 0:
            obj = jax.device_get(jax_utils.unreplicate(obj))
            chosen_samples = jax.device_get(
                jax_utils.unreplicate(chosen_samples))
            ratio = obj / reference_obj
            is_better = ratio > best_ratio
            best_ratio = np.maximum(best_ratio, ratio)
            best_samples = np.where(np.expand_dims(is_better, -1),
                                    chosen_samples, best_samples)
            trajectory.append(best_ratio[sample_mask])
            avg_ratio = np.mean(best_ratio[sample_mask])
            logger.write_scalars(
                step,
                {'temperature@batch%d' % batch_idx: jax.device_get(cur_temp)})
            logger.write_scalars(step, {'ratio@batch%d' % batch_idx: avg_ratio})
            if is_local and exp_config.use_tqdm:
              pbar.set_description(
                  'best ratio: %.4e, temperature: %.4e' % (avg_ratio, cur_temp))
        if step % flush_every == 0:
          logger.flush()
      if exp_config.get('temp0_steps', 0):
        assert batch_repeat == 1
        best_samples = jnp.reshape(jnp.array(best_samples),
                                   bshape + (1,) + best_samples.shape[1:])
        print(best_ratio)
        for _ in range(exp_config.temp0_steps):
          new_obj, best_samples = step0_fn(params, best_samples)
        best_samples = np.reshape(jax.device_get(best_samples),
                                  [exp_config.num_models, -1])
        best_ratio = np.reshape(jax.device_get(new_obj), [-1]) / reference_obj
        print(best_ratio)
        step = exp_config.chain_length + exp_config.temp0_steps
        trajectory.append(best_ratio[sample_mask])
        avg_ratio = np.mean(best_ratio[sample_mask])
        logger.write_scalars(step, {'temperature@batch%d' % batch_idx: 0.0})
        logger.write_scalars(step, {'ratio@batch%d' % batch_idx: avg_ratio})
      trajectory = np.stack(trajectory, axis=1)
      for key, val in [('best_ratio', best_ratio),
                       ('sample_idx', sample_idx),
                       ('reference_obj', reference_obj)]:
        results[key] = np.concatenate([results[key], val[sample_mask]])
      if 'trajectory' not in results:
        results['trajectory'] = trajectory
        results['best_samples'] = best_samples
      else:
        results['trajectory'] = np.concatenate(
            [results['trajectory'], trajectory], axis=0)
        results['best_samples'] = np.concatenate(
            [results['best_samples'], best_samples], axis=0)

  if jax.process_index() == 0:
    with open(os.path.join(exp_config.save_root, 'results.pkl'), 'wb') as f:
      pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
  print('done sampling!')
