"""Experiment class that runs sampler on the model to generate chains."""
import functools
import importlib
import pdb
import time
from discs.common import math_util as math
from discs.common import utils
import flax
import jax
import jax.numpy as jnp
from ml_collections import config_dict
import numpy as np
import optax
import tqdm


class Experiment:
  """Experiment class that generates chains of samples."""

  def __init__(self, config):
    self.config = config.experiment
    self.config_model = config.model
    self.parallel = False
    self.sample_idx = None
    self.num_saved_samples = config.get('nun_saved_samples', 4)
    if jax.local_device_count() != 1 and self.config.run_parallel:
      self.parallel = True

  def _initialize_model_and_sampler(
      self, rnd, model, sampler_init_state_fn, model_init_params_fn
  ):
    """Initializes model params, sampler state and gets the initial samples."""
    rng_param, rng_x0, rng_x0_ess, rng_state = jax.random.split(rnd, num=4)
    params = model_init_params_fn(
        jax.random.split(rng_param, self.config.num_models)
    )
    num_samples = self.config.batch_size * self.config.num_models
    x0 = model.get_init_samples(rng_x0, num_samples)
    x0_ess = model.get_init_samples(rng_x0_ess, 1)
    state = sampler_init_state_fn(
        jax.random.split(rng_state, self.config.num_models)
    )
    return params, x0, state, x0_ess

  def _prepare_data(self, params, x, state):
    distributing_batch = False
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
        distributing_batch = True
        batch_size_per_device = (
            self.config.batch_size // jax.local_device_count()
        )
        params = jax.device_put_replicated(params, jax.local_devices())
        state = jax.device_put_replicated(state, jax.local_devices())
        bshape = (jax.local_device_count(), self.config.num_models)
        x_shape = bshape + (batch_size_per_device,) + self.config_model.shape
        if self.sample_idx:
          self.sample_idx = jnp.array(
              [self.sample_idx]
              * (jax.local_device_count() // self.config.num_models)
          )
    else:
      bshape = (self.config.num_models,)
      x_shape = bshape + (self.config.batch_size,) + self.config_model.shape
    fn_breshape = lambda x: jnp.reshape(x, bshape + x.shape[1:])
    if not distributing_batch:
      state = jax.tree_map(fn_breshape, state)
      params = jax.tree_map(fn_breshape, params)
    x = jnp.reshape(x, x_shape)

    print('x shape: ', x.shape)
    print('state shape: ', state['steps'].shape)
    # key = list(params.keys())[0]
    # print('params shape: ', params[key].shape)
    return params, x, state, fn_breshape, bshape

  def _compile_sampler_step(self, step_fn):
    if not self.parallel:
      compiled_step = jax.jit(step_fn)
    else:
      compiled_step = jax.pmap(step_fn)
    return compiled_step

  def _compile_evaluator(self, obj_fn):
    if not self.parallel:
      compiled_obj_fn = jax.jit(obj_fn)
    else:
      compiled_obj_fn = jax.pmap(obj_fn)
    return compiled_obj_fn

  def _get_vmapped_functions(self, sampler, model, evaluator):
    model_init_params_fn = jax.vmap(model.make_init_params)
    sampler_init_state_fn = jax.vmap(sampler.make_init_state)
    step_fn = jax.vmap(functools.partial(sampler.step, model=model))
    obj_fn = self._vmap_evaluator(evaluator, model)
    model_frwrd = jax.vmap(model.forward)
    return (
        model_init_params_fn,
        sampler_init_state_fn,
        step_fn,
        obj_fn,
        model_frwrd,
    )

  def _compile_fns(self, step_fn, obj_fn):
    compiled_step = self._compile_sampler_step(step_fn)
    get_hop = jax.jit(self._get_hop)
    compiled_step_burnin = compiled_step
    compiled_step_mixing = compiled_step
    compiled_obj_fn = self._compile_evaluator(obj_fn)
    return (
        compiled_step_burnin,
        compiled_step_mixing,
        get_hop,
        compiled_obj_fn,
    )

  def get_results(self, model, sampler, evaluator, saver):
    self._get_chains_and_evaluations(model, sampler, evaluator, saver)

  def preprocess(self, model, sampler, evaluator, saver, rnd_key=0):
    (
        model_init_params_fn,
        sampler_init_state_fn,
        step_fn,
        obj_fn,
        model_frwrd,
    ) = self._get_vmapped_functions(sampler, model, evaluator)
    rnd = jax.random.PRNGKey(rnd_key)
    params, x, state, x0_ess = self._initialize_model_and_sampler(
        rnd, model, sampler_init_state_fn, model_init_params_fn
    )
    if params is None:
      print('Params is NONE')
      return False
    params, x, state, fn_reshape, breshape = self._prepare_data(
        params, x, state
    )
    compiled_fns = self._compile_fns(step_fn, obj_fn)
    return [
        compiled_fns,
        state,
        params,
        rnd,
        x,
        x0_ess,
        saver,
        evaluator,
        fn_reshape,
        breshape,
        model_frwrd,
        model,
    ]

  def _get_chains_and_evaluations(
      self, model, sampler, evaluator, saver, rnd_key=0
  ):
    """Sets up the model and the samlping alg and gets the chain of samples."""

    preprocessed_info = self.preprocess(
        model, sampler, evaluator, saver, rnd_key=0
    )
    if not preprocessed_info:
      return False
    self._compute_chain(*preprocessed_info)
    return True

  def _get_hop(self, x, new_x):
    return (
        jnp.sum(abs(x - new_x))
        / self.config.batch_size
        / self.config.num_models
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
      evaluator,
      fn_reshape,
      bshape,
      model_frwrd=None,
      model=None,
  ):
    raise NotImplementedError

  def vmap_evaluator(self, evaluator, model):
    raise NotImplementedError


class Sampling_Experiment(Experiment):
  """Experiment class that generates chains of samples."""

  def _vmap_evaluator(self, evaluator, model):
    obj_fn = evaluator.evaluate
    return obj_fn

  def _compute_chain(
      self,
      compiled_fns,
      state,
      params,
      rng,
      x,
      x0_ess,
      saver,
      evaluator,
      fn_reshape,
      bshape,
      model_frwrd,
      model,
  ):
    """Generates the chain of samples."""
    assert self.config.num_models == 1
    (chain, acc_ratios, hops, running_time, samples) = (
        self._initialize_chain_vars()
    )

    stp_burnin, stp_mixing, get_hop, obj_fn = compiled_fns
    get_mapped_samples, eval_metric = self._compile_additional_fns(evaluator)
    rng = jax.random.PRNGKey(10)
    selected_chains = jax.random.choice(
        rng,
        jnp.arange(self.config.batch_size),
        shape=(self.num_saved_samples,),
        replace=False,
    )

    # burn in
    burn_in_length = int(self.config.chain_length * self.config.ess_ratio) + 1
    for step in tqdm.tqdm(range(1, burn_in_length)):
      rng = jax.random.fold_in(rng, step)
      step_rng = fn_reshape(jax.random.split(rng, math.prod(bshape)))
      new_x, state, acc = stp_burnin(
          rng=step_rng,
          x=x,
          model_param=params,
          state=state,
      )

      if (
          self.config.save_samples or self.config.get_estimation_error
      ) and step % self.config.save_every_steps == 0:
        chains = new_x.reshape(self.config.batch_size, -1)
        samples.append(chains[selected_chains])

      if self.config.get_additional_metrics:
        # avg over all models
        acc = jnp.mean(acc)
        acc_ratios.append(acc)
        # hop avg over batch size and num models
        hops.append(get_hop(x, new_x))
      x = new_x

    for step in tqdm.tqdm(range(burn_in_length, 1 + self.config.chain_length)):
      rng = jax.random.fold_in(rng, step)
      step_rng = fn_reshape(jax.random.split(rng, math.prod(bshape)))
      start = time.time()
      new_x, state, acc = stp_mixing(
          rng=step_rng,
          x=x,
          model_param=params,
          state=state,
      )
      running_time += time.time() - start

      if (
          self.config.save_samples or self.config.get_estimation_error
      ) and step % self.config.save_every_steps == 0:
        chains = new_x.reshape(self.config.batch_size, -1)
        samples.append(chains[selected_chains])

      if self.config.get_additional_metrics:
        # avg over all models
        acc = jnp.mean(acc)
        acc_ratios.append(acc)
        # hop avg over batch size and num models
        hops.append(get_hop(x, new_x))
      mapped_sample = get_mapped_samples(new_x, x0_ess)
      mapped_sample = jax.device_put(mapped_sample, jax.devices('cpu')[0])
      chain.append(mapped_sample)
      x = new_x

    chain = jnp.array(chain)
    if self.parallel:
      chain = jnp.array([chain])
      rng = jnp.array([rng])
    ess = float(obj_fn(samples=chain, rnd=rng).reshape(-1))
    num_ll_calls = int(state['num_ll_calls'][0])
    metrics = eval_metric(ess, running_time, num_ll_calls)
    saver.save_results(acc_ratios, hops, metrics, running_time)
    if self.config.save_samples or self.config.get_estimation_error:
      if self.config.save_samples and self.config_model.name in [
          'rbm',
          'resnet',
      ]:
        saver.dump_samples(samples, visualize=True)
      elif (
          self.config.get_estimation_error
          and self.config_model.name == 'bernoulli'
      ):
        saver.dump_samples(samples, visualize=False)
        # samples= np.array(samples)
        params = params['params'][0].reshape(self.config_model.shape)
        saver.dump_params(params)
        # saver.plot_estimation_error(model, params, samples)

  def _initialize_chain_vars(self):
    chain = []
    acc_ratios = []
    hops = []
    samples = []
    running_time = 0

    return (
        chain,
        acc_ratios,
        hops,
        running_time,
        samples,
    )

  def _compile_additional_fns(self, evaluator):
    get_mapped_samples = jax.jit(self._get_mapped_samples)
    eval_metric = jax.jit(evaluator.get_eval_metrics)
    return get_mapped_samples, eval_metric

  def _get_mapped_samples(self, samples, x0_ess):
    samples = samples.reshape((-1,) + self.config_model.shape)
    samples = samples.reshape(samples.shape[0], -1)
    x0_ess = x0_ess.reshape((-1,) + self.config_model.shape)
    x0_ess = x0_ess.reshape(x0_ess.shape[0], -1)
    return jnp.sum(jnp.abs(samples - x0_ess), -1)


class Text_Infilling_Experiment(Sampling_Experiment):

  def _initialize_model_and_sampler(self, rnd, model, sampler):
    """Initializes model params, sampler state and gets the initial samples."""
    rng_param, rng_x0, rng_state = jax.random.split(rnd, num=3)
    del rnd
    params = model.make_init_params(rng_param)
    x0 = model.get_init_samples(rng_x0, 1, params)
    state = sampler.make_init_state(rng_state)
    return params, x0, state

  def _compile_fns(self, sampler, model):
    mdl_decode = jax.jit(model.get_sent_and_ll)
    step_fn = jax.jit(functools.partial(sampler.step, model=model))
    return (
        mdl_decode,
        step_fn,
    )

  def preprocess(self, model, sampler, evaluator, saver, rnd_key=0):
    rnd = jax.random.PRNGKey(rnd_key)
    params, x, state = self._initialize_model_and_sampler(rnd, model, sampler)
    compiled_fns = self._compile_fns(sampler, model)
    return [
        compiled_fns,
        state,
        params,
        rnd,
        x,
        saver,
        evaluator,
        model,
    ]

  def get_results(self, model, sampler, evaluator, saver):
    self._get_chains_and_evaluations(model, sampler, evaluator, saver)

  def get_topk_sentence(self, sentences, loglikes):
    topk_sentences = []
    if self.config.use_topk:
      sent_to_loglike = dict(zip(sentences, loglikes))
      sorted_sent = {
          k: v
          for k, v in sorted(sent_to_loglike.items(), key=lambda item: item[1])
      }
      topk_sentences = list(sorted_sent.keys())[-self.config.topk_num :]
      for i, sent in enumerate(topk_sentences):
        topk_sentences[i] = sent[2:]
      for i, sent in enumerate(sentences):
        sentences[i] = sent[2:]
    return sentences, topk_sentences

  def _get_chains_and_evaluations(
      self, model, sampler, evaluator, saver, rnd_key=0
  ):
    preprocessed_info = self.preprocess(
        model, sampler, evaluator, saver, rnd_key=0
    )

    params = preprocessed_info[2]
    rng = preprocessed_info[3]
    infill_sents = []
    infill_sents_topk = []
    while params:
      sentences = []
      loglikes = []
      for i in range(self.config.num_same_resample):
        sent, loglike = self._compute_chain(*preprocessed_info)
        if self.config.use_topk:
          sent = str(i) + ' ' + sent
          loglikes.append(loglike)
        sentences.append(sent)
        rng, _ = jax.random.split(rng)
        preprocessed_info[3] = rng

      sentences, topk_sentences = self.get_topk_sentence(sentences, loglikes)
      infill_sents.extend(sentences)
      if self.config.use_topk:
        infill_sents_topk.extend(topk_sentences)
      params = model.get_params()
      preprocessed_info[2] = params

    res_topk = []
    res = evaluator.evaluate(infill_sents, self.config_model.data_root)
    if self.config.use_topk:
      res_topk = evaluator.evaluate(
          infill_sents_topk, self.config_model.data_root
      )
    saver.dump_dict(res, res_topk)

  def _compute_chain(
      self,
      compiled_fns,
      state,
      params,
      rng,
      x,
      saver,
      evaluator,
      model,
  ):
    """Generates the chain of samples."""
    assert self.config.num_models == 1
    running_time = self._initialize_chain_vars()

    model_decode, stp = compiled_fns
    
    for step in tqdm.tqdm(range(1, 1+self.config.chain_length)):
      rng = jax.random.fold_in(rng, step)
      step_rng, rng = jax.random.split(rng)
      start = time.time()
      new_x, state, _ = stp(
          rng=step_rng,
          x=x,
          model_param=params,
          state=state,
      )
      running_time += time.time() - start

      x = new_x
    running_time = running_time / 2

    sent_tokenized, loglike = model_decode(params, x, self.config.use_topk)
    sampled_sentence = model.tokenizer.decode(sent_tokenized)
    print('Sampled Sentence: ', sampled_sentence, 'Likelihood: ', loglike)
    return sampled_sentence, loglike

  def _initialize_chain_vars(self):
    return 0


class CO_Experiment(Experiment):

  def _vmap_evaluator(self, evaluator, model):
    obj_fn = jax.vmap(functools.partial(evaluator.evaluate, model=model))
    return obj_fn

  def get_results(self, model, sampler, evaluator, saver):
    while True:
      if not self._get_chains_and_evaluations(model, sampler, evaluator, saver):
        break

  def _initialize_model_and_sampler(
      self, rnd, model, sampler_init_state_fn, model_init_params_fn
  ):
    data_list, x0, state, x0_es = super()._initialize_model_and_sampler(
        rnd, model, sampler_init_state_fn, model.make_init_params
    )
    if data_list is None:
      return None, x0, state, x0_es
    sample_idx, params, reference_obj = zip(*data_list)
    params = flax.core.frozen_dict.unfreeze(utils.tree_stack(params))
    self.ref_obj = jnp.array(reference_obj)
    if self.config_model.name == 'mis':
      self.ref_obj = jnp.ones_like(self.ref_obj)
    self.sample_idx = jnp.array(sample_idx)
    return params, x0, state, x0_es

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

  def _compute_chain(
      self,
      compiled_fns,
      state,
      params,
      rng,
      x,
      x0_ess,
      saver,
      evaluator,
      fn_reshape,
      bshape,
      model_frwrd,
      model,
  ):
    """Generates the chain of samples."""

    (
        chain,
        acc_ratios,
        hops,
        running_time,
        best_ratio,
        init_temperature,
        t_schedule,
        sample_mask,
        best_samples,
    ) = self._initialize_chain_vars(bshape)

    stp_burnin, stp_mixing, get_hop, obj_fn = compiled_fns
    # burn in
    burn_in_length = int(self.config.chain_length * self.config.ess_ratio) + 1

    for step in tqdm.tqdm(range(1, burn_in_length)):
      cur_temp = t_schedule(step)
      params['temperature'] = init_temperature * cur_temp
      rng = jax.random.fold_in(rng, step)
      step_rng = fn_reshape(jax.random.split(rng, math.prod(bshape)))
      new_x, state, acc = stp_burnin(
          rng=step_rng,
          x=x,
          model_param=params,
          state=state,
          x_mask=params['mask'],
      )

      if step % self.config.log_every_steps == 0:
        eval_val = obj_fn(samples=new_x, params=params)
        eval_val = eval_val.reshape(self.config.num_models, -1)
        ratio = jnp.max(eval_val, axis=-1).reshape(-1) / self.ref_obj
        is_better = ratio > best_ratio
        best_ratio = jnp.maximum(ratio, best_ratio)
        sample_mask = sample_mask.reshape(best_ratio.shape)

        br = np.array(best_ratio[sample_mask])
        br = jax.device_put(br, jax.devices('cpu')[0])
        chain.append(br)

        if self.config.save_samples or self.config_model.name == 'normcut':
          step_chosen = jnp.argmax(eval_val, axis=-1, keepdims=True)
          rnew_x = jnp.reshape(
              new_x,
              (self.config.num_models, self.config.batch_size)
              + self.config_model.shape,
          )
          chosen_samples = jnp.take_along_axis(
              rnew_x, jnp.expand_dims(step_chosen, -1), axis=-2
          )
          chosen_samples = jnp.squeeze(chosen_samples, -2)
          best_samples = jnp.where(
              jnp.expand_dims(is_better, -1), chosen_samples, best_samples
          )

      if self.config.get_additional_metrics:
        # avg over all models
        acc = jnp.mean(acc)
        acc_ratios.append(acc)
        # hop avg over batch size and num models
        hops.append(get_hop(x, new_x))
      x = new_x

    for step in tqdm.tqdm(range(burn_in_length, 1 + self.config.chain_length)):
      cur_temp = t_schedule(step)
      params['temperature'] = init_temperature * cur_temp
      rng = jax.random.fold_in(rng, step)
      step_rng = fn_reshape(jax.random.split(rng, math.prod(bshape)))
      start = time.time()
      new_x, state, acc = stp_mixing(
          rng=step_rng,
          x=x,
          model_param=params,
          state=state,
          x_mask=params['mask'],
      )
      running_time += time.time() - start
      if step % self.config.log_every_steps == 0:
        eval_val = obj_fn(samples=new_x, params=params)
        eval_val = eval_val.reshape(self.config.num_models, -1)
        ratio = jnp.max(eval_val, axis=-1).reshape(-1) / self.ref_obj
        is_better = ratio > best_ratio
        best_ratio = jnp.maximum(ratio, best_ratio)
        sample_mask = sample_mask.reshape(best_ratio.shape)

        br = np.array(best_ratio[sample_mask])
        br = jax.device_put(br, jax.devices('cpu')[0])
        chain.append(br)

        if self.config.save_samples or self.config_model.name == 'normcut':
          step_chosen = jnp.argmax(eval_val, axis=-1, keepdims=True)
          rnew_x = jnp.reshape(
              new_x,
              (self.config.num_models, self.config.batch_size)
              + self.config_model.shape,
          )
          chosen_samples = jnp.take_along_axis(
              rnew_x, jnp.expand_dims(step_chosen, -1), axis=-2
          )
          chosen_samples = jnp.squeeze(chosen_samples, -2)
          best_samples = jnp.where(
              jnp.expand_dims(is_better, -1), chosen_samples, best_samples
          )

      if self.config.get_additional_metrics:
        # avg over all models
        acc = jnp.mean(acc)
        acc_ratios.append(acc)
        # hop avg over batch size and num models
        hops.append(get_hop(x, new_x))
      x = new_x

    if not (self.config.save_samples or self.config_model.name == 'normcut'):
      best_samples = []
    saver.dump_results(
        chain, best_ratio[sample_mask], running_time, best_samples
    )
    saver.save_results(acc_ratios, hops, None, running_time)

  def _initialize_chain_vars(self, bshape):
    t_schedule = self._build_temperature_schedule(self.config)
    sample_mask = self.sample_idx >= 0
    chain = []
    acc_ratios = []
    hops = []
    running_time = 0
    best_ratio = jnp.ones(self.config.num_models, dtype=jnp.float32) * -1e9
    init_temperature = jnp.ones(bshape, dtype=jnp.float32)
    dim = math.prod(self.config_model.shape)
    best_samples = jnp.zeros([self.config.num_models, dim])
    return (
        chain,
        acc_ratios,
        hops,
        running_time,
        best_ratio,
        init_temperature,
        t_schedule,
        sample_mask,
        best_samples,
    )
