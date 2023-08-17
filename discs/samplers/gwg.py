"""Gibbs with gradient."""

import pdb
from discs.common import math_util as math
from discs.samplers import locallybalanced
import jax
import jax.numpy as jnp
import ml_collections


class GibbsWithGradSampler(locallybalanced.LocallyBalancedSampler):
  """Gibbs With Grad Sampler Class."""

  def select_sample(
      self, rng, log_acc, current_sample, new_sample, sampler_state
  ):
    y, sampler_state = super().select_sample(
        rng, log_acc, current_sample, new_sample, sampler_state
    )
    sampler_state['num_ll_calls'] += 4
    return y, sampler_state

  def step(self, model, rng, x, model_param, state, x_mask=None):
    if self.num_categories != 2:
      x = jax.nn.one_hot(x, self.num_categories, dtype=jnp.float32)
    rng_new_sample, rng_acceptance = jax.random.split(rng)

    ll_x, grad_x = model.get_value_and_grad(model_param, x)
    dist_x = self.get_dist_at(x, grad_x, x_mask)
    y, aux = self.sample_from_proposal(rng_new_sample, x, dist_x, state)
    ll_x2y = self.get_ll_onestep(dist_x, aux=aux, src_to_dst='x2y')
    ll_y, grad_y = model.get_value_and_grad(model_param, y)
    dist_y = self.get_dist_at(y, grad_y, x_mask)
    ll_y2x = self.get_ll_onestep(dist_y, aux=aux, src_to_dst='y2x')
    log_acc = ll_y + ll_y2x - ll_x - ll_x2y
    new_x, new_state = self.select_sample(rng_acceptance, log_acc, x, y, state)

    acc = jnp.mean(jnp.clip(jnp.exp(log_acc), a_max=1))
    return new_x, new_state, acc


class BinaryGWGSampler(GibbsWithGradSampler):
  """GWG for binary data."""

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__(config)
    self.num_flips = config.sampler.get('num_flips', 1)

  def get_ll_onestep(self, dist, aux, src_to_dst):
    ll = dist[
        jnp.expand_dims(jnp.arange(dist.shape[0]), axis=1), aux[src_to_dst]
    ]
    ll = jnp.sum(ll, axis=-1)
    return ll

  def get_dist_at(self, x, grad_x, x_mask):
    ll_delta = (1 - 2 * x) * grad_x
    score_change_x = self.apply_weight_function_logscale(ll_delta)
    score_change_x = jnp.reshape(score_change_x, (score_change_x.shape[0], -1))
    if x_mask is not None:
      score_change_x = score_change_x * x_mask + -1e9 * (1 - x_mask)
    log_prob = jax.nn.log_softmax(score_change_x, axis=-1)
    return log_prob

  def sample_from_proposal(self, rng, x, dist_x, state):
    idx = math.multinomial(
        rng,
        dist_x,
        num_samples=self.num_flips,
        replacement=True,
    )
    x_shape = x.shape
    x = jnp.reshape(x, (x_shape[0], -1))
    rows = jnp.expand_dims(jnp.arange(idx.shape[0]), axis=1)
    y = x.at[rows, idx].set(1 - x[rows, idx])
    y = jnp.reshape(y, x_shape)
    return y, {'x2y': idx, 'y2x': idx}


class AdaptiveGWGSampler(BinaryGWGSampler):
  """Adaptive GWG for binary data."""

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__(config)
    self.target_acceptance_rate = config.sampler.target_acceptance_rate

  def make_init_state(self, rng):
    state = super().make_init_state(rng)
    state['radius'] = jnp.ones(shape=(), dtype=jnp.float32)
    return state

  def select_sample(
      self, rng, log_acc, current_sample, new_sample, sampler_state
  ):
    y, new_state = super().select_sample(
        rng, log_acc, current_sample, new_sample, sampler_state
    )
    acc = jnp.mean(jnp.exp(jnp.clip(log_acc, a_max=0.0)))
    r = sampler_state['radius'] + acc - self.target_acceptance_rate
    new_state['radius'] = jnp.clip(r, a_min=1, a_max=math.prod(y.shape[1:]))
    return y, new_state

  def get_ll_onestep(self, dist, aux, src_to_dst):
    ll = dist * aux[src_to_dst].astype(jnp.float32)
    ll = jnp.sum(ll, axis=-1)
    return ll

  def sample_from_proposal(self, rng, x, dist_x, state):
    ns = jnp.clip(
        jnp.round(state['radius']).astype(jnp.int32),
        a_min=1,
        a_max=dist_x.shape[1],
    )
    selected = math.multinomial(
        rng, dist_x, num_samples=ns, replacement=True, is_nsample_const=False
    )
    indicator = (selected > 0).astype(jnp.int32)
    x_shape = x.shape
    x = jnp.reshape(x, (x_shape[0], -1))
    y = x * (1 - indicator) + indicator * (1 - x)
    y = jnp.reshape(y, x_shape)
    return y, {'x2y': selected, 'y2x': selected}


class CategoricalGWGSampler(GibbsWithGradSampler):
  """GWG for categorical data."""

  def select_sample(
      self, rng, log_acc, current_sample, new_sample, sampler_state
  ):
    y, new_state = super().select_sample(
        rng, log_acc, current_sample, new_sample, sampler_state
    )
    return y, new_state

  # TODO: kati using mask
  def get_dist_at(self, x, grad_x, x_mask):
    ll_delta = grad_x - jnp.sum(grad_x * x, axis=-1, keepdims=True)
    score_change_x = self.apply_weight_function_logscale(ll_delta)
    score_change_x = score_change_x - x * 1e9
    score_change_x = jnp.reshape(score_change_x, (score_change_x.shape[0], -1))
    log_prob = jax.nn.log_softmax(score_change_x, axis=-1)
    return log_prob

  def sample_from_proposal(self, rng, x, dist_x, state):
    # Hamming 1
    idx = math.multinomial(
        rng,
        dist_x,
        num_samples=1,
        replacement=True,
    ).reshape(-1)
    x_int = jnp.argmax(x, axis=-1)
    x_shape = x_int.shape
    x_int = jnp.reshape(x_int, (x_int.shape[0], -1))
    idx_c = idx % self.num_categories
    idx_i = jnp.floor_divide(idx, self.num_categories)
    y_int = x_int.at[jnp.arange(x.shape[0]), idx_i].set(idx_c)
    idx_y2x_c = jnp.sum(jnp.where(x_int != y_int, 1, 0) * x_int, -1)
    idx_y2x = idx_i * self.num_categories + idx_y2x_c
    y_int = jnp.reshape(y_int, x_shape)
    y = jax.nn.one_hot(y_int, self.num_categories, dtype=jnp.float32)
    return y, {'x2y': idx, 'y2x': idx_y2x}

  def get_ll_onestep(self, dist, aux, src_to_dst):
    ll = dist[
        jnp.expand_dims(jnp.arange(dist.shape[0]), axis=1), aux[src_to_dst]
    ]
    ll = jnp.sum(ll, axis=-1)
    return ll


def build_sampler(config):
  if config.model.num_categories == 2:
    if config.sampler.get('adaptive', False):
      return AdaptiveGWGSampler(config)
    else:
      return BinaryGWGSampler(config)
  else:
    return CategoricalGWGSampler(config)
