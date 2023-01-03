"""Gibbs with gradient."""

from discs.common import math
from discs.samplers import locallybalanced
import jax
import jax.numpy as jnp
import ml_collections


class GibbsWithGradSampler(locallybalanced.LocallyBalancedSampler):
  """Gibbs With Grad Sampler Class."""

  def step(self, model, rng, x, model_param, state, x_mask=None):
    """Given the current sample, returns the next sample of the chain.

    Args:
      model: target distribution.
      rng: random key generator for JAX.
      x: current sample.
      model_param: target distribution parameters used for loglikelihood
        calulation.
      state: the state of the sampler.
      x_mask: (optional) broadcast to x, masking out certain dimensions.

    Returns:
      New sample.
    """
    if self.num_categories != 2:
      x = jax.nn.one_hot(x, self.num_categories, dtype=jnp.float32)
    rng_new_sample, rng_acceptance = jax.random.split(rng)

    ll_x, grad_x = model.get_value_and_grad(model_param, x)
    dist_x = self.get_dist_at(x, grad_x, x_mask)
    y, aux = self.sample_from_proposal(rng_new_sample, x, dist_x, state)
    ll_x2y = self.get_ll_onestep(dist_x, src=x, dst=y, aux=aux)
    ll_y, grad_y = model.get_value_and_grad(model_param, y)
    dist_y = self.get_dist_at(y, grad_y, x_mask)
    ll_y2x = self.get_ll_onestep(dist_y, src=y, dst=x, aux=aux)
    log_acc = ll_y + ll_y2x - ll_x - ll_x2y
    new_x, new_state = self.select_sample(rng_acceptance, log_acc, x, y, state)
    return new_x, new_state


class BinaryGWGSampler(GibbsWithGradSampler):
  """GWG for binary data."""

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__(config)
    self.num_flips = config.sampler.get('num_flips', 1)

  def select_sample(self, rng, log_acc,
                    current_sample, new_sample, sampler_state):
    y, acc = math.mh_step(rng, log_acc, current_sample, new_sample)
    sampler_state = {
        'num_ll_calls': sampler_state['num_ll_calls'] + 2,
        'acceptance_rate': jnp.mean(acc.astype(jnp.float32))
    }
    return y.astype(jnp.int32), sampler_state

  def get_ll_onestep(self, dist, src, dst, aux):
    ll = dist[jnp.expand_dims(jnp.arange(dist.shape[0]), axis=1), aux]
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
        rng, dist_x, num_samples=self.num_flips, replacement=True)
    x_shape = x.shape
    x = jnp.reshape(x, (x_shape[0], -1))
    rows = jnp.expand_dims(jnp.arange(idx.shape[0]), axis=1)
    y = x.at[rows, idx].set(1 - x[rows, idx])
    y = jnp.reshape(y, x_shape)
    return y, idx


class AdaptiveGWGSampler(BinaryGWGSampler):
  """Adaptive GWG for binary data."""

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__(config)
    self.target_acceptance_rate = config.sampler.target_acceptance_rate

  def make_init_state(self):
    state = super().make_init_state()
    state['radius'] = jnp.ones(shape=(), dtype=jnp.float32)
    return state

  def select_sample(self, rng, log_acc,
                    current_sample, new_sample, sampler_state):
    y, new_state = super().select_sample(
        rng, log_acc, current_sample, new_sample, sampler_state)
    acc = jnp.mean(jnp.exp(jnp.clip(log_acc, a_max=0.0)))
    r = sampler_state['radius'] + acc - self.target_acceptance_rate
    new_state['radius'] = jnp.clip(r, a_min=1, a_max=math.prod(y.shape[1:]))
    return y, new_state

  def get_ll_onestep(self, dist, src, dst, aux):
    ll = dist * aux.astype(jnp.float32)
    ll = jnp.sum(ll, axis=-1)
    return ll

  def sample_from_proposal(self, rng, x, dist_x, state):
    ns = jnp.clip(jnp.round(state['radius']).astype(jnp.int32),
                  a_min=1, a_max=dist_x.shape[1])
    selected = math.multinomial(
        rng, dist_x, num_samples=ns, replacement=True, is_nsample_const=False)
    indicator = (selected > 0).astype(jnp.int32)
    x_shape = x.shape
    x = jnp.reshape(x, (x_shape[0], -1))
    y = x * (1 - indicator) + indicator * (1 - x)
    y = jnp.reshape(y, x_shape)
    return y, selected


class CategoricalGWGSampler(GibbsWithGradSampler):
  """GWG for categorical data."""

  def select_sample(self, rng, log_acc,
                    current_sample, new_sample, sampler_state):
    y = math.mh_step(rng, log_acc, current_sample, new_sample)
    y = jnp.argmax(y, axis=-1)  # onehot to int-tensor
    return y, sampler_state

  def get_dist_at(self, x, grad_x):
    ll_delta = grad_x - jnp.sum(grad_x * x, axis=-1, keepdims=True)
    score_change_x = self.apply_weight_function_logscale(ll_delta)
    score_change_x = score_change_x - x * 1e9
    log_prob = jnp.reshape(score_change_x, list(x.shape[:-1]) + [-1])
    return log_prob

  def sample_from_proposal(self, rng, x, dist_x, state):
    raise NotImplementedError


def build_sampler(config):
  if config.model.num_categories == 2:
    if config.sampler.get('adaptive', False):
      return AdaptiveGWGSampler(config)
    else:
      return BinaryGWGSampler(config)
  else:
    return CategoricalGWGSampler(config)
