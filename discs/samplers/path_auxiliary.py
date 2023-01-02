"""Path auxiliary sampler."""

from discs.common import math
from discs.samplers import locallybalanced
import jax
import jax.numpy as jnp
import ml_collections


class PathAuxiliarySampler(locallybalanced.LocallyBalancedSampler):
  """Path auxiliary sampler."""

  def step(self, model, rng, x, model_param, state, x_mask=None):
    if self.num_categories != 2:
      x = jax.nn.one_hot(x, self.num_categories, dtype=jnp.float32)
    rng_new_sample, rng_acceptance = jax.random.split(rng)

    ll_x, y, trajectory, num_calls_forward = self.proposal(
        model, rng_new_sample, x, model_param, state, x_mask)
    ll_x2y = trajectory['ll_x2y']
    ll_y, ll_y2x, num_calls_backward = self.ll_y2x(
        model, x, model_param, trajectory, y)
    log_acc = ll_y + ll_y2x - ll_x - ll_x2y
    new_x, new_state = self.select_sample(
        rng_acceptance, num_calls_forward + num_calls_backward,
        log_acc, x, y, state)

    return new_x, new_state

  def select_sample(self, rng, num_calls, log_acc,
                    current_sample, new_sample, sampler_state):
    y, acc = math.mh_step(rng, log_acc, current_sample, new_sample)
    sampler_state = {
        'num_ll_calls': sampler_state['num_ll_calls'] + num_calls,
        'acceptance_rate': jnp.mean(acc.astype(jnp.float32))
    }
    return y.astype(jnp.int32), sampler_state


class PASTrajectory(PathAuxiliarySampler):
  pass


class PAFSNoReplacement(PathAuxiliarySampler):
  """Path auxiliary sampler with no replacement proposal sampling."""

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__(config)
    self.adaptive = config.sampler.get('adaptive', False)
    if self.adaptive:
      self.target_acceptance_rate = config.sampler.target_acceptance_rate
    else:
      self.num_flips = config.sampler.get('num_flips', 1)
    self.approx_with_grad = config.sampler.get('approx_with_grad', True)

  def select_sample(self, rng, num_calls, log_acc,
                    current_sample, new_sample, sampler_state):
    y, new_state = super().select_sample(
        rng, num_calls, log_acc, current_sample, new_sample, sampler_state)
    if self.adaptive:
      acc = jnp.mean(jnp.exp(jnp.clip(log_acc, a_max=0.0)))
      r = sampler_state['radius'] + acc - self.target_acceptance_rate
      new_state['radius'] = jnp.clip(r, a_min=1, a_max=math.prod(y.shape[1:]))
    return y, new_state

  def make_init_state(self, rng):
    state = super().make_init_state(rng)
    state['radius'] = jnp.ones(shape=(), dtype=jnp.float32) * self.num_flips
    return state

  def get_local_dist(self, model, x, model_param):
    if self.approx_with_grad:
      ll_x, grad_x = model.get_value_and_grad(model_param, x)
      if self.num_categories != 2:
        logratio = grad_x - jnp.sum(grad_x * x, axis=-1, keepdims=True)
      else:
        logratio = (1 - 2 * x) * grad_x
      num_calls = 1
    else:
      logratio, num_calls, _ = model.logratio_in_neighborhood(model_param, x)
      ll_x = model.forward(model_param, x)
      num_calls = num_calls + 1
      assert logratio.shape == x.shape
    logits = self.apply_weight_function_logscale(logratio)
    log_prob = jax.nn.log_softmax(jnp.reshape(logits, (x.shape[0], -1)), -1)
    return ll_x, log_prob, num_calls

  def proposal(self, model, rng, x, model_param, state, x_mask):
    ll_x, log_prob, num_calls = self.get_local_dist(model, x, model_param)
    num_classes = log_prob.shape[1]
    if self.adaptive:
      num_samples = jnp.clip(jnp.round(state['radius']).astype(jnp.int32),
                             a_min=1, a_max=num_classes)
    else:
      num_samples = self.num_flips
    selected_idx, ll_selected = math.multinomial(
        rng, log_prob, num_samples, replacement=False, return_ll=True,
        is_nsample_const=not self.adaptive, need_ordering_info=True)
    x_shape = x.shape
    x = jnp.reshape(x, log_prob.shape)
    assert self.num_categories == 2
    if self.adaptive:
      mask = selected_idx['selected_mask']
      y = x * (1 - mask) + mask * (1 - x)
    else:
      assert self.num_categories == 2
      rows = jnp.expand_dims(jnp.arange(x.shape[0]), axis=1)
      y = x.at[rows, selected_idx].set(1 - x[rows, selected_idx])
    y = jnp.reshape(y, x_shape)
    trajectory = {
        'll_x2y': jnp.sum(ll_selected, axis=-1),
        'selected_idx': selected_idx,
    }
    return ll_x, y, trajectory, num_calls

  def ll_y2x(self, model, x, model_param, forward_trajectory, y):
    ll_y, log_prob, num_calls = self.get_local_dist(model, y, model_param)
    if self.adaptive:
      selected_mask = forward_trajectory['selected_idx']['selected_mask']
      order_info = forward_trajectory['selected_idx']['perturbed_ll']
      raise NotImplementedError
    else:
      reverse_idx_traj = jnp.flip(forward_trajectory['selected_idx'], axis=-1)
      ll_idx = jnp.take_along_axis(log_prob, reverse_idx_traj, -1)
      ll_y2x_traj = math.noreplacement_sampling_renormalize(ll_idx)
      ll_y2x = jnp.sum(ll_y2x_traj, axis=-1)
    return ll_y, ll_y2x, num_calls


def build_sampler(config):
  if config.sampler.use_fast_path:
    return PAFSNoReplacement(config)
  else:
    raise NotImplementedError
