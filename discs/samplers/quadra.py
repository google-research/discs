"""Quadratic Sampler: A version of Any Scale Balanced Sampler"""

from discs.common import math_util as math
from discs.samplers import abstractsampler
import jax
import jax.numpy as jnp
from jax.scipy import special
import ml_collections


class BinaryQuadraSampler(abstractsampler.AbstractSampler):
  """Quadra sampler. 
  Current implementation only supports binary distribution.
  """

  def get_value_and_rates(self, model, model_param, x):
    ll_x, grad_x = model.get_value_and_grad(model_param, x)
    if self.num_categories == 2:
      ll_delta = (1 - 2 * x) * grad_x
    else:
      ll_delta = grad_x - jnp.sum(grad_x * x, axis=-1, keepdims=True)
    log_weight_x = self.apply_weight_function_logscale(ll_delta)
    return ll_x, {'weights': log_weight_x, 'delta': ll_delta}
  
  def get_dist_at(self, x, log_rate_x, rng_aux):
    _ = x
    log_weight_x = log_rate_x['weights']
    aux = x @ self.transform.T + jnp.random.normal(rng_aux, x.shape)
    mu = log_weight_x - (x @ self.hessian + x * self.diag) + aux @ self.transform
    log_rate = (1. - 2. * x) * mu - self.diag / 2.
    return jnp.exp(log_rate)
  
  def sample_from_proposal(self, rng, x, dist_x):
    flip = jax.random.bernoulli(rng, p=dist_x)
    y = x * (1 - flip) + flip * (1 - x)
    return y

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__(config)
    self.diag_type = config.model.diag_type
    self.diag = None
    self.transform = None
  
  def init_curv(self, model):
    self.hessian = model.hessian
    if self.diag_type == 'shift':
      eig_v, eig_vec = jnp.linalg.eigh(self.hessian)
      diag = jnp.full(shape=self.hessian.shape[0], 
                      fill_value=jnp.clip(- eig_v.min(),
                                          a_min=1e-9)
                      )
      transform = eig_vec @ jnp.diag(jnp.clip(eig_v, a_min=0) ** 0.5) @ eig_vec.T
      self.diag = diag
      self.transform = transform
    else:
      raise NotImplementedError

  def step(self, model, rng, x, model_param, state, x_mask=None):
    _ = x_mask
    if self.num_categories != 2:
      x = jax.nn.one_hot(x, self.num_categories, dtype=jnp.float32)
    rng_aux, rng_new_sample = jax.random.split(rng, 2)

    _, log_rate_x = self.get_value_and_rates(model, model_param, x)

    dist_x = self.get_dist_at(x, log_rate_x, rng_aux)
    new_x = self.sample_from_proposal(rng_new_sample, x, dist_x)
    
    acc = jnp.full(1., x.shape[0])
    new_state = {
        'num_ll_calls': state['num_ll_calls'] + 2,
    }
    return new_x, new_state, acc


def build_sampler(config):
  if config.model.num_categories == 2:
    return BinaryQuadraSampler(config)
  else:
    raise NotImplementedError
