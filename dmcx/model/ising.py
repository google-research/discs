"""Ising Energy Function."""

from dmcx.model import abstractmodel
import jax
import jax.numpy as jnp
import ml_collections
from tqdm import tqdm
import dmcx.sampler.randomwalk as randomwalk_sampler
import pdb


class Ising(abstractmodel.AbstractModel):
  """Bernouli Distribution."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.init_sigma = config.init_sigma
    self.lamda = config.lamda
    if isinstance(config.shape, int):
      self.shape = (config.shape, config.shape)
    else:
      self.shape = config.shape

    self.sampler_config = ml_collections.ConfigDict(
        initial_dictionary=dict(
            adaptive=False,
            target_acceptance_rate=0.234,
            sample_shape=self.shape,
            num_categories=2))
    self.sampler = randomwalk_sampler.RandomWalkSampler(self.sampler_config)

    self.expected_val = None
    self.var = None

  def make_init_params(self, rnd):

    params_weight_h = -self.lamda * jnp.ones([self.shape[0], self.shape[1] - 1])
    params_weight_v = -self.lamda * jnp.ones([self.shape[0] - 1, self.shape[1]])
    params_b = jax.random.normal(rnd, shape=self.shape) * self.init_sigma
    return [params_b, params_weight_h, params_weight_v]

  def get_init_samples(self, rnd, num_samples: int):
    x0 = jax.random.choice(
        rnd,
        jnp.array([-1, 1]),
        shape=(num_samples,) + self.shape,
        replace=True)
    return x0

  def forward(self, params, x):

    w_b = params[0]
    w_h = params[1]
    w_v = params[2]
    sum_neighbors = jnp.zeros((x.shape[0],) + self.shape)
    sum_neighbors = sum_neighbors.at[:, :, :-1].set(
        sum_neighbors[:, :, :-1] + x[:, :, :-1] * x[:, :, 1:] * w_h)  #right
    sum_neighbors = sum_neighbors.at[:, :, 1:].set(
        sum_neighbors[:, :, 1:] + x[:, :, 1:] * x[:, :, :-1] * w_h)  # left
    sum_neighbors = sum_neighbors.at[:, :-1, :].set(
        sum_neighbors[:, :-1, :] + x[:, :-1, :] * x[:, 1:, :] * w_v)  # down
    sum_neighbors = sum_neighbors.at[:, 1:, :].set(
        sum_neighbors[:, 1:, :] + x[:, 1:, :] * x[:, :-1, :] * w_v)  #up
    biases = w_b * x
    energy_indeces = sum_neighbors + biases
    loglikelihood = jnp.sum((energy_indeces).reshape(x.shape[0], -1), axis=-1)
    return loglikelihood

  def get_value_and_grad(self, params, x):
    x = x.astype(jnp.float32)  # int tensor is not differentiable

    def fun(z):
      loglikelihood = self.forward(params, z)
      return jnp.sum(loglikelihood), loglikelihood

    (_, loglikelihood), grad = jax.value_and_grad(fun, has_aux=True)(x)
    return loglikelihood, grad

  def get_expected_val(self, params):
    if self.expected_val is not None:
      return self.expected_val
    return self.compute_mean_and_var(params)[0]

  def get_var(self, params):    
    if self.var is not None:
      return self.var
    return self.compute_mean_and_var(params)[1]

  def get_expected_val_from_samples(self, samples):
    mean_over_chain = jnp.mean(samples, axis=0)
    mean_over_batch = jnp.mean(mean_over_chain, axis=0)
    return mean_over_batch

  def get_var_from_samples(self, samples):
    sample_mean = jnp.mean(samples, axis=0, keepdims=True)
    var_over_samples = jnp.sum(
        (samples - sample_mean)**2, axis=0) / (
            samples.shape[0] - 1)
    mean_var_over_batch = jnp.mean(var_over_samples, axis=0)
    return mean_var_over_batch

  def compute_mean_and_var(self, rnd):

    num_samples = 100
    chain_length = 5000
    sample_length = 500
    chain = []
    rnd = jax.random.PRNGKey(0)
    rng_param, rng_x0, rng_sampler, rng_sampler_step = jax.random.split(
        rnd, num=4)
    params = self.make_init_params(rng_param)
    x = self.get_init_samples(rng_x0, num_samples)
    step_jit = jax.jit(self.sampler.step, static_argnums=0)
    state = self.sampler.make_init_state(rng_sampler)
    for i in tqdm(range(chain_length)):
      rng_sampler_step_p = jax.random.split(rng_sampler_step)
      x, state = step_jit(self, rng_sampler_step_p, x, params, state)
      del rng_sampler_step_p
      rng_sampler_step, _ = jax.random.split(rng_sampler_step)
      if i > (chain_length - sample_length):
        chain.append(x)
    samples = jnp.array(chain)
    self.expected_val = self.get_expected_val_from_samples(samples)
    self.var = self.get_var_from_samples(samples)
    return [self.expected_val, self.var]
