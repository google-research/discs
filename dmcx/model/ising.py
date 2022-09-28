"""Ising Energy Function."""

import os

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
from dmcx.model import abstractmodel
import jax
import jax.numpy as jnp
import ml_collections
from tqdm import tqdm
import dmcx.sampler.blockgibbs as blockgibbs_sampler
import pdb


class Ising(abstractmodel.AbstractModel):
  """Ising Distribution with Cyclic 2D Lattice."""

  def __init__(self, config: ml_collections.ConfigDict):

    if isinstance(config.shape, int):
      self.shape = (config.shape, config.shape)
    else:
      self.shape = config.shape
    self.init_sigma = config.init_sigma
    self.lamda = config.lamda
    self.external_field_type = config.external_field_type
    self.expected_val = None
    self.var = None
    self.setup_sampler(config)

  def setup_sampler(self, config: ml_collections.ConfigDict):

    self.sampler_config = ml_collections.ConfigDict(
        initial_dictionary=dict(
            sample_shape=self.shape,
            num_categories=2,
            random_order=False,
            block_size=3))
    self.sampler = blockgibbs_sampler.BlockGibbsSampler(self.sampler_config)
    self.parallel_sampling = config.parallel_sampling

  def make_init_params(self, rnd):
    # connectivity strength
    params_weight_h = -self.lamda * jnp.ones(self.shape)
    params_weight_v = -self.lamda * jnp.ones(self.shape)
    # external force
    if self.external_field_type == 1:
      params_b = jax.random.normal(rnd, shape=self.shape) * self.init_sigma
    else:
      params_b = jnp.zeros(shape=self.shape)
    return jnp.array([params_b, params_weight_h, params_weight_v])

  def get_init_samples(self, rnd, num_samples: int):
    x0 = jax.random.randint(
        rnd,
        shape=(num_samples,) + self.shape,
        minval=0,
        maxval=2,
        dtype=jnp.int32)
    return x0

  def forward(self, params, x):

    w_b = params[0]
    w_h = params[1][:, :-1]
    w_v = params[2][:-1, :]
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

  def compute_mean_and_var(self, params):

    rnd = jax.random.PRNGKey(0)
    samples = self.generate_chain_of_samples(rnd, params)
    self.expected_val = self.get_expected_val_from_samples(samples)
    self.var = self.get_var_from_samples(samples)
    return [self.expected_val, self.var]

  def generate_chain_of_samples(self, rnd, params):
    """Using Block Gibbs Sampler Generates Chain of Samples."""

    num_samples = 100
    chain_length = 10
    sample_length = 1

    rng_x0, rng_sampler, rng_sampler_step = jax.random.split(rnd, num=3)
    del rnd
    state = self.sampler.make_init_state(rng_sampler)
    x = self.get_init_samples(rng_x0, num_samples)

    if self.parallel_sampling:
      sampler_step_fn = jax.pmap(
          self.sampler.step, static_broadcasted_argnums=[0])
      n_devices = jax.local_device_count()
      
      params = jnp.stack([params] * n_devices)
      state = jnp.stack([state] * n_devices)
      x = self.split(x, n_devices)
      rnd_split_num = n_devices
    else:
      sampler_step_fn = jax.jit(self.sampler.step, static_argnums=0)
      n_devices = 1
      rnd_split_num = n_devices + 1

    samples = self.compute_chains(rng_sampler_step, chain_length, sample_length,
                                  sampler_step_fn, params, state, x,
                                  rnd_split_num)
        
    if self.parallel_sampling:
      samples = samples.reshape((samples.shape[0], num_samples) + self.shape)

    return samples

  def compute_chains(self, rng_sampler_step, chain_length, sample_length,
                     sampler_step_fn, params, state, x, rnd_split_num):
    chain = []
    for i in tqdm(range(chain_length)):
      rng_sampler_step_p = jax.random.split(rng_sampler_step, num=rnd_split_num)
      x, state = sampler_step_fn(self, rng_sampler_step_p, x, params, state)
      del rng_sampler_step_p
      rng_sampler_step, _ = jax.random.split(rng_sampler_step)
      if i >= (chain_length - sample_length):
        chain.append(x)
    samples = jnp.array(chain)
    return samples

  def get_expected_val_from_samples(self, samples):
    """Computes distribution expected value from samples [chain lenght, batch size, sample shape]."""
    mean_over_chain = jnp.mean(samples, axis=0)
    mean_over_batch = jnp.mean(mean_over_chain, axis=0)
    return mean_over_batch

  def get_var_from_samples(self, samples):
    """Computes distribution variance from samples [chain lenght, batch size, sample shape]."""
    sample_mean = jnp.mean(samples, axis=0, keepdims=True)
    # unbiased var estimator
    var_over_samples = jnp.sum(
        (samples - sample_mean)**2, axis=0) / (
            samples.shape[0] - 1)
    mean_var_over_batch = jnp.mean(var_over_samples, axis=0)
    return mean_var_over_batch

  def split(self, arr, n_devices):
    return arr.reshape(n_devices, arr.shape[0] // n_devices, *arr.shape[1:])
