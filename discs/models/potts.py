"""Potts Energy Function."""

from discs.models import abstractmodel
import jax
import jax.numpy as jnp
import ml_collections
import pdb


class Potts(abstractmodel.AbstractModel):
  """Potts Distribution (2D cyclic ising model with one-hot representation)."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.sample_shape = config.shape
    self.lambdaa = config.lambdaa
    self.num_categories = config.num_categories
    self.shape = self.sample_shape + (self.num_categories,)

  def make_init_params(self, rnd):
    # connectivity strength
    params_weight_h = self.lambdaa * jnp.ones(self.shape)
    params_weight_v = self.lambdaa * jnp.ones(self.shape)

    return jnp.array([params_weight_h, params_weight_v])

  def get_init_samples(self, rnd, num_samples: int):
    x0 = jax.random.randint(
        rnd,
        shape=(num_samples,) + self.sample_shape,
        minval=0,
        maxval=self.num_categories,
        dtype=jnp.int32,
    )
    return x0

  def forward(self, params, x):
    if len(x.shape) - 1 == len(self.sample_shape):
      x = jax.nn.one_hot(x, self.num_categories)
      
    w_h = params[0][:, :-1, :]
    w_v = params[1][:-1, :, :]

    loglikelihood = jnp.zeros((x.shape[0],) + self.shape)
    loglikelihood = loglikelihood.at[:, :, :-1].set(
        loglikelihood[:, :, :-1] + x[:, :, :-1] * x[:, :, 1:] * w_h
    )  # right
    loglikelihood = loglikelihood.at[:, :, 1:].set(
        loglikelihood[:, :, 1:] + x[:, :, 1:] * x[:, :, :-1] * w_h
    )  # left
    loglikelihood = loglikelihood.at[:, :-1, :].set(
        loglikelihood[:, :-1, :] + x[:, :-1, :] * x[:, 1:, :] * w_v
    )  # down
    loglikelihood = loglikelihood.at[:, 1:, :].set(
        loglikelihood[:, 1:, :] + x[:, 1:, :] * x[:, :-1, :] * w_v
    )  # up
    
    
    return jnp.sum((loglikelihood).reshape(x.shape[0], -1), axis=-1)

  def get_value_and_grad(self, params, x):
    x = x.astype(jnp.float32)  # int tensor is not differentiable

    def fun(z):
      loglikelihood = self.forward(params, z)
      return jnp.sum(loglikelihood), loglikelihood

    (_, loglikelihood), grad = jax.value_and_grad(fun, has_aux=True)(x)
    return loglikelihood, grad


def build_model(config):
  return Potts(config.model)
