"""Potts Energy Function."""

from discs.models import abstractmodel
import jax
import jax.numpy as jnp
import ml_collections


class Potts(abstractmodel.AbstractModel):
  """Potts Distribution (2D cyclic ising model with one-hot representation)."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.sample_shape = config.shape
    self.lambdaa = config.lambdaa
    self.num_categories = config.num_categories
    self.shape = self.sample_shape + (self.num_categories,)
    self.mu = config.mu
    self.init_sigma = config.init_sigma

  def inner_or_outter(self, n, shape):
    if (n[0] / shape - 0.5) ** 2 + (n[1] / shape - 0.5) ** 2 < 0.5 / jnp.pi:
      return 1
    else:
      return -1

  def make_init_params(self, rnd):
    params = {}
    # connectivity strength
    params_weight_h = -self.lambdaa * jnp.ones(self.shape)
    params_weight_v = -self.lambdaa * jnp.ones(self.shape)

    params_b = (
        2 * jax.random.uniform(rnd, shape=self.shape) - 1
    ) * self.init_sigma
    indices = jnp.indices(self.shape)
    inner_outter = self.mu * jnp.where(
        (indices[0] / self.shape[0] - 0.5) ** 2
        + (indices[1] / self.shape[1] - 0.5) ** 2
        < 0.5 / jnp.pi,
        1,
        -1,
    )

    params_b += inner_outter
    params['params'] = jnp.array([params_weight_h, params_weight_v, params_b])
    return params


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
    params = params['params']
    if len(x.shape) - 1 == len(self.sample_shape):
      x = jax.nn.one_hot(x, self.num_categories)
    w_h = params[0][:, :-1, :]
    w_v = params[1][:-1, :, :]

    loglikelihood = jnp.zeros_like(x)
    loglikelihood = loglikelihood.at[:, :, :-1].set(
        loglikelihood[:, :, :-1] + x[:, :, 1:] * w_h
    )  # right
    loglikelihood = loglikelihood.at[:, :, 1:].set(
        loglikelihood[:, :, 1:] + x[:, :, :-1] * w_h
    )  # left
    loglikelihood = loglikelihood.at[:, :-1, :].set(
        loglikelihood[:, :-1, :] + x[:, 1:, :] * w_v
    )  # down
    loglikelihood = loglikelihood.at[:, 1:, :].set(
        loglikelihood[:, 1:, :] + x[:, :-1, :] * w_v
    )  # up
    w_b = params[2]
    loglikelihood = loglikelihood / 2 + w_b
    loglike = x * loglikelihood
    loglike = loglike.reshape(x.shape[0], -1)
    return -jnp.sum(loglike, axis=-1)
  
  def get_value_and_grad(self, params, x):
    x = x.astype(jnp.float32)  # int tensor is not differentiable

    def fun(z):
      loglikelihood = self.forward(params, z)
      return jnp.sum(loglikelihood), loglikelihood

    (_, loglikelihood), grad = jax.value_and_grad(fun, has_aux=True)(x)
    return loglikelihood, grad


def build_model(config):
  return Potts(config.model)
