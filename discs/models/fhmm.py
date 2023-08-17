"""FHMM Factorized Energy Function."""

import functools
from discs.models import abstractmodel
import jax
import jax.numpy as jnp
import ml_collections


class FHMM(abstractmodel.AbstractModel):
  """FHMM class."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.shape = config.shape
    self.num_categories = config.num_categories
    self.l = self.shape[0]
    self.k = self.shape[1]
    self.sigma = config.sigma
    self.alpha = config.alpha
    self.beta = config.beta


  def get_value_and_grad(self, params, x):
    x = x.astype(jnp.float32)  # int tensor is not differentiable

    def fun(z):
      loglikelihood = self.forward(params, z)
      return jnp.sum(loglikelihood), loglikelihood

    (_, loglikelihood), grad = jax.value_and_grad(fun, has_aux=True)(x)

    return loglikelihood, grad


class BinaryFHMM(FHMM):
  """FHMM Distribution."""

  def make_init_params(self, rnd):
    rnd = rnd.reshape(-1)
    rng1, rng2, rng3, rng4 = jax.random.split(rnd, 4)
    x = self.sample_X(rng1)
    w = jax.random.normal(rng2, (self.k, 1))
    b = jax.random.normal(rng3, (1, 1))
    y = self.sample_Y(rng4, x, w, b)
    params = {}
    params['w'] = w
    params['b'] = b
    params['y'] = y
    return params

  def sample_X(self, rng):
    x = jnp.ones([self.l, self.k])
    x = x.at[0].set(jax.random.bernoulli(rng, p=x[0] * self.alpha))
    for l in range(1, self.l):
      rng, _ = jax.random.split(rng)
      p = self.beta * x[l - 1] + (1 - self.beta) * (1 - x[l - 1])
      x = x.at[l].set(jax.random.bernoulli(rng, p))
    return x

  def sample_Y(self, rng, x, w, b):
    return jax.random.normal(rng, (self.l, 1)) * self.sigma + x @ w + b

  def log_probab_of_px(self, x, p):
    return jnp.log(p) * x + jnp.log(1 - p) * (1 - x)

  def get_init_samples(self, rng, num_samples: int):
    x0 = jax.random.bernoulli(
        rng,
        shape=(num_samples,) + (self.l, self.k),
    ).astype(jnp.int32)
    return x0

  def forward(self, params, x):
    x = x.reshape(-1, self.l, self.k)
    w = params['w']
    b = params['b']
    y = params['y']
    logp_y = -jnp.sum(jnp.square(y - (x @ w) - b), [-1, -2]) / (
        2 * self.sigma**2
    )
    x_0 = x[:, 0, :]
    x_cur = x[:, :-1, :]
    x_next = x[:, 1:, :]
    x_c = x_cur * (1 - x_next) + (1 - x_cur) * x_next
    logp_x = jnp.sum(self.log_probab_of_px(x_0, self.alpha), -1) + jnp.sum(
        self.log_probab_of_px(x_c, 1 - self.beta), [-1, -2]
    )
    loglikelihood = logp_x + logp_y
    return loglikelihood


class CategFHMM(FHMM):
  """FHMM Distribution."""

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__(config)
    beta = jnp.eye(self.num_categories) * (
        self.beta - (1 - self.beta) / (self.num_categories - 1)
    )
    beta += (1 - beta) / (self.num_categories - 1)
    self.beta = beta
    self.log_beta = jnp.log(beta)

  def get_init_samples(self, rng, num_samples: int):
    x0 = jax.random.categorical(
        rng,
        logits=jnp.log(jnp.ones([self.num_categories])),
        shape=(num_samples, self.l, self.k),
    ).astype(jnp.int32)
    print('init sample shapes: ', x0.shape)
    return x0

  def make_init_params(self, rnd):
    rnd = rnd.reshape(-1)
    rng1, rng2, rng3, rng4 = jax.random.split(rnd, 4)
    alpha = jax.random.uniform(rng1, [self.num_categories])
    alpha = alpha.at[0].set(1 - self.alpha)
    alpha = alpha.at[1:].set(self.alpha * alpha[1:] / jnp.sum(alpha[1:]))
    alpha_logits = jnp.log(alpha)
    self.p_x0 = functools.partial(jax.random.categorical, logits=alpha_logits)
    x = self.sample_X(rng1)
    w = jax.random.normal(rng2, (self.k, self.num_categories))  # [k, n]
    b = jax.random.normal(rng3, (1, 1))  # [1, 1]
    y = self.sample_Y(rng4, x, w, b)  # [L, 1]
    params = {}
    params['w'] = w
    params['b'] = b
    params['y'] = y
    params['alpha'] = alpha_logits
    return params

  def sample_X(self, rng):
    x = jnp.ones([self.l, self.k, self.num_categories])
    val = self.p_x0(rng, shape=(self.k,))
    val = jax.nn.one_hot(val, self.num_categories)
    x = x.at[0].set(val)
    for l in range(1, self.l):
      rng, _ = jax.random.split(rng)
      dist0 = functools.partial(
          jax.random.categorical, logits=jnp.log(x[l - 1] @ self.beta)
      )
      val = dist0(rng, shape=(self.k,))
      val = jax.nn.one_hot(val, self.num_categories)
      x = x.at[l].set(val)
    return x

  def sample_Y(self, rng, x, w, b):
    return (
        jax.random.normal(rng, (self.l,)) * self.sigma
        + jnp.sum(x * w, [-1, -2])
        + b
    )

  def log_probab_of_px_categ(self, x, alpha):
    log_powers = jnp.sum(x * alpha, axis=-1)
    log_factorial_n = jax.lax.lgamma(jnp.sum(x, -1) + 1)
    log_factorial_xs = jnp.sum(jax.lax.lgamma(x + 1), -1)
    return log_factorial_n - log_factorial_xs + log_powers

  def forward(self, params, x):
    w = params['w']
    b = params['b']
    y = params['y']
    alpha = params['alpha']

    if x.shape[-1] != self.num_categories:
      x = jax.nn.one_hot(x, self.num_categories)
    x = x.reshape(-1, self.l, self.k, self.num_categories)
    x_0 = x[:, 0]
    x_cur = x[:, :-1]
    x_next = x[:, 1:]

    logp_0 = jnp.sum(self.log_probab_of_px_categ(x_0, alpha), 1)
    logp_x = jnp.sum(
        (
            jnp.expand_dims(x_cur, -1)
            * jnp.expand_dims(x_next, -2)
            * self.log_beta
        ),
        [1, 2, 3, 4],
    )
    logp_y = -jnp.sum(jnp.square(y - jnp.sum(x * w, [2, 3]) - b), 1) / (
        2 * self.sigma**2
    )

    loglikelihood = logp_x + logp_y + logp_0
    return loglikelihood


def build_model(config):
  if config.model.num_categories == 2:
    return BinaryFHMM(config.model)
  return CategFHMM(config.model)
