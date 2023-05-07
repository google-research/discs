"""Ising Energy Function."""

from discs.models import abstractmodel
import jax
import jax.numpy as jnp
import ml_collections


class Ising(abstractmodel.AbstractModel):
  """Ising Distribution with Cyclic 2D Lattice."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.shape = config.shape
    self.lambdaa = config.lambdaa
    self.external_field_type = config.external_field_type
    self.init_sigma = config.init_sigma
    self.mu = config.mu

  def make_init_params(self, rnd):
    params = {}
    # # connectivity strength
    # params_weight_h = self.lambdaa * jnp.ones(self.shape)
    # params_weight_v = self.lambdaa * jnp.ones(self.shape)

    # # TODO: Enums
    # # external force (default value is zero)
    # if self.external_field_type == 1:
    #   params_b = (
    #       2 * jax.random.uniform(rnd, shape=self.shape) - 1
    #   ) * self.init_sigma
    #   indices = jnp.indices(self.shape)
    #   inner_outter = self.mu * jnp.where(
    #       (indices[0] / self.shape[0] - 0.5) ** 2
    #       + (indices[1] / self.shape[1] - 0.5) ** 2
    #       < 0.5 / jnp.pi,
    #       1,
    #       -1,
    #   )

    #   params_b += inner_outter
    #   params_b = -1 * params_b
    #   params['params'] = jnp.array([params_weight_h, params_weight_v, params_b])
    #   return params

    # params['params'] = jnp.array([params_weight_h, params_weight_v])
    # return params

    rnd1, _ = jax.random.split(rnd, 2)
    p = self.shape[0]
    self.p = p

    n_w = (
        2 * jax.random.uniform(rnd1, shape=self.shape) - 1
    ) * self.init_sigma
    for i in range(p):
      for j in range(p):
        n_w[i, j] += self._weight((i, j), p, self.mu)
    params['e_w_h'] = -self.lambdaa * jnp.ones([p, p - 1])
    params['e_w_v'] = -self.lambdaa * jnp.ones([p - 1, p])
    params['n_w'] = n_w

    return params
    # # Gaussian Integral Trick
    # # D = 16 * math.fabs(lamda)
    # W = torch.zeros(p ** 2, p ** 2).to(device)
    # for i in range(p):
    #     for j in range(p):
    #         if i > 0:
    #             W[i * p + j, i * p - p + j] = lamda
    #             W[i * p - p + j, i * p + j] = lamda
    #         if j > 0:
    #             W[i * p + j, i * p + j - 1] = lamda
    #             W[i * p - 1 + j, i * p + j] = lamda
    # W = 4 * W
    # a = - 2 * self.n_w.view(-1)
    # D = 1e-3 - torch.linalg.eigh(W)[0].min()
    # WpD = torch.eye(p ** 2).to(device) * D + W
    # self.Ainv = torch.linalg.cholesky(WpD)
    # self.bmD = a - W.sum(dim=1) / 2 - D / 2

    # # for debug
    # self.W = W
    # self.a = - self.n_w.view(-1)

  def _weight(self, n, p, mu):
    if (n[0] / p - 0.5) ** 2 + (n[1] / p - 0.5) ** 2 < 0.5 / jnp.pi:
      return mu
    else:
      return -mu

  def get_init_samples(self, rnd, num_samples: int):
    x0 = jax.random.bernoulli(
        rnd,
        shape=(num_samples,) + self.shape,
    )
    return x0

  def forward(self, params, x):
    batch = x.shape[:-1]
    p = self.shape[0]
    x = x.view(-1, p, p)
    message = self.aggr(params, x)
    message = message / 2 + params['n_w']
    return jnp.reshape(-jnp.sum((2 * x - 1) * message, axis=[-1, -2]), batch)

  def aggr(self, params, x):
    message = jnp.zeros_like(x)
    message[:, :-1, :] += (2 * x[:, 1:, :] - 1) * params['e_w_v']
    message[:, 1:, :] += (2 * x[:, :-1, :] - 1) * params['e_w_v']
    message[:, :, :-1] += (2 * x[:, :, 1:] - 1) * params['e_w_h']
    message[:, :, 1:] += (2 * x[:, :, :-1] - 1) * params['e_w_h']
    return message

  def get_value_and_grad(self, params, x):
    x = x.astype(jnp.float32)  # int tensor is not differentiable

    def fun(z):
      loglikelihood = self.forward(params, z)
      return jnp.sum(loglikelihood), loglikelihood

    (_, loglikelihood), grad = jax.value_and_grad(fun, has_aux=True)(x)
    return loglikelihood, grad


def build_model(config):
  return Ising(config.model)
