"""EBM for combinatorial optimization problems."""

from discs.common.utils import get_datagen
from discs.models import abstractmodel
import jax
import jax.numpy as jnp
import ml_collections


class CombEBM(abstractmodel.AbstractModel):
  """Comb-opt EBM."""

  def __init__(self, config: ml_collections.ConfigDict):
    # loads the data from model config data_root
    self.datagen = get_datagen(config)

  def make_init_params(self, rng):
    raise NotImplementedError

  def objective(self, params, x):
    raise NotImplementedError

  def penalty(self, params, x):
    return 0.0

  def forward(self, params, x):
    x = x.astype(jnp.float32)
    obj = self.objective(params, x) - self.penalty(params, x)
    return obj / params['temperature']

  def get_value_and_grad(self, params, x):
    x = x.astype(jnp.float32)  # int tensor is not differentiable

    def fun(z):
      loglikelihood = self.forward(params, z)
      return jnp.sum(loglikelihood), loglikelihood

    (_, loglikelihood), grad = jax.value_and_grad(fun, has_aux=True)(x)
    return loglikelihood, grad

  def evaluate(self, params, x):
    return self.forward(params, x) * params['temperature']


class BinaryNodeCombEBM(CombEBM):
  """Comb-opt EBM for binary cases with node variables."""

  def get_init_samples(self, rng, num_samples):
    return jax.random.bernoulli(
        key=rng, p=0.5, shape=(num_samples, self.max_num_nodes)
    ).astype(jnp.int32)

  def get_neighbor_fn(self, _, x, neighbhor_idx):
    brange = jnp.arange(x.shape[0])
    cur_val = x[brange, neighbhor_idx]
    y = x.at[brange, neighbhor_idx].set(1 - cur_val)
    return y
