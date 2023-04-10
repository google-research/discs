"""EBM for combinatorial optimization problems."""

from discs.models import abstractmodel
from discs.common.utils import get_datagen
import jax
import ml_collections
import jax.numpy as jnp


class CombEBM(abstractmodel.AbstractModel):
  """Comb-opt EBM."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.datagen = get_datagen(config)
    print("***************= ", len(self.datagen))
    print("After Data Gen................................")

  def make_init_params(self, rng):
    print("Calling Make initttt................................")
    
    try:
      data_list = next(self.datagen)
    except:
      print("In execption................................")
      return None
    return data_list

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
