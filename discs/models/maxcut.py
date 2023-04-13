"""Maxcut model."""

from discs.models import comb_ebm
import jax
import jax.numpy as jnp
import ml_collections


class Maxcut(comb_ebm.BinaryNodeCombEBM):
  """Maxcut model."""

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__(config)
    self.config = config.model
    self.max_num_nodes = self.config.max_num_nodes

  def make_init_params(self, rng):
    try:
      data_list = next(self.datagen)
    except:
      return None
    return data_list

  def get_init_samples(self, rng, num_samples):
    return jax.random.bernoulli(
        key=rng, p=0.5, shape=(num_samples, self.max_num_nodes)
    ).astype(jnp.int32)

  def objective(self, params, x):
    edge_from = params['edge_from']
    edge_to = params['edge_to']
    edge_weights = params['edge_weight'][None]
    x = x * 2 - 1
    gather2src = x[:, edge_from]
    gather2dst = x[:, edge_to]
    is_cut = (1 - gather2src * gather2dst) / 2.0
    cut_weight = jnp.sum(is_cut * edge_weights, axis=1)
    return cut_weight

  def logratio_in_neighborhood(self, params, x):
    edge_from = params['bidir_edge_from']
    edge_to = params['bidir_edge_to']
    edge_weights = params['bidir_edge_weight'][None]

    gather2src = x[:, edge_from]
    gather2dst = x[:, edge_to]
    is_cut = (gather2src - gather2dst) ** 2
    cut_weight = (
        jnp.sum(is_cut * edge_weights, axis=1) / 2.0 / params['temperature']
    )
    diff_weight = edge_weights * (1 - 2 * gather2dst)
    diff_arr = jnp.zeros(x.shape, dtype=edge_weights.dtype)
    diff_arr = diff_arr.at[
        jnp.expand_dims(jnp.arange(x.shape[0]), axis=1), edge_from
    ].add(diff_weight)
    sign = 1 - x * 2
    logratio = diff_arr * sign / params['temperature']
    return cut_weight, logratio, 1, self.get_neighbor_fn


def build_model(config):
  return Maxcut(config)
