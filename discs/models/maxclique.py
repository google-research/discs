"""Max Clique model."""

from discs.models import comb_ebm
import jax.numpy as jnp
import ml_collections


class MaxClique(comb_ebm.BinaryNodeCombEBM):
  """Max Clique model."""

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__(config)
    self.config = config.model
    self.max_num_nodes = self.config.max_num_nodes
    self.penalty_coeff = self.config.get('penalty', 2.0)

  def make_init_params(self, rng):
    try:
      data_list = next(self.datagen)
    except:
      return None
    return data_list

  def penalty(self, params, x):
    x = x * params['mask']
    edge_mask = params['edge_mask']
    edge_from = params['edge_from']
    edge_to = params['edge_to']
    gather2src = x[:, edge_from]
    gather2dst = x[:, edge_to]

    clique_size = jnp.sum(x, axis=1)
    total_edges_needed = clique_size * (clique_size - 1) / 2
    covered_edges = jnp.sum(gather2src * gather2dst * edge_mask, axis=1)
    penalty = self.penalty_coeff * (total_edges_needed - covered_edges)
    return penalty

  def objective(self, params, x):
    _ = params
    x = x * params['mask']
    return jnp.sum(x, axis=1)


def build_model(config):
  return MaxClique(config)
