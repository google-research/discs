"""Normalized cut model."""

from discs.models import comb_ebm
import jax
import jax.numpy as jnp
import ml_collections


class NormCut(comb_ebm.CombEBM):
  """Normalized Cut model."""

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__(config)
    self.config = config.model
    self.penalty = self.config.penalty
    self.num_nodes = self.config.max_num_nodes
    self.num_edges = self.config.max_num_edges
    self.num_categories = self.config.num_categories
    self.avg_partition_size = self.num_nodes / self.num_categories
    if self.penalty > 0.0:
      self.tolerance = self.num_nodes * self.config.tolerance
      self.stype = self.config.get('stype', 'span')
    self.eps = 1e-8


  def make_init_params(self, rng):
    try:
      data_list = next(self.datagen)
    except:
      return None
    return data_list
  
  def get_init_samples(self, rng, num_samples):
    rng = jax.random.split(rng, num_samples)
    def init_fn(rng):
      base = jnp.arange(self.num_categories, dtype=jnp.int32)
      rng, shuffle_rng = jax.random.split(rng)
      more = jax.random.randint(
          rng, shape=(self.num_nodes - self.num_categories,),
          minval=0, maxval=self.num_categories)
      joint = jnp.concatenate([base, more], axis=0)
      return jax.random.shuffle(shuffle_rng, joint)
    return jax.vmap(init_fn)(rng)

  def get_cut_sizes(self, params, x):
    if x.ndim < 3:
      x = jax.nn.one_hot(x, self.num_categories, dtype=jnp.float32)
    edge_from = params['edge_from']
    edge_to = params['edge_to']
    gather2src = x[:, edge_from]
    gather2dst = x[:, edge_to]

    edge_sum = gather2src + gather2dst
    cut_sizes = jnp.sum(-edge_sum ** 2 + 2 * edge_sum, axis=1)
    return cut_sizes

  def objective(self, params, x):
    if x.ndim < 3:
      x = jax.nn.one_hot(x, self.num_categories, dtype=jnp.float32)
    cut_sizes = self.get_cut_sizes(params, x)

    if self.penalty > 0:
      partition_sizes = jnp.sum(x, axis=1)
      if self.stype == 'quad':
        ncost = jnp.sum((partition_sizes - self.avg_partition_size) ** 2, -1)
      else:
        ub_cost = jax.nn.relu(
            partition_sizes - self.avg_partition_size - self.tolerance) ** 2
        lb_cost = jax.nn.relu(
            self.avg_partition_size - self.tolerance - partition_sizes) ** 2
        ncost = jnp.sum(ub_cost + lb_cost, axis=-1)
      cost = jnp.sum(cut_sizes, axis=-1) + self.penalty * ncost
    else:
      volume = jnp.sum(x * jnp.expand_dims(params['node_degrees'], 1), axis=1)
      cost = jnp.sum((cut_sizes + self.eps) / (volume + self.eps ** 2),
                     axis=-1)
    return -cost

  def forward(self, params, x):
    obj = self.objective(params, x)
    return obj / params['temperature']

  def logratio_in_neighborhood(self, params, x):
    if x.ndim < 3:
      x = jax.nn.one_hot(x, self.num_categories, dtype=jnp.int32)
    x = x.astype(jnp.int32)
    bsize = x.shape[0]
    edge_from = params['bidir_edge_from']
    edge_to = params['bidir_edge_to']
    node_degrees = jnp.expand_dims(params['node_degrees'], axis=1)
    gather2src = x[:, edge_from]
    gather2dst = x[:, edge_to]
    cur_edge_cut = (gather2src + gather2dst) % 2
    cur_cut_sizes = jnp.sum(cur_edge_cut, axis=1).astype(jnp.float32) / 2.0

    new_choices = jnp.eye(self.num_categories, dtype=jnp.int32)
    new_choices = jnp.expand_dims(new_choices, (0, 1))
    edge_info = (new_choices + jnp.expand_dims(gather2dst, axis=-2)) % 2
    edge_delta = edge_info - jnp.expand_dims(cur_edge_cut, axis=-2)
    diff_arr = jnp.zeros(
        (bsize, self.num_nodes, self.num_categories, self.num_categories),
        dtype=jnp.int32)
    diff_arr = diff_arr.at[jnp.expand_dims(jnp.arange(bsize), (1, 2)),
                           jnp.expand_dims(edge_from, -1),
                           jnp.arange(self.num_categories)].add(edge_delta)
    new_cut_sizes = jnp.expand_dims(cur_cut_sizes, (1, 2)) + diff_arr

    diff_volume = (new_choices - jnp.expand_dims(x, axis=-2)) * jnp.expand_dims(
        node_degrees, axis=-1)
    cur_volume = jnp.sum(x * node_degrees, axis=1)
    new_volume = jnp.expand_dims(cur_volume, (1, 2)) + diff_volume
    new_obj = -jnp.sum((new_cut_sizes + self.eps) / (
        new_volume + self.eps ** 2), axis=-1) / params['temperature']
    cur_obj = -jnp.sum((cur_cut_sizes + self.eps) / (
        cur_volume + self.eps ** 2), axis=-1) / params['temperature']
    log_ratio = new_obj - jnp.expand_dims(cur_obj, axis=(1, 2))
    return cur_obj, log_ratio, 1, None


def build_model(config):
  return NormCut(config)
