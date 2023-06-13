"""Common class for graph generator."""

import abc
import jax
import networkx as nx
import numpy as np


def gen_connected(g_type, min_n, max_n, **kwargs):
  """Generate random connected graph."""
  n_tried = 0
  if 'w_type' in kwargs:
    w_type = kwargs['w_type']
    if w_type == 'int' or w_type == 'uniform':
      w_min = kwargs['w_min']
      w_max = kwargs['w_max']
      if w_type == 'int':
        w_min = int(w_min)
        w_max = int(w_max)
        rand_fn = lambda n: np.random.randint(w_min, w_max, n)
      else:
        rand_fn = lambda n: np.random.uniform(w_min, w_max, n)
    elif w_type == 'normal':
      std, mean = kwargs['w_std'], kwargs['w_mean']
      rand_fn = lambda n: np.random.randn(n) * std + mean
    else:
      raise ValueError('Unknown edge weight type %s' % w_type)
  else:
    w_type = None

  while n_tried < 20:
    n_tried += 1
    cur_n = np.random.randint(max_n - min_n + 1) + min_n
    if g_type == 'erdos_renyi':
      g = nx.erdos_renyi_graph(n=cur_n, p=kwargs['er_p'])
      g_idx = max(nx.connected_components(g), key=len)
      gcc = g.subgraph(list(g_idx))
      # generate another graph if this one has fewer nodes than min_n
      if nx.number_of_nodes(gcc) < min_n:
        continue

      max_idx = max(gcc.nodes())
      if max_idx != nx.number_of_nodes(gcc) - 1:
        idx_map = {}
        for idx in gcc.nodes():
          t = len(idx_map)
          idx_map[idx] = t

        g = nx.Graph()
        g.add_nodes_from(range(0, nx.number_of_nodes(gcc)))
        for edge in gcc.edges():
          g.add_edge(idx_map[edge[0]], idx_map[edge[1]])
        gcc = g
      max_idx = max(gcc.nodes())
      assert max_idx == nx.number_of_nodes(gcc) - 1

      # check number of nodes in induced subgraph
      if len(gcc) < min_n or len(gcc) > max_n:
        continue
    elif g_type == 'barabasi_albert':
      gcc = nx.barabasi_albert_graph(n=cur_n, m=kwargs['ba_m'])
    else:
      raise ValueError('Unknown random graph type %s' % g_type)

    if w_type is not None:
      n_edges = len(gcc.edges())
      weights = rand_fn(n_edges)
      edge_list = []
      for i, e in enumerate(gcc.edges()):
        edge_list.append((e[0], e[1], weights[i]))
      g = nx.Graph()
      g.add_nodes_from(range(len(gcc)))
      g.add_weighted_edges_from(edge_list)
      gcc = g
    return gcc


class GraphGenerator(abc.ABC):
  """Abstract class of graph generator."""

  def __init__(self):
    self._max_num_nodes = 0
    self._max_num_edges = 0
    self._num_instances = 0

  def get_dummy_sample(self):
    raise NotImplementedError

  @property
  def max_num_nodes(self):
    return self._max_num_nodes

  @property
  def max_num_edges(self):
    return self._max_num_edges

  @property
  def num_instances(self):
    return self._num_instances

  @abc.abstractmethod
  def sample_gen(self, phase, repeat=False):
    pass

  def get_iterator(self, phase, batch_size, sharding=False):
    """Get sharded/distributed data loader."""
    num_proc = 1
    proc_idx = 0
    local_batch_size = batch_size
    if sharding:
      proc_idx = jax.process_index()
      num_proc = jax.process_count()
      assert batch_size % num_proc == 0
      local_batch_size = batch_size // num_proc
    total_batches = self.num_instances // batch_size
    if self.num_instances % batch_size != 0:
      total_batches += 1
    generator = self.sample_gen(phase, repeat=False)
    num_batches = 0
    buffer = []
    for idx, (g, sol) in enumerate(generator):
      if idx % num_proc == proc_idx:
        buffer.append((idx, self.graph2edges(g), sol))
        if len(buffer) == local_batch_size:
          num_batches += 1
          yield buffer
          buffer = []
    has_dummy = False
    for _ in range(num_batches, total_batches):
      if not has_dummy:
        has_dummy = True
        dummy_g, dummy_sol = self.get_dummy_sample()
        dummy_param = self.graph2edges(dummy_g)
      pad_size = local_batch_size - len(buffer)
      for _ in range(pad_size):
        buffer.append((-1, dummy_param, dummy_sol))
      yield buffer
      buffer = []

  @abc.abstractmethod
  def graph2edges(self, g):
    raise NotImplementedError
