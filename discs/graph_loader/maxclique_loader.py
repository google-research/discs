"""Load MazClique graphs."""

import os
import pickle5 as pickle
import jax.numpy as jnp
import networkx as nx
import numpy as np
from discs.common import utils
from discs.graph_loader import common as data_common


class MaxCliqueGen(data_common.GraphGenerator):
  """Generator for mis graphs."""

  def get_dummy_sample(self):
    g = nx.Graph()
    g.add_edge(0, 1)
    return g, 2

  def graph2edges(self, g):
    params = utils.graph2edges(
        g, build_bidir_edges=True, has_edge_weights=False,
        padded_num_edges=self._max_num_edges)
    num_nodes = len(g)
    params['num_nodes'] = num_nodes
    params['mask'] = jnp.arange(self.max_num_nodes) < num_nodes
    return params


class RBTestGraphGen(MaxCliqueGen):
  """Generator for RB test graphs."""

  def __init__(self, data_root, model_config):
    super().__init__()
    data_folder = os.path.join(data_root, 'RB_test')
    file_list = []
    for fname in os.listdir(data_folder):
      if fname.startswith('RB'):
        file_list.append(os.path.join(data_folder, fname))
    self.file_list = sorted(file_list)
    if (model_config.max_num_nodes > 0 and model_config.max_num_edges > 0 and
        model_config.num_instances > 0):
      self._max_num_nodes = model_config.max_num_nodes
      self._max_num_edges = model_config.max_num_edges
      self._num_instances = model_config.num_instances
    else:
      for fname in self.file_list:
        with open(fname, 'rb') as f:
          g_list = pickle.load(f)
          for _, g in g_list:
            self._max_num_nodes = max(self._max_num_nodes, len(g))
            self._max_num_edges = max(self._max_num_edges, len(g.edges()))
            self._num_instances += 1
    print('max num nodes', self.max_num_nodes)
    print('max num edges', self.max_num_edges)
    print('num instances', self.num_instances)

  def sample_gen(self, phase, repeat=False):
    assert phase == 'test'
    while True:
      for fname in self.file_list:
        with open(fname, 'rb') as f:
          g_list = pickle.load(f)
          for obj, g in g_list:
            yield g, obj
      if not repeat:
        break


class TwitterGraphs(MaxCliqueGen):
  """Generator for twitter test graphs."""

  def __init__(self, data_root, model_config):
    super().__init__()
    data_folder = os.path.join(data_root, 'twitter')
    file_list = []
    for fname in os.listdir(data_folder):
      if fname.startswith('twitter'):
        file_list.append(os.path.join(data_folder, fname))
    self.file_list = sorted(file_list)
    if (model_config.max_num_nodes > 0 and model_config.max_num_edges > 0 and
        model_config.num_instances > 0):
      self._max_num_nodes = model_config.max_num_nodes
      self._max_num_edges = model_config.max_num_edges
      self._num_instances = model_config.num_instances
    else:
      for fname in self.file_list:
        with open(fname, 'rb') as f:
          g_list = pickle.load(f)
          for g in g_list[0]:
            self._max_num_nodes = max(self._max_num_nodes, len(g))
            self._max_num_edges = max(self._max_num_edges, len(g.edges()))
            self._num_instances += 1
    print('max num nodes', self.max_num_nodes)
    print('max num edges', self.max_num_edges)
    print('num instances', self.num_instances)

  def sample_gen(self, phase, repeat=False):
    assert phase == 'test'
    while True:
      for fname in self.file_list:
        with open(fname, 'rb') as f:
          data_list = pickle.load(f)
          for g, obj in zip(data_list[0], data_list[1]):
            yield g, obj
      if not repeat:
        break