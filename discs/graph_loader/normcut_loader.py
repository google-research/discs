"""Load normalized cut graphs."""

import os
from discs.common import utils
from discs.graph_loader import common as data_common
import networkx as nx
import numpy as np
import pickle5 as pickle


class NormCutGen(data_common.GraphGenerator):
  """Generator for mis graphs."""

  def get_dummy_sample(self):
    g = nx.Graph()
    for i in range(self.num_categories):
      g.add_node(i)
    return g, 0

  def graph2edges(self, g):
    params = utils.graph2edges(
        g,
        build_bidir_edges=True,
        has_edge_weights=False,
        padded_num_edges=self._max_num_edges,
    )
    num_nodes = len(g)
    params['num_nodes'] = num_nodes
    params['node_degrees'] = np.array(g.degree)[:, 1].astype(np.float32)
    params['mask'] = np.arange(self.max_num_nodes) < num_nodes
    return params

  def sample_gen(self, phase, repeat):
    raise NotImplementedError


class ComputationGraphs(NormCutGen):
  """Generator for computational graphs."""

  def __init__(self, data_root, model_config):
    super().__init__()
    fname = os.path.join(data_root, 'nets', model_config.rand_type + '.pkl')
    with open(fname, 'rb') as f:
      self.graph = pickle.load(f)
    self.num_categories = model_config.num_categories
    self._max_num_nodes = len(self.graph)
    self._max_num_edges = len(self.graph.edges())
    self._num_instances = 1
    print('max num nodes', self.max_num_nodes)
    print('max num edges', self.max_num_edges)
    print('num instances', self.num_instances)

  def sample_gen(self, phase, repeat=False):
    assert phase == 'test'
    while True:
      yield self.graph, 1.0
      if not repeat:
        break


class RandGraphs(NormCutGen):
  """Generator for random graphs."""

  def __init__(self, data_root, model_config):
    super().__init__()
    data_folder = os.path.join(
        data_root, model_config.graph_type, model_config.rand_type
    )
    file_list = []
    for fname in os.listdir(data_folder):
      if fname.endswith('pkl'):
        file_list.append(os.path.join(data_folder, fname))
    self.file_list = sorted(file_list)
    self._max_num_nodes = model_config.max_num_nodes
    self._max_num_edges = model_config.max_num_edges
    self._num_instances = model_config.num_instances
    self.num_categories = model_config.num_categories
    print('max num nodes', self.max_num_nodes)
    print('max num edges', self.max_num_edges)
    print('num instances', self.num_instances)

  def sample_gen(self, phase, repeat=False):
    assert phase == 'test'
    while True:
      for fname in self.file_list:
        with open(fname, 'rb') as f:
          g = pickle.load(f)
          yield g, 1.0
      if not repeat:
        break
