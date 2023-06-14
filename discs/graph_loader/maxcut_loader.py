"""Graph loader for maxcut class."""

import os
from discs.common import utils
from discs.graph_loader import common as data_common
import networkx as nx
import numpy as np
import pickle5 as pickle


class MaxcutGen(data_common.GraphGenerator):
  """Generator for maxcut graphs."""

  def graph2edges(self, g):
    return utils.graph2edges(
        g, build_bidir_edges=True, padded_num_edges=self._max_num_edges
    )

  def get_dummy_sample(self):
    g = nx.Graph()
    g.add_edge(0, 1, weight=1.0)
    return g, 1.0


class RandGraphGen(MaxcutGen):
  """Generator for random graphs."""

  def __init__(self, data_root, model_config):
    super().__init__()
    data_folder = os.path.join(data_root, 'maxcut-%s' % model_config.graph_type)
    data_folder = os.path.join(
        data_folder, 'maxcut-%s' % model_config.rand_type
    )
    file_list = []
    for fname in os.listdir(data_folder):
      if fname.startswith('test-'):
        file_list.append(os.path.join(data_folder, fname))
    self.file_list = sorted(file_list)
    if (
        model_config.max_num_nodes > 0
        and model_config.max_num_edges > 0
        and model_config.num_instances > 0
    ):
      self._max_num_nodes = model_config.max_num_nodes
      self._max_num_edges = model_config.max_num_edges
      self._num_instances = model_config.num_instances
    else:
      for fname in self.file_list:
        with open(fname, 'rb') as f:
          while True:
            try:
              data = pickle.load(f)
              g = data[0]
              self._max_num_nodes = max(self._max_num_nodes, len(g))
              self._max_num_edges = max(self._max_num_edges, len(g.edges()))
              self._num_instances += 1
            except EOFError:
              break
    print('max num nodes', self.max_num_nodes)
    print('max num edges', self.max_num_edges)
    print('num instances', self.num_instances)

  def sample_gen(self, phase, repeat=False):
    assert phase == 'test'
    while True:
      for fname in self.file_list:
        with open(fname, 'rb') as f:
          while True:
            try:
              data = pickle.load(f)
              g, sol = data[0], data[1][0]
              yield g, sol
            except EOFError:
              break
      if not repeat:
        break


class OptsicomStatic(MaxcutGen):
  """Generate grpahs from optsicom dataset."""

  def __init__(self, data_root):
    super().__init__()
    fname = os.path.join(data_root, 'optsicom', 'b.pkl')
    self.graphs = []
    with open(fname, 'rb') as f:
      while True:
        try:
          data = pickle.load(f)
          g = data[0]
          sol = data[1][0]
          self.graphs.append((g, sol))
          self._max_num_nodes = max(self._max_num_nodes, len(g))
          self._max_num_edges = max(self._max_num_edges, len(g.edges()))
          self._num_instances += 1
        except EOFError:
          break
    print('max num nodes', self.max_num_nodes)
    print('max num edges', self.max_num_edges)
    print('num instances', self.num_instances)

  def sample_gen(self, phase, repeat=False):
    assert phase == 'test'
    while True:
      for g, sol in self.graphs:
        yield g, sol
      if not repeat:
        break


class OptsicomGen(MaxcutGen):
  """Generate grpahs from optsicom dataset."""

  def __init__(self, rand_type):
    super().__init__()
    file_list = [
        'sg3dl051000.mc',
        'sg3dl052000.mc',
        'sg3dl053000.mc',
        'sg3dl054000.mc',
        'sg3dl055000.mc',
        'sg3dl056000.mc',
        'sg3dl057000.mc',
        'sg3dl058000.mc',
        'sg3dl059000.mc',
        'sg3dl0510000.mc',
    ]
    self.graphs = []
    if len(rand_type) > 1 and rand_type[-1] == 'b':
      self.bias = 1
      rand_type = rand_type[:-1]
    else:
      self.bias = 0
    if rand_type == 'b':
      self.rand_type = rand_type
    else:
      self.rand_type = float(rand_type)
    folder = os.path.dirname(os.path.realpath(__file__))
    self._max_num_nodes = 0
    self._max_num_edges = 0
    for idx in range(len(file_list)):
      fname = '%s/optsicom/%s' % (folder, file_list[idx])
      g = nx.Graph()
      with open(fname, 'r') as f:
        line = f.readline()
        num_nodes, num_edges = [int(w) for w in line.strip().split(' ')]
        for row in f:
          x, y, w = [int(t) for t in row.strip().split(' ')]
          x -= 1
          y -= 1
          w = float(w)
          g.add_edge(x, y, weight=w)
      assert len(g) == num_nodes
      assert len(g.edges()) == num_edges
      self._max_num_nodes = max(self._max_num_nodes, num_nodes)
      self._max_num_edges = max(self._max_num_edges, num_edges)
      self.graphs.append(g)
    self._num_instances = len(self.graphs)

  def gen_graph(self, idx=-1, rand_weight=False):
    """Generate a graph with random weights."""
    if idx < 0:
      idx = np.random.randint(len(self.graphs))
    orig = self.graphs[idx]
    g = nx.Graph()
    g.add_nodes_from(range(len(g)))
    for e in orig.edges(data=True):
      x, y, w = e[0], e[1], e[2]['weight']
      if rand_weight:
        if self.rand_type == 'b':
          w = np.random.randint(2) * 2.0 - 1.0
        else:
          t = np.random.randn() * self.rand_type
          w += t
        w += self.bias
      g.add_edge(x, y, weight=w)
    return g

  def sample_gen(self, phase, repeat=False):
    assert phase == 'test'
    while True:
      for idx in range(len(self.graphs)):
        g = self.gen_graph(idx)
        yield g, 1.0
      if not repeat:
        break
