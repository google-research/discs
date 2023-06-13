"""Load MIS graphs."""

import os
from discs.common import utils
from discs.graph_loader import common as data_common
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pickle5 as pickle
from pysat.formula import CNF


class MISGen(data_common.GraphGenerator):
  """Generator for mis graphs."""

  def get_dummy_sample(self):
    g = nx.Graph()
    g.add_edge(0, 1)
    return g, 1

  def graph2edges(self, g):
    params = utils.graph2edges(
        g,
        build_bidir_edges=True,
        has_edge_weights=False,
        padded_num_edges=self._max_num_edges,
    )
    num_nodes = len(g)
    params['num_nodes'] = num_nodes
    params['mask'] = jnp.arange(self.max_num_nodes) < num_nodes
    return params


class ErTestGraphGen(MISGen):
  """Generator for ErdosRenyi test graphs."""

  def __init__(self, data_root, model_config):
    super().__init__()
    data_folder = os.path.join(data_root, 'er_%s_test' % model_config.rand_type)
    file_list = []
    for fname in os.listdir(data_folder):
      if fname.startswith('ER'):
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
          g = pickle.load(f)
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
          g = pickle.load(f)
          obj = 0
          for node in g.nodes(data=True):
            if 'label' in node[1]:
              obj += node[1]['label']
          if obj == 0:
            obj = 1
          yield g, obj
      if not repeat:
        break


class ErDensityGraphGen(MISGen):
  """Generator for ErdosRenyi test graphs with different densities."""

  def __init__(self, data_root, model_config):
    super().__init__()
    data_folder = os.path.join(data_root, 'er_700_800')
    fname = os.path.join(
        data_folder, 'ER-700-800-%s.pkl' % model_config.rand_type
    )
    with open(fname, 'rb') as f:
      g_list = pickle.load(f)
      for g in g_list:
        self._max_num_nodes = max(self._max_num_nodes, len(g))
        self._max_num_edges = max(self._max_num_edges, len(g.edges()))
        self._num_instances += 1
      self.g_list = g_list
    print('max num nodes', self.max_num_nodes)
    print('max num edges', self.max_num_edges)
    print('num instances', self.num_instances)

  def sample_gen(self, phase, repeat=False):
    assert phase == 'test'
    while True:
      for g in self.g_list:
        yield g, 1
      if not repeat:
        break


class SatLibGraphGen(MISGen):
  """Generator for SATLIB test graphs."""

  def cnf2graph(self, cnf):
    nv = cnf.nv
    clauses = list(filter(lambda x: x, cnf.clauses))
    ind = {
        k: []
        for k in np.concatenate([np.arange(1, nv + 1), -np.arange(1, nv + 1)])
    }
    edges = []
    for i, clause in enumerate(clauses):
      a = clause[0]
      b = clause[1]
      c = clause[2]
      aa = 3 * i + 0
      bb = 3 * i + 1
      cc = 3 * i + 2
      ind[a].append(aa)
      ind[b].append(bb)
      ind[c].append(cc)
      edges.append((aa, bb))
      edges.append((aa, cc))
      edges.append((bb, cc))

    for i in np.arange(1, nv + 1):
      for u in ind[i]:
        for v in ind[-i]:
          edges.append((u, v))
    g = nx.from_edgelist(edges)
    return g

  def __init__(self, data_root, model_config):
    super().__init__()
    data_folder = os.path.join(data_root, 'satlib_test')
    file_list = []
    for fname in os.listdir(data_folder):
      if fname.endswith('.cnf'):
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
        cnf = CNF(fname)
        g = self.cnf2graph(cnf)
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
        cnf = CNF(fname)
        g = self.cnf2graph(cnf)
        yield g, 1.0
      if not repeat:
        break
