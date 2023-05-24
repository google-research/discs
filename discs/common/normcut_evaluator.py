"""Evaluate using GAP code."""

import os
import sys
import networkx as nx
import numpy as np
import pickle5 as pickle
import flags


flags.DEFINE_string('evaluation_type', 'co', 'where results are being saved')
flags.DEFINE_string('key', 'name', 'what key to plot against')

def bincount(groups):
  y = np.bincount(groups)
  ii = np.nonzero(y)[0]
  return np.vstack((ii, y[ii])).T


def num_cut(groups, num_nodes, adjacency_list):
  edge_cut = 0.0
  for i in range(0, num_nodes):
    for j in adjacency_list[i]:
      if groups[i] != groups[j]:
        edge_cut += 1
  return edge_cut / 2.0


def num_cut_adj_mat(groups, num_nodes, adjacency_matrix, sparse):
  """find the number of cutting edges among clusters/groups.

  Args:
    groups: group of each node in graph.
    num_nodes: number of nodes in graph.
    adjacency_matrix: adjacency matrix of the graph.
    sparse: if True the adjacency_matrix is scipy csr matrix.

  Returns:
    number of cutting edges. Each edge counts as two directed edges.
  """
  edge_cut = 0.0
  for i in range(0, num_nodes):
    try:
      if sparse:
        neighbors = np.nonzero(adjacency_matrix[i, :] == 1.0)[1]
      else:
        neighbors = np.nonzero(adjacency_matrix[i, :] == 1.0)[0]
    except:  # pylint: disable=bare-except
      print(i, 'no neighbor')
      continue
    for j in neighbors:
      if groups[i] != groups[j]:
        edge_cut += 1
  return edge_cut


def calc_volume(groups, num_nodes, adjacency_matrix, num_groups):
  volumes = np.zeros(num_groups)
  for i in range(0, num_nodes):
    group = groups[i]
    volumes[group] += np.sum(adjacency_matrix[i, :])
  return volumes


def evaluate(groups,
             num_nodes,
             sum_degrees,
             adjacency_matrix,
             num_groups,
             sparse=False,
             stat=None):
  """evaluate the result of clustering.

  Args:
    groups: result of clustering.
    num_nodes: number of nodes.
    sum_degrees: twice the number of edges.
    adjacency_matrix: adjacency matrix of the graph.
    num_groups: number of clusters.
    sparse: true if adjacency matrix is aparse csr matrix.
    stat: total time in hmetis or the loss in nn.

  Returns:
    return the score by which to evaluate the model.

  """
  bins = bincount(groups)
  cuts = num_cut_adj_mat(groups, num_nodes, adjacency_matrix, sparse)
  cut_ratio = cuts / sum_degrees
  volumes = calc_volume(groups, num_nodes, adjacency_matrix, num_groups)
  outer_prod = np.outer(bins[:, 1], bins[:, 1])
  np.fill_diagonal(outer_prod, 0)
  balanceness = (np.sum(outer_prod) * num_groups) / (
      (num_groups - 1) * (num_nodes**2))
  error_vol = np.sum(
      np.absolute(volumes - sum_degrees / num_groups)) / sum_degrees
  result = str(stat) + ' ' + str(cuts / 2) + ' ' + str(
      bins[:, 1]) + ' ' + str(volumes) + ' ' + str(cut_ratio) + ' ' + str(
          balanceness) + ' ' + str(error_vol) + '\n'
  return cut_ratio, balanceness, result


if __name__ == '__main__':
  ng = 3
  graph_file = './sco/nets/INCEPTION.pkl'
  result_pkl = './discs/results/normcut/results.pkl'
  gfile_list = []
  if os.path.isdir(graph_file):
    fnames = os.listdir(graph_file)
    for fname in fnames:
      if fname.endswith('pkl'):
        gfile_list.append(os.path.join(graph_file, fname))
  else:
    gfile_list.append(graph_file)
  glist = []
  for graph_file in gfile_list:
    with open(graph_file, 'rb') as f:
      g = pickle.load(f)
      glist.append(g)
  with open(result_pkl, 'rb') as f:
    results = pickle.load(f)
  for idx, g in enumerate(glist):
    x = results['best_samples'][idx]
    adj_mat = nx.to_numpy_array(g)
    cr, bl, message = evaluate(
        x, len(g), len(g.edges()) * 2, adj_mat, ng
    )
    print(message)
    print('cut ratio', cr)
    print('balanceness', bl)