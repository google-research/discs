"""Evaluate using GAP code."""

import os
import sys
import networkx as nx
import numpy as np
import pickle
from absl import flags
import pdb
from absl import app
import csv


flags.DEFINE_string('graph_file', './sco/nets/VGG.pkl', 'dir to graph')
flags.DEFINE_string(
    'gcs_results_path',
    './discs/common/discs-normcut-vgg_57805067',
    'where results are being saved',
)
FLAGS = flags.FLAGS

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

def get_experiment_config(exp_config):
  exp_config = exp_config[1 + exp_config.find('_') :]
  keys = []
  values = []
  splits = str.split(exp_config, ',')
  for split in splits:
    key_value = str.split(split, '=')
    if len(key_value) == 2:
      key, value = key_value
      if value[0] == "'" and value[-1] == "'":
        value = value[1:-1]
      elif len(value) >= 2 and value[1] == '(':
        value = value[2:]
    keys.append(str.split(key, '.')[-1])
    values.append(value)
  return dict(zip(keys, values))


def process_keys(dict_o_keys):
  if dict_o_keys['name'] == 'hammingball':
    dict_o_keys['name'] = 'hb-10-1'
  elif dict_o_keys['name'] == 'blockgibbs':
    dict_o_keys['name'] = 'bg-2'
  elif dict_o_keys['name'] == 'randomwalk':
    dict_o_keys['name'] = 'rmw'
  elif dict_o_keys['name'] == 'path_auxiliary':
    dict_o_keys['name'] = 'pafs'

  if 'solver' in dict_o_keys:
    if dict_o_keys['solver'] == 'euler_forward':
      dict_o_keys['name'] = str(dict_o_keys['name']) + 'f'
    del dict_o_keys['solver']

  if 'adaptive' in dict_o_keys:
    if dict_o_keys['adaptive'] == 'False':
      dict_o_keys['name'] = str(dict_o_keys['name']) + '-nA'
    del dict_o_keys['adaptive']
    if 'step_size' in dict_o_keys:
      dict_o_keys['name'] = str(dict_o_keys['name']) + dict_o_keys['step_size']
      del dict_o_keys['step_size']
    if 'n' in dict_o_keys:
      dict_o_keys['name'] = str(dict_o_keys['name']) + '-' + dict_o_keys['n']
      del dict_o_keys['n']
    if 'num_flips' in dict_o_keys:
      dict_o_keys['name'] = str(dict_o_keys['name']) + '-' + dict_o_keys['num_flips']
      del dict_o_keys['num_flips']

  if 'balancing_fn_type' in dict_o_keys:
    if 'name' in dict_o_keys:
      if dict_o_keys['balancing_fn_type'] == 'SQRT':
        dict_o_keys['name'] = str(dict_o_keys['name']) + '(s)'
      elif dict_o_keys['balancing_fn_type'] == 'RATIO':
        dict_o_keys['name'] = str(dict_o_keys['name']) + '(r)'
      elif dict_o_keys['balancing_fn_type'] == 'MIN':
        dict_o_keys['name'] = str(dict_o_keys['name']) + '(min)'
      elif dict_o_keys['balancing_fn_type'] == 'MAX':
        dict_o_keys['name'] = str(dict_o_keys['name']) + '(max)'
      del dict_o_keys['balancing_fn_type']
  return dict_o_keys


def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  ng = 3
  graph_file = FLAGS.graph_file
  with open(graph_file, 'rb') as f:
    g = pickle.load(f)

  folders = os.listdir(FLAGS.gcs_results_path)
  folders = sorted(folders)
  res = []
  res.append(['sampler', 'cut ratio', 'balanceness'])
  for folder in folders:
    subfolderpath = os.path.join(FLAGS.gcs_results_path, folder)
    res_dic = get_experiment_config(folder)
    res_dic = process_keys(res_dic)
    filename = os.path.join(subfolderpath, 'results.pkl')
    new_res = []
    with open(filename, 'rb') as f:
      results = pickle.load(f)
      x = results['best_samples'][0]
      x = np.array(x).astype(int)
      adj_mat = nx.to_numpy_array(g)
      cr, bl, message = evaluate(x, len(g), len(g.edges()) * 2, adj_mat, ng)
      print(message)
      print(res_dic['name'])
      print('cut ratio', cr)
      print('balanceness', bl)
      new_res.append(res_dic['name'])
      new_res.append(cr)
      new_res.append(bl)
    res.append(new_res)
  
  f_name = str.split(FLAGS.gcs_results_path,'/')[-1]
  file_path = os.path.join('./discs/common/normcut_csv/', f_name)
  if not os.path.exists(file_path):
    os.makedirs(file_path)
  with open(file_path+'/res.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(res)
      

if __name__ == '__main__':
  app.run(main)
