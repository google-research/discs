import csv
import os
import pdb
import pickle
from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np


flags.DEFINE_string(
    'gcs_results_path',
    './discs-maxcut-ba_sampler_sweep_56579701',
    'where results are being saved',
)
flags.DEFINE_string('evaluation_type', 'co', 'where results are being saved')
flags.DEFINE_string('key', 'name', 'what key to plot against')

FLAGS = flags.FLAGS


def get_diff_key(key_diff, dict1, dict2):
  if key_diff in dict1 and key_diff in dict2:
    if dict1[key_diff] == dict2[key_diff]:
      return False
  else:
    return False
  for key in dict1.keys():
    if key == 'results':
      continue
    if key not in dict2:
      return False
    if dict1[key] != dict2[key] and key != key_diff:
      return None
  return True


def get_clusters_key_based(key, results_dict_list):
  results_index_cluster = []
  for i, result_dict in enumerate(results_dict_list):
    if key not in result_dict:
      continue
    if len(results_index_cluster) == 0:
      results_index_cluster.append([i])
      continue

    found_match = False
    for j, cluster in enumerate(results_index_cluster):
      if get_diff_key(key, results_dict_list[cluster[0]], result_dict):
        found_match = True
        results_index_cluster[j].append(i)
        break
    if key in results_dict_list[i] and not found_match:
      results_index_cluster.append([i])
  return results_index_cluster


def plot_results(results_index_cluster, results_dict_list, key):
  print(results_index_cluster)
  for num, cluster in enumerate(results_index_cluster):
    plot_graph_cluster(num, results_dict_list, cluster, key)


def plot_graph_cluster(num, dict_results, indeces, key_diff):
  result_keys = dict_results[0]['results'].keys()
  for key in result_keys:    
    f = plt.figure()
    f.set_figwidth(12)
    f.set_figheight(4)
    bar_width = 0.15
    x_poses = 0.5 * np.arange(len(indeces))
    xticks = []
    save_title_set = False
    for i, index in enumerate(indeces):
      # computing graph name config-based
      if not save_title_set:
        graph_save = ''
        for key_dict, val_dict in dict_results[index].items():
          if key_dict not in [key_diff, 'results']:
            graph_save += str(key_dict) + '=' + str(val_dict) + ','
        save_title_set = True
      
      value = float(dict_results[index]['results'][key])
      xticks.append(str(dict_results[index][key_diff]))
      print('Valueeueeeeeee: ', float(value))
      if FLAGS.evaluation_type == 'ess':
        plt.yscale('log')
        plt.bar(x_poses[i], value, width=bar_width)
      else:
        threshold = 0.00025
        value = value - 1.0 - threshold
        plt.ylim([0.98, 1.02])
        plt.bar(x_poses[i], value, width=bar_width, bottom=1)

    plt.xticks(x_poses, xticks)
    plt.grid()
    plt.show()
    plt.savefig(
        FLAGS.gcs_results_path + f'/{key}_{key_diff}_based_{graph_save}.png', bbox_inches='tight'
    )


def get_diff_key(key_diff, dict1, dict2):
  if key_diff in dict1 and key_diff in dict2:
    if dict1[key_diff] == dict2[key_diff]:
      return False
  else:
    return False
  for key in dict1.keys():
    if key in ['results']:
      continue
    if key not in dict2:
      return False
    if dict1[key] != dict2[key] and key != key_diff:
      return None
  return True


def get_clusters_key_based(key, results_dict_list):
  results_index_cluster = []
  for i, result_dict in enumerate(results_dict_list):
    if key not in result_dict:
      continue
    if len(results_index_cluster) == 0:
      results_index_cluster.append([i])
      continue

    found_match = False
    for j, cluster in enumerate(results_index_cluster):
      if get_diff_key(key, results_dict_list[cluster[0]], result_dict):
        found_match = True
        results_index_cluster[j].append(i)
        break
    if key in results_dict_list[i] and not found_match:
      results_index_cluster.append([i])

  return results_index_cluster


def get_experiment_config(exp_config):
  exp_config = exp_config[1 + exp_config.find('_') :]
  keys = []
  values = []
  splits = str.split(exp_config, ',')
  for split in splits:
    key_value = str.split(split, '=')
    if len(key_value) == 2:
      key, value = key_value
      if value[0] == "'":
        value = value[1:-1]
    # if key != 'cfg_str':
    keys.append(str.split(key, '.')[-1])
    values.append(value)
  # keys.append('cfg_str')
  # idx = exp_config.find('cfg_str')
  # string = str.split(exp_config[len('cfg_str') + idx + 4 :], "'")[0]
  # method = str.split(string, ',')[0]
  # values.append(method)
  return dict(zip(keys, values))


def process_keys(dict_o_keys):
  if 'name' in dict_o_keys:
    if dict_o_keys['name'] == 'hammingball':
      dict_o_keys['name'] = 'hb-10-1'
    elif dict_o_keys['name'] == 'blockgibbs':
      dict_o_keys['name'] = 'bg-2'
    elif dict_o_keys['name'] == 'randomwalk':
      dict_o_keys['name'] = 'rmwl'
    elif dict_o_keys['name'] == 'path_auxiliary':
      dict_o_keys['name'] = 'pafs'

  if 'solver' in dict_o_keys:
    if dict_o_keys['solver'] == 'euler_forward':
      dict_o_keys['name'] = str(dict_o_keys['name']) + 'f'
    del dict_o_keys['solver']
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

  experiments_results = []
  folders = os.listdir(FLAGS.gcs_results_path)
  folders = sorted(folders)
  for folder in folders:
    if folder[1] == '_' or folder[2] == '_':
      subfolderpath = os.path.join(FLAGS.gcs_results_path, folder)
      res_dic = get_experiment_config(folder)
      # print(res_dic)
      # print('******************')
      res_dic = process_keys(res_dic)
      print(res_dic)
      print('******************')

      filename = os.path.join(subfolderpath, 'results.csv')
      filename = open(filename, 'r')
      # creating dictreader object
      file = csv.DictReader(filename)
      results = {}
      for col in file:
        if FLAGS.evaluation_type == 'ess':
          results['ess_ee'] = float(col['ESS_EE']) * 50000
          results['ess_clock'] = float(col['ESS_T'])
        else:
          results['best_ratio_mean'] = col['best_ratio_mean']
          # results['running_time'] = col['running_time']
      res_dic['results'] = results
      experiments_results.append(res_dic)
  results_index_cluster = get_clusters_key_based(FLAGS.key, experiments_results)
  print(FLAGS.key, results_index_cluster)
  plot_results(results_index_cluster, experiments_results, FLAGS.key)

  # experiments_results = []
  # subfolders = os.listdir(FLAGS.gcs_results_path)
  # for subfolder in subfolders:
  #   subfolderpath = os.path.join(FLAGS.gcs_results_path, subfolder)

  #   idx = subfolder.find('_') + 1
  #   experiment_result = get_experiment_config(subfolder)

  #   if os.path.isdir(subfolderpath) and subfolder[idx:].startswith(
  #       'experiment'
  #   ):
  #     subsubfolders = os.listdir(subfolderpath)
  #     for subsubfolder in subsubfolders:
  #       subsubfolder_path = os.path.join(subfolderpath, subsubfolder)
  #       if FLAGS.file_type == 'pkl':
  #         results_path = os.path.join(subsubfolder_path, 'results.pkl')
  #         if os.path.exists(results_path):
  #           results = pickle.load(open(results_path, 'rb'))
  #           experiment_result['traj_mean'] = np.mean(
  #               results['trajectory'], axis=0
  #           )
  #           experiment_result['traj_var'] = np.var(results['trajectory'], axis=0)
  #           experiments_results.append(experiment_result)
  #       else:
  #         results_path = os.path.join(subsubfolder_path, 'results.csv')
  #         filename = open(results_path, 'r')

  # plot_results(experiments_results)


if __name__ == '__main__':
  app.run(main)
