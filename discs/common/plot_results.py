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


def plot_results(all_mapped_names, key_diff, xticks):
  for num, res_cluster in enumerate(all_mapped_names):
    plot_graph_cluster(num, res_cluster, key_diff, xticks)


def plot_graph_cluster(num, res_cluster, key_diff, xticks):
  key0 = list(res_cluster.keys())[0]
  result_keys = res_cluster[key0].keys()
  num_samplers = len(res_cluster.keys())
  for res_key in result_keys:
    f = plt.figure()
    f.set_figwidth(12)
    f.set_figheight(4)
    bar_width = 0.1
    for i, sampler in enumerate(res_cluster.keys()):
      if i == 0:
        x_poses = (
            2
            * num_samplers
            * bar_width
            * np.arange(len(res_cluster[sampler][res_key]))
        )
        local_pos = np.arange(num_samplers) - (num_samplers / 2)

      assert len(x_poses) == len(xticks)
      if FLAGS.evaluation_type == 'ess':
        plt.yscale('log')
        if res_key == 'ess_ee':
          plt.ylabel('ESS EE')
        else:
          plt.ylabel('ESS Clock')
      else:
        threshold = 0.00025
        value = value - 1.0 - threshold
        plt.ylim([0.98, 1.02])
      
      # pdb.set_trace()
      values = res_cluster[sampler][res_key]
      plt.bar(x_poses + local_pos[i] * bar_width, values, bar_width, label = sampler)
      
    plt.title(f'The effect of {key_diff}', fontsize=16)
    plt.xticks(x_poses, xticks)
    plt.legend(
      loc='lower right',
      fontsize=16,
      fancybox=True,
      framealpha=0.4,)
    plt.grid()
    plt.show()
    plt.savefig(
        FLAGS.gcs_results_path
        + f'/{res_key}_{key_diff}_based.png',
        bbox_inches='tight',
    )

  # result_keys = res_cluster[0].keys()
  # for key in result_keys:
  #   if key == 'indices':
  #     continue
  #   f = plt.figure()
  #   f.set_figwidth(12)
  #   f.set_figheight(4)
  #   bar_width = 0.15
  #   x_poses = 0.5 * np.arange(len(mapped_names))

  #   for name in mapped_names:

  # xticks = []
  # save_title_set = False
  # for i, index in enumerate(indeces):
  #   # computing graph name config-based
  #   if not save_title_set:
  #     graph_save = ''
  #     for key_dict, val_dict in dict_results[index].items():
  #       if key_dict not in [key_diff, 'results']:
  #         graph_save += str(key_dict) + '=' + str(val_dict) + ','
  #     save_title_set = True

  #   value = float(dict_results[index]['results'][key])
  #   xticks.append(str(dict_results[index][key_diff]))
  #   print('Valueeueeeeeee: ', float(value))
  #   if FLAGS.evaluation_type == 'ess':
  #     plt.yscale('log')
  #     plt.bar(x_poses[i], value, width=bar_width)
  #   else:
  #     threshold = 0.00025
  #     value = value - 1.0 - threshold
  #     plt.ylim([0.98, 1.02])
  #     plt.bar(x_poses[i], value, width=bar_width, bottom=1)

  # plt.xticks(x_poses, xticks)
  # plt.grid()
  # plt.show()
  # plt.savefig(
  #     FLAGS.gcs_results_path + f'/{key}_{key_diff}_based_{graph_save}.png', bbox_inches='tight'
  # )


def get_diff_key(key_diff, dict1, dict2):
  # pdb.set_trace()
  # if key_diff in dict1 and key_diff in dict2:
  #   if dict1[key_diff] == dict2[key_diff]:
  #     return False
  # else:
  #   return False
  for key in dict1.keys():
    if key in ['results', 'name']:
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
      if value[0] == "'" and value[-1] == "'":
        value = value[1:-1]
      elif len(value) >= 2 and value[1] == '(':
        value = value[2:]
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


def organize_experiments(results_index_cluster, experiments_results, key_diff):
  all_mapped_names = []
  for i, cluster in enumerate(results_index_cluster):
    name_mapped_index = {}
    for index in cluster:
      dict_o_keys = experiments_results[index]
      curr_name = dict_o_keys['name']
      if key_diff == 'balancing_fn_type':
        curr_name = curr_name[0 : curr_name.find('(')]
      if curr_name not in name_mapped_index:
        name_mapped_index[curr_name] = {}
        # name_mapped_index[curr_name]['indeces'] = []
        # name_mapped_index[curr_name][key_diff] = []
        for res_key in dict_o_keys['results']:
          name_mapped_index[curr_name][f'{res_key}'] = []
          
      # name_mapped_index[curr_name]['indeces'].append(index)
      # name_mapped_index[curr_name][key_diff].append(dict_o_keys[key_diff])
      for res_key in dict_o_keys['results']:
        name_mapped_index[curr_name][f'{res_key}'].append(
            dict_o_keys['results'][res_key]
        )
    all_mapped_names.append(name_mapped_index)
  
  return all_mapped_names

def sort_based_on_key(folders, key_diff):
  keydiff_vals = []
  for folder in folders:
    if folder[-3:] == 'png' or folder[-3:] == 'pdf':
      continue
    value_of_keydiff = folder[1+folder.find(key_diff)+len(key_diff):]
    if value_of_keydiff.find(',') != -1:
      value_of_keydiff = value_of_keydiff[0:value_of_keydiff.find(',')]
    if value_of_keydiff.find('(') != -1:
      value_of_keydiff = value_of_keydiff[1+value_of_keydiff.find('('):]
    
    if isinstance(value_of_keydiff, str):
      keydiff_vals.append(value_of_keydiff)
    else:
      keydiff_vals.append(float(value_of_keydiff))
  xticks = sorted(keydiff_vals)
  dict_to_sort = dict(zip(folders, keydiff_vals))
  sorted_dict = {k: v for k, v in sorted(dict_to_sort.items(), key=lambda item: item[1])}
  xticks = np.unique(xticks)
  print("xticks = ", xticks)
  return sorted_dict.keys(), xticks
  
  
def sort_based_on_samplers(all_mapped_names):
  
  sampler_list = ['h', 'b', 'r', 'g', 'dma', 'p', 'dlmcf', 'dlmc']
  for i, cluster_dict in enumerate(all_mapped_names):
    
    sampler_to_index = {}
    for key in cluster_dict.keys():
      for sampler_id, sampler in enumerate(sampler_list):
        if key.startswith(sampler):
          sampler_to_index[key] = sampler_id
          break
    sorted_sampler_to_index = {k: v for k, v in sorted(sampler_to_index.items(), key=lambda item: item[1])}
    sorted_keys_based_on_list = sorted_sampler_to_index.keys()
    print("***********************")
    print(sorted_keys_based_on_list)
    for key in sorted_keys_based_on_list:
      print(cluster_dict[key])
    print("***********************")
    sorted_res = {key: cluster_dict[key] for key in sorted_keys_based_on_list}
    print("%%%%%%%%")
    print(sorted_res)
    print("%%%%%%%%")
    all_mapped_names[i] = sorted_res

  return all_mapped_names

def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  key_diff = FLAGS.key
  if FLAGS.key == 'balancing_fn_type':
    FLAGS.key = 'name'

  experiments_results = []
  folders = os.listdir(FLAGS.gcs_results_path)
  folders = sorted(folders)
  folders, x_ticks = sort_based_on_key(folders, key_diff)
  for folder in folders:
    if folder[-3:] != 'png' and folder[-3:] != 'pdf':
      subfolderpath = os.path.join(FLAGS.gcs_results_path, folder)
      res_dic = get_experiment_config(folder)
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
  # print(FLAGS.key, results_index_cluster)
  all_mapped_names = organize_experiments(
      results_index_cluster, experiments_results, key_diff
  )
  for key in all_mapped_names[0].keys():
    print(key, " ", all_mapped_names[0][key])
  all_mapped_names = sort_based_on_samplers(all_mapped_names)
  print("xtickssssss: ", x_ticks)
  plot_results(all_mapped_names, FLAGS.key, x_ticks)


if __name__ == '__main__':
  app.run(main)
