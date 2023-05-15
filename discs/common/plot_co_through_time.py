import copy
import os
import pdb
import pickle

from absl import app
from absl import flags
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


flags.DEFINE_string(
    'gcs_results_path',
    './discs-mis-ertest_10k_sampler_sweep_56643607',
    'where results are being saved',
)
GRAPH_KEY = flags.DEFINE_string('graphkey', 'name', 'key to plot based on')
GRAPHTYPE = flags.DEFINE_string('graphtype', 'MIS', 'graph type')
GRAPHTITLE = flags.DEFINE_string('graphtitle', 'sampler', 'title of the graph')
GRAPHLABEL = flags.DEFINE_string('graphlabel', 'sampler', 'title of the graph')

FLAGS = flags.FLAGS

DEFAULT_SAMPLER = 'dlmc(s)'

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


def plot_results_keybased(key, results_dict_list):
  results_index_cluster = get_clusters_key_based(key, results_dict_list)
  if not results_index_cluster:
    results_index_cluster = [np.arange(len(results_dict_list))]
  print(results_index_cluster)
  for num, cluster in enumerate(results_index_cluster):
    plot_graph_cluster(num, key, results_dict_list, cluster)


def plot_graph_cluster(num, key, dict_results, indeces):
  f = plt.figure()
  f.set_figwidth(10)
  f.set_figheight(6)

  save_title_set = False
  for i, index in enumerate(indeces):
    # computing graph name config-based
    if not save_title_set or dict_results[index]['cfg_str'] == 'path_auxiliary':
      graph_save = ''
      for key_dict, val_dict in dict_results[index].items():
        if key_dict not in [key, 'results']:
          graph_save += str(key_dict) + '=' + str(val_dict) + ','
      save_title_set = True

    # computing graph label
    graph_label = GRAPHLABEL.value.replace('_', ' ')
    if key == 't_schedule':
      graph_label = '\u03C4 schedule: '

    # computing graph title
    graph_title = GRAPHTITLE.value.replace('_', ' ')
    if key != 'cfg_str':
      gragh_title = f'The effect of {graph_title}'
    else:
      gragh_title = f'Comparison of Gibbs and PAS on {GRAPHTYPE.value} '

    # plot x and y values
    key_value = dict_results[index][key]
    result = dict_results[index]
    traj_mean = result['results']['traj_mean']
    traj_var = result['results']['traj_var']
    x = 1 + (
        np.arange(len(traj_mean))
        * int(dict_results[index]['results']['log_every_steps'])
    )
    idx = 0
    x = x[idx:]
    traj_mean = traj_mean[idx:]
    traj_var = 0 * traj_var[idx:]

    # plotting
    if key != 'cfg_str':
      plt.plot(
          x, traj_mean, color=cm.tab20(2 * i), label=f'{key_value}', linewidth=2
      )
    else:
      if dict_results[index]['cfg_str'] == 'gibbs':
        plt.plot(
            x, traj_mean, color=cm.tab20(2 * i), label='Gibbs', linewidth=2
        )
      else:
        plt.plot(x, traj_mean, color=cm.tab20(2 * i), label='PAS', linewidth=2)
        plt.axhline(y=np.max(traj_mean), color='black', linestyle='--')
    if key != 'samples_per_instance':
      plt.fill_between(
          x,
          traj_mean - traj_var,
          traj_mean + traj_var,
          alpha=0.25,
          color=cm.tab20(2 * i),
      )
    
      plt.ylim(top=1.2)  # adjust the top leaving bottom unchanged
      plt.ylim(bottom=0)

  # sorting the labels
  dict_labels = plt.gca().get_legend_handles_labels()
  dict_labels = dict(zip(dict_labels[1], dict_labels[0]))
  if not key_value.isdigit():
    sorted_keys = sorted(dict_labels.keys())
  else:
    keys = dict_labels.keys()
    keys = [int(x) for x in keys]
    if key == 'chain_length':
      keys = keys * int(dict_results[index]['results']['log_every_steps'])
    keys = sorted(keys)
    sorted_keys = [str(x) for x in keys]
  dict_labels = {key_d: dict_labels[key_d] for key_d in sorted_keys}
  lines = [dict_labels[key_d] for key_d in dict_labels]
  labels = [f'{graph_label}={key_d}' for key_d in dict_labels]
  plt.legend(
      lines,
      labels,
      loc='lower right',
      fontsize=16,
      fancybox=True,
      framealpha=0.4,
  )

  # configing plot
  if key == 'cfg_str':
    plt.xscale('log')
  # plt.yscale('log')
  plt.grid()
  plt.title(gragh_title, fontsize=16)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.xlabel('Steps', fontsize=16)
  if GRAPHTYPE.value == 'MIS':
    plt.ylabel('Size of Independent Set', fontsize=16)
  if GRAPHTYPE.value == 'MAXCUT':
    plt.ylabel('Ratio \u03B1', fontsize=16)
  if GRAPHTYPE.value == 'TSP':
    plt.ylabel('Negative Tour Length', fontsize=16)
  ax = plt.gca()
  ax.set_axisbelow(True)
  for item in (
      [ax.xaxis.label, ax.yaxis.label]
      + ax.get_xticklabels()
      + ax.get_yticklabels()
  ):
    item.set_fontsize(16)
  plt.show()

  plot_dir = f'./plots/{FLAGS.gcs_results_path}/'
  if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
  print(gragh_title)
  plt.savefig(f'{plot_dir}/{graph_save}.pdf', bbox_inches='tight')
  plt.savefig(f'{plot_dir}/{graph_save}.png', bbox_inches='tight')


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


def get_experiment_config(exp_config):
  keys = []
  values = []
  splits = str.split(exp_config, ',')
  for split in splits:
    key_value = str.split(split, '=')
    if len(key_value) == 2:
      key, value = key_value
      if value[0] == "'":
        value = value[1:-1]
      if key != 'cfg_str':
        keys.append(str.split(key, '.')[-1])
        values.append(value)
  keys.append('cfg_str')
  idx = exp_config.find('cfg_str')
  if idx != -1:
    string = str.split(exp_config[len('cfg_str') + idx + 4 :], "'")[0]
    method = str.split(string, ',')[0]
    values.append(method)
  else:
    values.append(DEFAULT_SAMPLER)
    
  return dict(zip(keys, values))


def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  experiments_results = []
  folders = os.listdir(FLAGS.gcs_results_path)
  folders = sorted(folders)
  for subfolder in folders:
    subfolderpath = os.path.join(FLAGS.gcs_results_path, subfolder)
    # if subfolder[1] == '_' or subfolder[2] == '_':
    if True:
      results_path = os.path.join(subfolderpath, 'results.pkl')
      experiment_result = get_experiment_config(subfolder)
      print(experiment_result)
      print("*******")
      experiment_result = process_keys(experiment_result)
      print(experiment_result)
      print("######")
      experiment_result['results'] = {}
      if os.path.exists(results_path):
        results = pickle.load(open(results_path, 'rb'))
        experiment_result['results']['traj_mean'] = np.mean(
            results['trajectory'], axis=-1
        )
        experiment_result['results']['traj_var'] = np.sqrt(
            np.var(results['trajectory'], axis=-1)
        )
        if 'log_every_steps' not in experiment_result:
          experiment_result['results']['log_every_steps'] = '1000'
        else:
          experiment_result['results']['log_every_steps'] = '100'
          del experiment_result['log_every_steps']
        experiments_results.append(experiment_result)
  for key in [GRAPH_KEY.value]:
    plot_results_keybased(key, experiments_results)


if __name__ == '__main__':
  app.run(main)
