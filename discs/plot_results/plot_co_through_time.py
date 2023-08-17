import csv
import os
import pdb
import pickle
from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import plot_utils as utils

flags.DEFINE_string(
    'gcs_results_path',
    './discs-mis-ertest_10k_sampler_sweep_56643607',
    'where results are being saved',
)
GRAPH_KEY = flags.DEFINE_string('graphkey', 'name', 'key to plot based on')
GRAPHTYPE = flags.DEFINE_string('graphtype', 'mis', 'graph type')
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

  label_to_last_val = {}
  save_title_set = False
  for x_label in ['Steps', 'Time (s)']:
    for i, index in enumerate(indeces):
      # computing graph name config-based
      if not save_title_set:
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
        gragh_title = f'Comparison of samplers performance on {GRAPHTYPE.value}'
      else:
        gragh_title = f'Comparison of Gibbs and PAS on {GRAPHTYPE.value} '

      # plot x and y values
      key_value = dict_results[index][key]
      result = dict_results[index]
      traj_mean = result['results']['traj_mean']
      traj_std = result['results']['traj_std']
      x = 1 + (
          np.arange(len(traj_mean))
          * int(dict_results[index]['results']['log_every_steps'])
      )
      idx = 0
      traj_mean = traj_mean[idx:]
      traj_std = traj_std[idx:]

      if x_label != 'Steps':
        x = np.arange(0, 1, (1 / len(x))) * float(
            result['results']['running_time']
        )
        
      if GRAPHTYPE.value == 'maxcut':
        plt.xscale('log')

      # plt.yscale('log')
      plt.plot(
          x,
          traj_mean,
          color=utils.get_color(key_value),
          label=f'{key_value}',
          linewidth=2,
      )
      label_to_last_val[key_value] = traj_mean[-1]
      plt.fill_between(
          x,
          traj_mean - traj_std,
          traj_mean + traj_std,
          alpha=0.2,
          color=utils.get_color(key_value),
      )
    sorted_label_bo_value = {
        k: v
        for k, v in sorted(
            label_to_last_val.items(), key=lambda item: item[1], reverse=True
        )
    }
    # sorting the labels
    dict_labels = plt.gca().get_legend_handles_labels()
    dict_labels = dict(zip(dict_labels[1], dict_labels[0]))
    lines = [dict_labels[key_d] for key_d in sorted_label_bo_value.keys()]
    labels = [
        f'{graph_label}={key_d}' for key_d in sorted_label_bo_value.keys()
    ]
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
    # plt.title(gragh_title, fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel(x_label, fontsize=16)
    if GRAPHTYPE.value == 'mis':
      splits = str.split(graph_save, ',')
      for split in splits:
        if split.startswith('cfg_str'):
          val = str.split(split, '=')[1]
          if val == '0.05':
            plt.ylim(-100, 115)
            plt.axhline(y=98.59, color='black', linestyle='--')
          elif val == '0.10':
            plt.ylim(-100, 70)
            plt.axhline(y=57.4, color='black', linestyle='--')
          elif val == '0.20':
            plt.ylim(-100, 45)
            plt.axhline(y=31.56, color='black', linestyle='--')
          elif val == '0.25':
            plt.ylim(-100, 35)
            plt.axhline(y=26.25, color='black', linestyle='--')
          elif val == '800':
            plt.axhline(y=41.68, color='black', linestyle='--')
            plt.ylim(-500, 100)
          elif val == '10k':
            plt.axhline(y=381.31, color='black', linestyle='--')
            plt.ylim(-1000, 500)
          elif val == 'dlmc(s)':
            plt.axhline(y=425.96, color='black', linestyle='--')
            plt.ylim(300, 450)

          break
      plt.ylabel('Size of Independent Set', fontsize=16)
    elif GRAPHTYPE.value == 'maxclique':
      plt.ylabel('Ratio \u03B1', fontsize=16)

      if graph_save == 'cfg_str=dlmc(s),':
        plt.ylim(bottom=0, top=1.1)
      else:
        plt.axhline(y=0.789, color='black', linestyle='--')
        plt.ylim(bottom=0, top=1.1)

    if GRAPHTYPE.value == 'maxcut':
      splits = str.split(graph_save, ',')
      for split in splits:
        if split.startswith('cfg_str'):
          val = str.split(split, '=')[1]
          if val == 'b':
            plt.ylim(bottom=0.8)
          if val[0:2] == 'er':
            continue
          plt.legend(
              lines,
              labels,
              loc='upper left',
              fontsize=16,
              fancybox=True,
              framealpha=0.4,
          )
            

    if GRAPHTYPE.value == 'maxcut':
      plt.ylabel('Ratio \u03B1', fontsize=16)
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
    plt.savefig(f'{plot_dir}/{x_label}_{graph_save}.pdf', bbox_inches='tight')
    plt.savefig(f'{plot_dir}/{x_label}_{graph_save}.png', bbox_inches='tight')
    plt.clf()


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
    results_path = os.path.join(subfolderpath, 'results.pkl')
    experiment_result = get_experiment_config(subfolder)
    print(experiment_result)
    experiment_result = utils.process_keys(experiment_result)
    print(experiment_result)
    print('######')
    experiment_result['results'] = {}
    if os.path.exists(results_path):
      results = pickle.load(open(results_path, 'rb'))
      traj = np.array(results['trajectory'])
      experiment_result['results']['traj_mean'] = np.mean(
          traj, axis=-1
      )
      # pdb.set_trace()
      experiment_result['results']['traj_std'] = np.std(
          np.clip(traj, a_min=0, a_max=None) , axis=-1
      )

      if 'log_every_steps' not in experiment_result:
        experiment_result['results']['log_every_steps'] = '1000'
      else:
        experiment_result['results']['log_every_steps'] = str(
            experiment_result['log_every_steps']
        )
        del experiment_result['log_every_steps']
      csv_path = os.path.join(subfolderpath, 'results.csv')
      filename = open(csv_path, 'r')
      file = csv.DictReader(filename)
      for col in file:
        experiment_result['results']['best_ratio_mean'] = col['best_ratio_mean']
        experiment_result['results']['running_time'] = 2 * float(
            col['running_time']
        )
      experiments_results.append(experiment_result)

  for key in [GRAPH_KEY.value]:
    plot_results_keybased(key, experiments_results)


if __name__ == '__main__':
  app.run(main)
