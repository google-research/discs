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
    './discs-maxcut-ba_sampler_sweep_56579701',
    'where results are being saved',
)
flags.DEFINE_string('evaluation_type', 'co', 'where results are being saved')
flags.DEFINE_string('key', 'name', 'what key to plot against')
GRAPHTYPE = flags.DEFINE_string('graphtype', 'mis', 'graph type')

FLAGS = flags.FLAGS


def get_diff_key(key_diff, dict1, dict2):
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
    if not results_index_cluster:
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
    if res_key in ['running_time']:
      continue
    f = plt.figure()
    f.set_figwidth(12)
    f.set_figheight(8)
    bar_width = 0.5
    for i, sampler in enumerate(res_cluster.keys()):
      if sampler == 'save_title':
        save_title = res_cluster[sampler]
        continue
      elif sampler == 'model':
        model = res_cluster[sampler]
        continue
      if i == 0:
        x_poses = (
            num_samplers
            * bar_width
            * np.arange(len(res_cluster[sampler][res_key]))
        )
        if xticks[0] != 'samplers':
          local_pos = np.arange(num_samplers) - (num_samplers / 2) + 1.5
        else:
          alphas = [1, 1, 1, 1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5]
          bar_widths = [
              0.3,
              0.3,
              0.3,
              0.15,
              0.15,
              0.15,
              0.15,
              0.15,
              0.15,
              0.15,
              0.15,
              0.15,
              0.15,
          ]
          x_poses = [
              1.0,
              1.4,
              1.8,
              2.125,
              2.275,
              2.525,
              2.675,
              2.925,
              3.075,
              3.325,
              3.475,
              3.725,
              3.875,
          ]

      values = res_cluster[sampler][res_key]
      c = utils.get_color(sampler)

      if sampler[-3:] == '(r)':
        label_sampler = sampler[0:-3] + '$\\frac{t}{t+1}$'
      elif sampler[-3:] == '(s)':
        label_sampler = sampler[0:-3] + '$\\sqrt{t}$'
      else:
        label_sampler = sampler

      if xticks[0] != 'samplers':
        assert len(x_poses) == len(xticks)
        if len(values) != len(x_poses):
          print('Gonna be Appending 0')
        while len(values) < len(x_poses):
          values.append(0)

      if FLAGS.evaluation_type == 'ess':
        plt.yscale('log')
        if res_key == 'ess_ee':
          plt.ylabel('ESS w.r.t Energy Evaluation', fontsize=16)
        else:
          plt.ylabel('ESS w.r.t Clock', fontsize=16)

        if xticks[0] != 'samplers':
          alpha = 1
          plt.bar(
              x_poses + local_pos[i] * bar_width,
              values,
              bar_width,
              label=label_sampler,
              color=c,
              alpha=alpha,
          )
        else:
          len(x_poses)
          plt.bar(
              x_poses[i],
              values,
              bar_widths[i],
              label=label_sampler,
              color=c,
              alpha=alphas[i],
          )
      else:
        if FLAGS.evaluation_type != 'lm':
          threshold = 0.00025
          values = [float(values[0]) - 1.0 - threshold]
          plt.bar(
              x_poses + local_pos[i] * bar_width,
              values,
              bar_width,
              label=label_sampler,
              bottom=1,
              color=c,
          )
        else:
          values = [float(values[0]) * 100]
          plt.bar(
              x_poses + local_pos[i] * bar_width,
              values,
              bar_width,
              label=label_sampler,
              color=c,
          )

    if model not in ['fhmm', 'rbm']:
      model = model.capitalize()
    else:
      if model == 'fhmm':
        model = 'FHMM'
      else:
        model = 'RBM'

    if key_diff == 'shape':
      key_diff = 'sample dimension'
    elif key_diff == 'balancing_fn_type':
      key_diff = 'balancing function type'
    elif key_diff == 'num_categories':
      key_diff = 'number of categories'
    if key_diff != 'name':
      #plt.title(f'Effect of {key_diff} on {model}', fontsize=18)
      print("yum")
    else:
      if model == 'Bernoulli':
        splits = str.split(save_title, ',')
        for split in splits:
          if split.startswith('init_sigma'):
            sigma = str.split(split, '=')[1]
            break
        if sigma == '0.5':
          plt.title(f'High temperature {model}', fontsize=18)
        else:
          plt.title(f'Low temperature {model}', fontsize=18)
      elif model == 'Ising':
        splits = str.split(save_title, ',')
        for split in splits:
          if split.startswith('init_sigma'):
            sigma = str.split(split, '=')[1]
            break
        if sigma == '1.5':
          plt.title(f'High temperature {model}', fontsize=18)
        elif sigma == '3':
          plt.title(f'Low temperature {model}', fontsize=18)
        elif sigma == '4.5':
          plt.title(f'Very low temperature {model}', fontsize=18)
        elif sigma == '6':
          plt.title(f'Extremely low temperature {model}', fontsize=18)
      elif model == 'potts':
        splits = str.split(save_title, ',')
        sigma = 'dummy'
        for split in splits:
          if split.startswith('init_sigma'):
            sigma = str.split(split, '=')[1]
            break
        plt.title(f'Potts model with sigma {sigma}', fontsize=18)
      elif model == 'RBM':
        splits = str.split(save_title, ',')
        sigma = 'dummy'
        for split in splits:
          if split.startswith('num_categories'):
            num_categories = str.split(split, '=')[1]
            if num_categories != '2':
              plt.title(
                  f'Categorical RBM with {num_categories} categories',
                  fontsize=18,
              )
              break
          if split.startswith('data_path=RBM_DATA-mnist-2-'):
            num_hidden = split[len('data_path=RBM_DATA-mnist-2-') : -1]
            if num_hidden == '25':
              plt.title(
                  f'Binary RBM with hidden dimension {num_hidden}', fontsize=18
              )
            elif num_hidden == '200':
              plt.title(
                  f'Binary RBM with hidden dimension {num_hidden}', fontsize=18
              )
      elif model == 'FHMM':
        splits = str.split(save_title, ',')
        sigma = 'dummy'
        for split in splits:
          if split.startswith('shape'):
            shape = str.split(split, '=')[1]
            if shape != '200':
              plt.title(
                  f'Sharp Binary FHMM',
                  fontsize=16,
              )
              break
            elif shape != '1000':
              plt.title(
                  f'Smooth Binary FHMM',
                  fontsize=18,
              )
              break
          if split.startswith('num_categories'):
            num_categories = str.split(split, '=')[1]
            if num_categories != '2':
              plt.title(
                  f'Categorical FHMM with {num_categories} categories',
                  fontsize=18,
              )
              break
      # else:
      #   plt.title(f'{model} model', fontsize=18)

    if len(xticks) != 1:
      if model not in ['Potts', 'Ising']:
        plt.xticks(x_poses, xticks, fontsize=16)
      else:
        if key_diff == 'sample dimension':
          new_xticks = []
          for i, tick in enumerate(xticks):
            new_xticks.append(str(tick) + 'x' + str(tick))
          plt.xticks(x_poses, new_xticks, fontsize=16)
        else:
          plt.xticks(x_poses, xticks, fontsize=16)
    else:
      plt.xticks([], [], fontsize=16)

    if key_diff in ['name', 'balancing function type']:
      plt.legend(
          loc='upper left',
          fontsize=14,
          fancybox=True,
          framealpha=0.2,
      )
    else:
      plt.legend(
          loc='upper right',
          fontsize=14,
          fancybox=True,
          framealpha=0.2,
      )

    if GRAPHTYPE.value == 'mis':
      
      plt.ylabel('Size of Independent Set', fontsize=16)
      plt.xlabel("λ", fontsize=16)
      
      splits = str.split(save_title, ',')
      for split in splits:
        if split.startswith('cfg_str'):
          val = str.split(split, '=')[1][-4:]
          if val == '0.05':
            plt.ylim(95, 115)
            plt.axhline(y=98.59, color='black', linestyle='--')
          elif val == '0.10':
            plt.ylim(55, 70)
            plt.axhline(y=57.4, color='black', linestyle='--')
          elif val == '0.20':
            plt.ylim(30, 40)
            plt.axhline(y=31.56, color='black', linestyle='--')
          elif val == '0.25':
            plt.ylim(24, 33)
            plt.axhline(y=26.25, color='black', linestyle='--')
          elif val == '800':
            plt.axhline(y=44.87, color='black', linestyle='--')
            plt.ylim(85, 100)

      # else:
      #   plt.ylabel('Ratio \u03B1', fontsize=16)
    elif GRAPHTYPE.value == 'maxclique':
      plt.ylabel('Ratio \u03B1', fontsize=16)
      # plt.ylim(0.5, 1.1)

    plt.grid(axis='y')
    plt.show()

    plot_dir = f'./plots/{FLAGS.gcs_results_path}/'
    if not os.path.exists(plot_dir):
      os.makedirs(plot_dir)
    plt.savefig(
        f'{plot_dir}/{res_key}_{key_diff}_based_{save_title}.png',
        bbox_inches='tight',
    )
    plt.savefig(
        f'{plot_dir}/{res_key}_{key_diff}_based_{save_title}.pdf',
        bbox_inches='tight',
    )


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


def organize_experiments(
    results_index_cluster, experiments_results, key_diff, model
):
  all_mapped_names = []
  for i, cluster in enumerate(results_index_cluster):
    name_mapped_index = {}
    for i, index in enumerate(cluster):
      if i == 0:
        dict_o_keys = experiments_results[index]
        graph_save = ''
        for key_dict, val_dict in dict_o_keys.items():
          if key_dict != 'results':
            graph_save += str(key_dict) + '=' + str(val_dict) + ','
        name_mapped_index['save_title'] = graph_save
        name_mapped_index['model'] = model

      dict_o_keys = experiments_results[index]
      curr_name = dict_o_keys['name']
      if key_diff == 'balancing_fn_type':
        curr_name = curr_name[0 : curr_name.find('(')]
      if curr_name not in name_mapped_index:
        name_mapped_index[curr_name] = {}
        for res_key in dict_o_keys['results']:
          name_mapped_index[curr_name][f'{res_key}'] = []
      for res_key in dict_o_keys['results']:
        name_mapped_index[curr_name][f'{res_key}'].append(
            dict_o_keys['results'][res_key]
        )
    all_mapped_names.append(name_mapped_index)

  return all_mapped_names


def sort_based_on_key(folders, key_diff):
  keydiff_vals = []
  type_tick_str = False
  for folder in folders:
    if folder[-3:] == 'png' or folder[-3:] == 'pdf':
      continue
    value_of_keydiff = folder[1 + folder.find(key_diff) + len(key_diff) :]
    if value_of_keydiff.find(',') != -1:
      value_of_keydiff = value_of_keydiff[0 : value_of_keydiff.find(',')]
    if value_of_keydiff.find('(') != -1:
      value_of_keydiff = value_of_keydiff[1 + value_of_keydiff.find('(') :]
    try:
      keydiff_vals.append(float(value_of_keydiff))
    except ValueError:
      keydiff_vals.append(str(value_of_keydiff))
      type_tick_str = True
  xticks = sorted(keydiff_vals)
  dict_to_sort = dict(zip(folders, keydiff_vals))
  sorted_dict = {
      k: v for k, v in sorted(dict_to_sort.items(), key=lambda item: item[1])
  }
  xticks = np.unique(xticks)
  print('xticks = ', xticks)
  return sorted_dict.keys(), xticks


def sort_based_on_samplers(all_mapped_names):
  sampler_list = [
      'h',
      'b',
      'r',
      'gwg(s',
      'gwg(r',
      'gwg',
      'dmala-',
      'dmala(s',
      'dmala(r',
      'dmala',
      'pas-',
      'pas(s',
      'pas(r',
      'pas',
      'dlmcf-',
      'dlmcf(s',
      'dlmcf(r',
      'dlmcf',
      'dlmc-',
      'dlmc(s',
      'dlmc(r',
      'dlmc',
  ]
  for i, cluster_dict in enumerate(all_mapped_names):
    sampler_to_index = {}
    for key in cluster_dict.keys():
      if key in ['save_title', 'model']:
        continue
      for sampler_id, sampler in enumerate(sampler_list):
        if key.startswith(sampler):
          sampler_to_index[key] = sampler_id
          break
    sorted_sampler_to_index = {
        k: v
        for k, v in sorted(sampler_to_index.items(), key=lambda item: item[1])
    }
    sorted_keys_based_on_list = sorted_sampler_to_index.keys()
    sorted_res = {key: cluster_dict[key] for key in sorted_keys_based_on_list}
    sorted_res['save_title'] = cluster_dict['save_title']
    sorted_res['model'] = cluster_dict['model']

    all_mapped_names[i] = sorted_res

  return all_mapped_names


def save_result_as_csv(all_mapped_names, dir_name):
  csv_dir = os.path.join(dir_name, FLAGS.gcs_results_path[2:])
  if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)
  for res in all_mapped_names:
    csv_file = res['save_title']
    # csv_dir = f'{csv_dir}/{csv_file}.csv'
    csv_file = os.path.join(csv_dir, f'{csv_file}.csv')
    del res['save_title']
    key0 = list(res.keys())[0]
    csv_columns = list(res[key0].keys())
    csv_columns.insert(0, 'sampler')
    with open(csv_file, 'w') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
      writer.writeheader()
      for sampler in res.keys():
        if sampler == 'model':
          continue
        data = res[sampler]
        data['sampler'] = sampler
        writer.writerow(data)


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
  model = str.split(FLAGS.gcs_results_path, '-')[1]
  for folder in folders:
    subfolderpath = os.path.join(FLAGS.gcs_results_path, folder)
    res_dic = get_experiment_config(folder)
    res_dic = utils.process_keys(res_dic)
    if 'save_samples' in res_dic:
      del res_dic['save_samples']
    print(res_dic)
    print('******************')

    if FLAGS.evaluation_type == 'lm':
      filename = os.path.join(subfolderpath, 'results_topk.pkl')
      results = pickle.load(open(filename, 'rb'))
      del results['infill_sents']
    else:
      filename = os.path.join(subfolderpath, 'results.csv')
      try:
        filename = open(filename, 'r')
      except:
        continue
      file = csv.DictReader(filename)
      results = {}
      for col in file:
        if FLAGS.evaluation_type == 'ess':
          if 'chain_length' not in res_dic:
            results['ess_ee'] = float(col['ESS_EE']) * 50000
          else:
            results['ess_ee'] = float(col['ESS_EE']) * int(
                float(res_dic['chain_length']) // 2
            )
          results['ess_clock'] = float(col['ESS_T'])
        elif FLAGS.evaluation_type == 'co':
          results['best_ratio_mean'] = col['best_ratio_mean']
        if FLAGS.evaluation_type == 'co':
          results['running_time'] = 2 * float(col['running_time'])
    res_dic['results'] = results
    experiments_results.append(res_dic)
  results_index_cluster = get_clusters_key_based(FLAGS.key, experiments_results)
  print(FLAGS.key, results_index_cluster)
  all_mapped_names = organize_experiments(
      results_index_cluster, experiments_results, key_diff, model
  )
  for key in all_mapped_names[0].keys():
    print(key, ' ', all_mapped_names[0][key])
  all_mapped_names = sort_based_on_samplers(all_mapped_names)
  if FLAGS.key == 'name' and key_diff != 'balancing_fn_type':
    x_ticks = ['samplers']
  elif key_diff == 'balancing_fn_type':
    x_ticks_new = []
    for i, tick in enumerate(x_ticks):
      if tick == "'SQRT'":
        x_ticks_new.append('$\\sqrt{t}$')
      elif tick == "'RATIO'":
        x_ticks_new.append('$\\frac{t}{t+1}$')
      elif tick == "'MIN'":
        x_ticks_new.append('1 \u2227 t')
      elif tick == "'MAX'":
        x_ticks_new.append('1 \u2228 t')
    x_ticks = x_ticks_new

  print('xtickssssss: ', x_ticks)
  # if FLAGS.evaluation_type == 'ess':
  plot_results(all_mapped_names, key_diff, x_ticks)
  if FLAGS.evaluation_type == 'lm':
    save_result_as_csv(all_mapped_names, 'lm_csv')
  elif FLAGS.evaluation_type == 'co':
    save_result_as_csv(all_mapped_names, 'co_csv')


if __name__ == '__main__':
  app.run(main)
