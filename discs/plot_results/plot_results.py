import csv
import os
import pdb
import pickle
import re
from absl import app
from absl import flags
import discs.plot_results.plot_utils as utils
import matplotlib.pyplot as plt
import numpy as np

flags.DEFINE_string(
    'gcs_results_path',
    './discs-maxcut-ba_sampler_sweep_56579701',
    'where to load the experiment results from',
)
flags.DEFINE_string('evaluation_type', 'co', 'depending on the task select from co/ess/lm')
flags.DEFINE_string('key', 'name', 'what key to plot against')

FLAGS = flags.FLAGS


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
    if res_key.endswith('std'):
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
          # NEW_SAMPLER: Add the position, alpha and bar width of your plot.
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
      if res_key + '_std' in res_cluster[sampler]:
        error = res_cluster[sampler][res_key + '_std']
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
          error.append(0)

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
            yerr=error,
            linewidth=10,
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
            yerr=error,
            linewidth=10,
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
      plt.title(f'Effect of {key_diff} on {model}', fontsize=18)
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
      else:
        plt.title(f'{model} model', fontsize=18)

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


def get_ticks_and_sorted_experiments(folders, key_diff):
  """Returns the plot ticks based on the provided key_diff and sorts the folders based on ticks.
     Ticks are the different values of the key_diff in the experiment.
  """
  keydiff_vals = []
  for folder in folders:
    if folder[-3:] == 'png' or folder[-3:] == 'pdf':
      continue
    value_of_keydiff = folder[1 + folder.find(key_diff) + len(key_diff) :]
    if value_of_keydiff.find(',') != -1:
      value_of_keydiff = value_of_keydiff[0 : value_of_keydiff.find(',')]
    if value_of_keydiff.find('(') != -1:
      value_of_keydiff = value_of_keydiff[1 + value_of_keydiff.find('(') :]
    try:
      keydiff_vals.append(int(float(value_of_keydiff)))
    except ValueError:
      keydiff_vals.append(str(value_of_keydiff))
  xticks = sorted(keydiff_vals)
  dict_to_sort = dict(zip(folders, keydiff_vals))
  sorted_dict = {
      k: v for k, v in sorted(dict_to_sort.items(), key=lambda item: item[1])
  }
  xticks = np.unique(xticks)
  print('xticks = ', xticks)
  return sorted_dict.keys(), xticks

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

def extract_floats(string_of_numbers):
  string_of_numbers = string_of_numbers[2:-2].strip()
  gap = string_of_numbers.find(' ')
  f1 = float(string_of_numbers[0:gap])
  f2 = float(string_of_numbers[gap + 1 :])
  return [f1, f2]


def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  key_diff = FLAGS.key
  """We process the sampler name and lbf is part of the name."""
  if FLAGS.key == 'balancing_fn_type':
    FLAGS.key = 'name'

  experiments_results = []
  folders = os.listdir(FLAGS.gcs_results_path)
  folders = sorted(folders)
  folders, x_ticks = get_ticks_and_sorted_experiments(folders, key_diff)
  model = str.split(FLAGS.gcs_results_path, '-')[1]
  for i, folder in enumerate(folders):
    subfolderpath = os.path.join(FLAGS.gcs_results_path, folder)
    res_dic = utils.get_experiment_config(folder)
    res_dic = utils.process_keys(res_dic)
    if 'save_samples' in res_dic:
      del res_dic['save_samples']
    print("Sub folder ID: ", i)
    print(res_dic)
    
    # loading the results.
    if FLAGS.evaluation_type == 'lm':
      filename = os.path.join(subfolderpath, 'results_topk.pkl')
      results = pickle.load(open(filename, 'rb'))
      del results['infill_sents']
    elif FLAGS.evaluation_type == 'co':
      filename = os.path.join(subfolderpath, 'results.pkl')
      results_pkl = pickle.load(open(filename, 'rb'))
      results = {}
      results['best_ratio_mean'] = np.mean(results_pkl['best_ratio'])
      results['running_time'] = 2 * float(results_pkl['running_time'])
      results['best_ratio_std'] = np.std(results_pkl['trajectory'][-1])
    else:
      filename = os.path.join(subfolderpath, 'results.csv')
      try:
        filename = open(filename, 'r')
      except:
        # The experiment has not been completed, we append 0.
        continue
      file = csv.DictReader(filename)
      results = {}
      for col in file:
        ESS = extract_floats(col['ESS_EE'])
        if 'chain_length' not in res_dic:
          results['ess_ee'] = ESS[0] * 50000
          results['ess_ee_std'] = ESS[1] * 50000
        else:
          results['ess_ee'] = ESS[0] * int(float(res_dic['chain_length']) // 2)
          results['ess_ee_std'] = float(ESS[1]) * int(
              float(res_dic['chain_length']) // 2
          )
        TIME = extract_floats(col['ESS_T'])
        print('time = ', col['ESS_T'])
        results['ess_clock'] = TIME[0]
        results['ess_clock_std'] = TIME[1]
    res_dic['results'] = results
    experiments_results.append(res_dic)
  results_index_cluster = utils.get_clusters_key_based(FLAGS.key, experiments_results)
  print(FLAGS.key, results_index_cluster)
  all_mapped_names = organize_experiments(
      results_index_cluster, experiments_results, key_diff, model
  )
  for key in all_mapped_names[0].keys():
    print(key, ' ', all_mapped_names[0][key])
  all_mapped_names = utils.sort_based_on_samplers(all_mapped_names)
  if FLAGS.key == 'name' and key_diff != 'balancing_fn_type':
    x_ticks = ['samplers']
  elif key_diff == 'balancing_fn_type':
    x_ticks = utils.process_ticks(x_ticks)

  print('xticks: ', x_ticks)
  if FLAGS.evaluation_type == 'ess':
    plot_results(all_mapped_names, key_diff, x_ticks)
  if FLAGS.evaluation_type == 'lm':
    save_result_as_csv(all_mapped_names, 'lm_csv')
  elif FLAGS.evaluation_type == 'co':
    save_result_as_csv(all_mapped_names, 'co_csv')


if __name__ == '__main__':
  app.run(main)
