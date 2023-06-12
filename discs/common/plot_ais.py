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
    './discs-binray_ebm-logz_57945373',
    'where results are being saved',
)
flags.DEFINE_string('key', 'name', 'what key to plot against')


FLAGS = flags.FLAGS


color_map = {}
color_map['rmw'] = 'green'
color_map['fdl'] = 'gray'
color_map['pas'] = 'saddlebrown'
color_map['gwg'] = 'red'
color_map['bg-'] = 'orange'
color_map['dma'] = 'purple'
color_map['hb-'] = 'blue'
color_map['blo'] = 'orange'


def get_color(sampler):
  if sampler[0:4] != 'dlmc':
    return color_map[sampler[0:3]]
  else:
    if sampler[0:5] == 'dlmcf':
      return 'gray'
  return 'pink'


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


def plot_results(all_mapped_names):
  for num, res_cluster in enumerate(all_mapped_names):
    plot_graph_cluster(num, res_cluster)


def plot_graph_cluster(num, res_cluster):

  f = plt.figure()
  f.set_figwidth(12)
  f.set_figheight(8)
  save_title = 'yum'
  for i, sampler in enumerate(res_cluster.keys()):
    if sampler == 'save_title':
      save_title = res_cluster[sampler]
      continue
    elif sampler == 'model':
      model = res_cluster[sampler]
      continue
    if sampler[-3:] == '(r)':
      label_sampler = sampler[0:-3] + '$\\frac{t}{t+1}$'
    elif sampler[-3:] == '(s)':
      label_sampler = sampler[0:-3] + '$\\sqrt{t}$'
    else:
      label_sampler = sampler
    res = res_cluster[sampler]['logz'][0]
    x = np.arange(len(res))
    y = np.array(res)
    print(x)
    print(y)
    plt.plot(x, y, label=label_sampler)

  plt.legend(
      loc='upper right',
      fontsize=14,
      fancybox=True,
      framealpha=0.2,
  )
  # plt.yscale('log')
  plt.xlabel('Steps', fontsize= 16)
  plt.show()

  plot_dir = f'./plots_ais/{FLAGS.gcs_results_path}/'
  if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
  plt.savefig(
      f'{plot_dir}/{save_title}.png',
      bbox_inches='tight',
  )
  plt.savefig(
      f'{plot_dir}/{save_title}.pdf',
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


def process_keys(dict_o_keys):
  if dict_o_keys['name'] == 'hammingball':
    dict_o_keys['name'] = 'hb-10-1'
  elif dict_o_keys['name'] == 'blockgibbs':
    dict_o_keys['name'] = 'bg-2'
  elif dict_o_keys['name'] == 'randomwalk':
    dict_o_keys['name'] = 'rmw'
  elif dict_o_keys['name'] == 'path_auxiliary':
    dict_o_keys['name'] = 'pas'

  if 'solver' in dict_o_keys:
    if dict_o_keys['solver'] == 'euler_forward':
      dict_o_keys['name'] = str(dict_o_keys['name']) + 'f'
    del dict_o_keys['solver']
    
  if 'approx_with_grad' in dict_o_keys:
    del dict_o_keys['approx_with_grad']

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
      dict_o_keys['name'] = (
          str(dict_o_keys['name']) + '-' + dict_o_keys['num_flips']
      )
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
      keydiff_vals.append(int(float(value_of_keydiff)))
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
    res_dic = process_keys(res_dic)
    if 'save_samples' in res_dic:
      del res_dic['save_samples']
    print(res_dic)
    print('******************')
    filename = os.path.join(subfolderpath, 'logz.pkl')
    results = pickle.load(open(filename, 'rb'))
    res_dic['results'] = {}
    res_dic['results']['logz'] = np.array(results['logz'])
    experiments_results.append(res_dic)
  results_index_cluster = get_clusters_key_based(FLAGS.key, experiments_results)
  all_mapped_names = organize_experiments(
      results_index_cluster, experiments_results, key_diff, model
  )

  all_mapped_names = sort_based_on_samplers(all_mapped_names)
  plot_results(all_mapped_names)


if __name__ == '__main__':
  app.run(main)
