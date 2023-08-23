import csv
import os
import pdb
import pickle
from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np



# gsutil -m cp -r \
#   "gs://xcloud-shared/kgoshvadi/results/discs/discs-ising-lamdasweep_small_57936764" \
#   .

# gsutil -m cp -r \
#   "gs://xcloud-shared/kgoshvadi/results/discs/discs-ising-lamdasweep_bigger_57937100" \
#   .

flags.DEFINE_string(
    'gcs_results_path',
    './discs-ising-lamdasweep_small_62715141',
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
  plot_graph(all_mapped_names)


def plot_graph(all_mapped_names):
  for res_key in ['ess_ee', 'ess_clock']:
    f = plt.figure()
    f.set_figwidth(18)
    f.set_figheight(12)
    # plt.yscale('log')
    for sampler in all_mapped_names.keys():
      if sampler in ['hb-10-1', 'bg-2', 'rmw']:
        continue
      if sampler[-3:] == '(r)':
        alpha = 0.5
        line_style= '--'
      else:
        alpha = 1
        line_style= '-'
      if sampler[-3:] == '(r)':
        label_sampler = sampler[0:-3] + '$\\frac{t}{t+1}$'
      elif sampler[-3:] == '(s)':
        label_sampler = sampler[0:-3] + '$\\sqrt{t}$'
      else:
        label_sampler = sampler
      print(all_mapped_names[sampler]['x'])
      
      vals = np.array(all_mapped_names[sampler][res_key])
      std = np.array(all_mapped_names[sampler][res_key+'_std'])
      plt.plot(
          all_mapped_names[sampler]['x'],
          vals,
          label=label_sampler,
          c=get_color(sampler),
          alpha=alpha,
          linestyle=line_style, marker='o'
      )
      
      plt.fill_between(
          all_mapped_names[sampler]['x'],
          vals - std,
          vals + std,
          alpha=0.25*alpha,
          color=get_color(sampler),
      )

    plt.legend(
        loc='upper left',
        fontsize=20,
        fancybox=True,
        framealpha=0.2,
    )
    # pdb.set_trace()
    
    plt.ylabel('ESS w.r.t Energy Evaluation', fontsize=24)
    plt.xlabel('Inverse temperature $\\beta$', fontsize=24)
    xticks = all_mapped_names[sampler]['x']
    plt.xticks(xticks, fontsize=14)
    plt.yticks(fontsize=16)
    plt.grid(axis='y')
    plt.show()

    plot_dir = f'./plots_ising/{FLAGS.gcs_results_path}/'
    if not os.path.exists(plot_dir):
      os.makedirs(plot_dir)
    plt.savefig(
        f'{plot_dir}/{res_key}.png',
        bbox_inches='tight',
    )
    plt.savefig(
        f'{plot_dir}/{res_key}.pdf',
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


def map_temp(save_title):
  lamda = str.split(save_title, ',')[-2]
  lamda = str.split(lamda, '=')[-1]
  print("iadkjasdlsjdglsadjhgdjlfh = ", lamda)
  # pdb.set_trace()
  # print(float(lamda))
  # print((float(lamda) - (0.1607))/0.4)
  # res = round((float(lamda) - (0.1607))/0.04) + 1
  # print(res)
  return float(lamda)


def prepare_for_ising(all_mapped_names):
  res = {}
  for sampler in all_mapped_names[0].keys():
    if sampler in ['save_title', 'model']:
      continue
    res[sampler] = {}
    res[sampler]['ess_ee'] = [all_mapped_names[0][sampler]['ess_ee'][0]]
    res[sampler]['ess_ee_std'] = [all_mapped_names[0][sampler]['ess_ee_std'][0]] 
    res[sampler]['ess_clock'] = [all_mapped_names[0][sampler]['ess_clock'][0]]
    res[sampler]['ess_clock_std'] = [all_mapped_names[0][sampler]['ess_clock_std'][0]]
    res[sampler]['x'] = [0.1607]
    for temp in range(1, len(all_mapped_names)):
      x_val = map_temp(all_mapped_names[temp]['save_title'])
      res[sampler]['x'].append(x_val)
      res[sampler]['ess_ee'].append(
          all_mapped_names[temp][sampler]['ess_ee'][0]
      )
      res[sampler]['ess_clock'].append(
          all_mapped_names[temp][sampler]['ess_clock'][0]
      )
      res[sampler]['ess_ee_std'].append(
          all_mapped_names[temp][sampler]['ess_ee_std'][0]
      )
      res[sampler]['ess_clock_std'].append(
          all_mapped_names[temp][sampler]['ess_clock_std'][0]
      )

  return res


def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  key_diff = FLAGS.key
  if FLAGS.key == 'balancing_fn_type':
    FLAGS.key = 'name'

  experiments_results = []
  folders = os.listdir(FLAGS.gcs_results_path)
  folders = sorted(folders)
  folders, x_ticks = sort_based_on_key(folders, 'lambdaa')
  model = str.split(FLAGS.gcs_results_path, '-')[1]
  for folder in folders:
    subfolderpath = os.path.join(FLAGS.gcs_results_path, folder)
    res_dic = get_experiment_config(folder)
    res_dic = process_keys(res_dic)
    if 'save_samples' in res_dic:
      del res_dic['save_samples']
    # print(res_dic)
    # print('******************')

    filename = os.path.join(subfolderpath, 'results.csv')
    try:
      filename = open(filename, 'r')
    except:
      print(filename)
      results = {}
      results['ess_ee'] = 0
      results['ess_clock'] = 0
      res_dic['results'] = results
      experiments_results.append(res_dic)
      print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
      continue
    file = csv.DictReader(filename)
    results = {}        
    for col in file:
      gap = str.find(col['ESS_EE'], ' ')
      if 'chain_length' not in res_dic:
        results['ess_ee'] = float(col['ESS_EE'][2:gap]) * 50000
        results['ess_ee_std'] = float(col['ESS_EE'][gap+1:-2])/10 * 50000
      else:
        results['ess_ee'] = float(col['ESS_EE'][2:gap]) * int(
            float(res_dic['chain_length']) // 2
        )
        results['ess_ee_std'] = float(col['ESS_EE'][gap+1:-2])/10 * int(
            float(res_dic['chain_length']) // 2
        )      
      gap_t = str.find(col['ESS_T'], ' ')
      results['ess_clock'] = float(col['ESS_T'][2:gap_t])
      results['ess_clock_std'] = float(col['ESS_T'][gap_t+1:-2])

    res_dic['results'] = results
    experiments_results.append(res_dic)
  results_index_cluster = get_clusters_key_based(FLAGS.key, experiments_results)
  print(FLAGS.key, results_index_cluster)
  all_mapped_names = organize_experiments(
      results_index_cluster, experiments_results, key_diff, model
  )


  # for key in all_mapped_names[0].keys():
  #   print(key, ' ', all_mapped_names[0][key])
  all_mapped_names = sort_based_on_samplers(all_mapped_names)
  all_mapped_names = prepare_for_ising(all_mapped_names)
  

  # # if FLAGS.key == 'name' and key_diff != 'balancing_fn_type':
  #   x_ticks = ['samplers']
  # elif key_diff == 'balancing_fn_type':
  #   x_ticks_new = []
  #   for i, tick in enumerate(x_ticks):
  #     if tick == "'SQRT'":
  #       x_ticks_new.append('$\\sqrt{t}$')
  #     elif tick == "'RATIO'":
  #       x_ticks_new.append('$\\frac{t}{t+1}$')
  #     elif tick == "'MIN'":
  #       x_ticks_new.append('1 \u2227 t')
  #     elif tick == "'MAX'":
  #       x_ticks_new.append('1 \u2228 t')
  #   x_ticks = x_ticks_new


  plot_results(all_mapped_names)


if __name__ == '__main__':
  app.run(main)
