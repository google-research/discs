from collections.abc import Sequence
import csv
import os
import pdb

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np

flags.DEFINE_string(
    'results_path',
    './Sampling_Experiment_56199929',
    'where results are being saved',
)
FLAGS = flags.FLAGS


def sort_dict(dictionary, x_labels):
  new_dict = {}

  for label in x_labels:
    if label == 'a_randomwalk':
      updated_label = 'rwm'
    elif label == 'blockgibbs':
      updated_label = 'bg-2'
    elif label == 'a_gwg(sqrt)':
      updated_label = 'gwg(s)'
    elif label == 'a_gwg(ratio)':
      updated_label = 'gwg(r)'
    elif label == 'a_path_auxiliary(ratio)':
      updated_label = 'pafs(r)'
    elif label == 'a_path_auxiliary(sqrt)':
      updated_label = 'pafs(s)'
    elif label == 'a_dlmc(sqrt)':
      updated_label = 'dlmc(s)'
    elif label == 'a_dlmc(ratio)':
      updated_label = 'dlmc(r)'
    elif label == 'a_dlmcf(sqrt)':
      updated_label = 'fdlmc(s)'
    elif label == 'a_dlmcf(ratio)':
      updated_label = 'fdlmc(r)'
    else:
      print('label not present: ', label)
      continue
    if label in dictionary:
      new_dict[updated_label] = dictionary[label]
      del dictionary[label]

  for k, v in dictionary.items():
    new_dict[k] = v
  return new_dict


def get_model_config(exp_config):
  exp_config = exp_config[1 + exp_config.find('_') :]
  model_val = ''
  splits = str.split(exp_config, ',')
  for split in splits:
    key_value = str.split(split, '=')
    if len(key_value) == 2:
      key, value = key_value
      if key.startswith('model'):
        model_val += str.split(key, '.')[-1] + str(value)
  res = {}
  res['model_config'] = model_val
  return res


def separate_model_based(experiment_results):
  res = {}
  for dic in experiment_results:
    if dic['model_config'] not in res:
      res[dic['model_config']] = {}
      res[dic['model_config']]['sampler'] = [dic['sampler']]
      res[dic['model_config']]['ess_ee'] = [dic['ess_ee']]
      res[dic['model_config']]['ess_clock'] = [dic['ess_clock']]
    else:
      res[dic['model_config']]['sampler'].append(dic['sampler'])
      res[dic['model_config']]['ess_ee'].append(dic['ess_ee'])
      res[dic['model_config']]['ess_clock'].append(dic['ess_clock'])
  return res


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  color_map = {}
  color_map['rwm'] = 'green'
  color_map['dlm'] = 'pink'
  color_map['fdl'] = 'gray'
  color_map['paf'] = 'saddlebrown'
  color_map['gwg'] = 'red'
  color_map['bg-'] = 'orange'
  

  x_labels = [
      'blockgibbs',
      'a_randomwalk',
      'a_gwg(sqrt)',
      'a_gwg(ratio)',
      'a_path_auxiliary(sqrt)',
      'a_path_auxiliary(ratio)',
      'a_dlmc(sqrt)',
      'a_dlmc(ratio)',
      'a_dlmcf(sqrt)',
      'a_dlmcf(ratio)',
  ]

  samplers = []
  ess_ee = []
  ess_clock = []
  experiments_results = []
  folders = os.listdir(FLAGS.results_path)
  for folder in folders:
    res_dic = get_model_config(folder)
    folderpath = os.path.join(FLAGS.results_path, folder)
    filename = os.path.join(folderpath, 'results.csv')

    filename = open(filename, 'r')
    # creating dictreader object
    file = csv.DictReader(filename)

    for col in file:
      model_name = str(col['model'])
      res_dic['sampler'] = col['sampler']
      res_dic['ess_ee'] = float(col['ESS_EE']) * 50000
      res_dic['ess_clock'] = float(col['ESS_T'])
    experiments_results.append(res_dic)


  results = separate_model_based(experiments_results)
  for key in results.keys():
    model = model_name +  key
    samplers = results[key]['sampler']
    ess_ee = results[key]['ess_ee']
    ess_clock = results[key]['ess_clock']
    dict_ee = dict(zip(samplers, ess_ee))
    dict_ee = sort_dict(dict_ee, x_labels)
    keys_ee = np.array(list(dict_ee.keys()))
    values_ee = np.array(list(dict_ee.values()))

    dict_clock = dict(zip(samplers, ess_clock))
    dict_clock = sort_dict(dict_clock, x_labels)
    keys_clock = np.array(list(dict_clock.keys()))
    values_clock = np.array(list(dict_clock.values()))

    x_pos = 0.5 * np.arange(len(keys_ee))
    c = []
    for key in keys_ee:
      print(key)
      alg = key[0:3]
      c.append(color_map[alg])
    fig = plt.figure(figsize=(10, 6))
    plt.yscale('log')
    plt.bar(x_pos, values_ee, width=0.2, color=c)
    plt.xticks(x_pos, keys_ee)

    plt.grid()
    plt.xlabel('Samplers')
    plt.ylabel('ESS')
    plt.title(f'ESS w.r.t. Energy Evaluations on {model}')
    plt.show()
    plt.savefig(FLAGS.results_path + f'/EssEE_{model}.png', bbox_inches='tight')

    #########
    c = []
    for key in keys_clock:
      splits = str.split(key, '_')
      if splits[0][0] != 'a':
        alg = splits[0][0:3]
      else:
        alg = splits[1][0:3]
      c.append(color_map[alg])

    fig = plt.figure(figsize=(10, 6))
    plt.yscale('log')
    plt.bar(x_pos, values_clock, width=0.2, color=c)
    plt.xticks(x_pos, keys_clock)

    plt.grid()
    plt.xlabel('Samplers')
    plt.ylabel('ESS')
    plt.title(f'ESS w.r.t. Wall Clock Time on {model}')
    plt.show()
    plt.savefig(
        FLAGS.results_path + f'/EssClock_{model}.png', bbox_inches='tight'
    )


if __name__ == '__main__':
  app.run(main)
