from collections.abc import Sequence

from absl import app
from discs.common import configs as common_configs
from discs.models.configs import bernoulli_config
from discs.evaluators import bernoulli_eval as b_eval

import importlib
import logging
import os
import pdb
from absl import app
from absl import flags
from discs.common import configs as common_configs
import discs.common.experiment_saver as saver_mod
import discs.common.utils as utils
from ml_collections import config_flags
import pdb
import pickle


flags.DEFINE_string(
    'gcs_results_path',
    './discs-maxcut-ba_sampler_sweep_56579701',
    'where results are being saved',
)
FLAGS = flags.FLAGS


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
    dict_o_keys['name'] = 'rmw'
  elif dict_o_keys['name'] == 'path_auxiliary':
    dict_o_keys['name'] = 'pafs'

  if 'solver' in dict_o_keys:
    if dict_o_keys['solver'] == 'euler_forward':
      dict_o_keys['name'] = str(dict_o_keys['name']) + 'f'
    del dict_o_keys['solver']
  if 'balancing_fn_type' in dict_o_keys:
    if 'name' in dict_o_keys:
      if dict_o_keys['balancing_fn_type'] == 'SQRT':
        dict_o_keys['name'] = str(dict_o_keys['name']) + '$\\sqrt{t}$'
      elif dict_o_keys['balancing_fn_type'] == 'RATIO':
        dict_o_keys['name'] = str(dict_o_keys['name']) + '$\\frac{t}{t+1}$'
      elif dict_o_keys['balancing_fn_type'] == 'MIN':
        dict_o_keys['name'] = str(dict_o_keys['name']) + '(min)'
      elif dict_o_keys['balancing_fn_type'] == 'MAX':
        dict_o_keys['name'] = str(dict_o_keys['name']) + '(max)'
      del dict_o_keys['balancing_fn_type']
  return dict_o_keys

def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  config = common_configs.get_config()
  config.model.update(bernoulli_config.get_config())

  ber_eval = b_eval.build_evaluator(config)
  # model
  model_mod = importlib.import_module('discs.models.bernoulli')
  model = model_mod.build_model(config)
  save_dir = f'./ee_plots/{FLAGS.gcs_results_path}/'
  if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
  folders = os.listdir(FLAGS.gcs_results_path)
  all_samples = []
  all_labels = []
  all_params = []
  for folder in folders:
    res_dic = get_experiment_config(folder)
    res_dic = process_keys(res_dic)
    label = res_dic['name']
    subfolderpath = os.path.join(FLAGS.gcs_results_path, folder)
    path_samples = os.path.join(subfolderpath, 'samples.pkl')
    path_params = os.path.join(subfolderpath, 'params.pkl')
    samples = pickle.load(open(path_samples, 'rb'))['trajectory']
    params = pickle.load(open(path_params, 'rb'))
    all_samples.append(samples)
    all_labels.append(label)
    all_params.append(params)
  ber_eval.plot_mixing_time_graph_over_chain(save_dir, model, all_params, all_samples, all_labels)
    
if __name__ == '__main__':
  app.run(main)
