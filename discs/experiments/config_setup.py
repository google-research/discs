"""Setting up the config values."""

import importlib
import logging
import pickle
import flax


from discs.common import utils
from absl import logging
from discs.common import configs as common_configs
from discs.graph_loader import graph_gen
import yaml
import jax.numpy as jnp

import pdb


def update_graph_cfg(config, graphs):
  config.model.max_num_nodes = graphs.max_num_nodes
  config.model.max_num_edges = graphs.max_num_edges
  config.model.shape = (graphs.max_num_nodes,)


def get_datagen(config):
  test_graphs = graph_gen.get_graphs(config)
  update_graph_cfg(config, test_graphs)
  logging.info(config)
  datagen = test_graphs.get_iterator('test', config.experiment.num_models)
  return datagen


def update_sampler_cfg(config, weight_fn_val):
  if 'balancing_fn_type' in config.sampler.keys():
    lbweighfn = importlib.import_module('discs.samplers.locallybalanced').LBWeightFn
    config.sampler.balancing_fn_type = getattr(lbweighfn, f'{weight_fn_val}')


def update_model_cfg(config):

  if config.model.get('data_path', None):
    path = config.model.data_path
    model = pickle.load(open(path + 'params.pkl', 'rb'))
    config.model.params = flax.core.frozen_dict.freeze(model['params'])
    model_config = yaml.unsafe_load(open(path + 'config.yaml', 'r'))
    config.model.update(model_config.model)

  elif config.model.get('cfg_str', None):
    datagen = get_datagen(config)
    data_list = next(datagen)
    sample_idx, params, reference_obj = zip(*data_list)
    params = utils.tree_stack(params)
    config.model.params = flax.core.frozen_dict.freeze(params)
    config.model.ref_obj = jnp.array(reference_obj)
    config.model.sample_idx = jnp.array(sample_idx)


def update_experiment_cfg(config):
  if config.model.get('cfg_str', None):
    config.experiment.evaluator = 'co_eval'
    co_exp_default_config = importlib.import_module(
        'discs.experiments.configs.co_experiment'
    )
    config.experiment.update(co_exp_default_config.get_co_default_config())
    graph_exp_config = importlib.import_module(
        'discs.experiments.configs.%s.%s'
        % (config.model.name, config.model.graph_type)
    )
    config.experiment.update(graph_exp_config.get_config())
  else:
    config.experiment.evaluator = 'ess_eval'
    config.experiment.log_every_steps = 1


def get_main_config(model_config, sampler_config, weight_fn):
  config = common_configs.get_config()
  config.sampler.update(sampler_config)
  update_sampler_cfg(config, weight_fn)
  config.model.update(model_config)
  logging.info(config)
  update_experiment_cfg(config)
  update_model_cfg(config)
  return config
