from discs.models import abstractmodel
import ml_collections
import jax
import jax.numpy as jnp
import os
import pickle
import flax
import yaml


class DeepEBM(abstractmodel.AbstractModel):
  """Deep EBM."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.load_params_configs(config)

  def make_init_params(self, rng):
    raise NotImplementedError

  def get_init_samples(self, rng, num_samples):
    raise NotImplementedError

  def forward(self, params, x):
    raise NotImplementedError

  def load_params_configs(self, config):
    path = os.path.join(config.data_path, 'params.pkl')
    if os.path.exists(path):
      model = pickle.load(open(path + 'params.pkl', 'rb'))
      config.params = flax.core.frozen_dict.freeze(model['params'])
      model_config = yaml.unsafe_load(open(path + 'config.yaml', 'r'))
      config.update(model_config.model)
