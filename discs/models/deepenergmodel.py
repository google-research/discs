import os
import pickle
from discs.models import abstractmodel
import flax
import jax
import jax.numpy as jnp
import ml_collections
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
      path = config.data_path
      model = pickle.load(open(os.path.join(path, 'params.pkl'), 'rb'))
      config.params = flax.core.frozen_dict.freeze(model['params'])
      model_config = yaml.unsafe_load(
          open(os.path.join(path, 'config.yaml'), 'r')
      )
      config.update(model_config.model)
