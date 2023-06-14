"""Deep Energy Based Models."""
import os
import pickle
from discs.models import abstractmodel
import flax
import ml_collections
import yaml


class DeepEBM(abstractmodel.AbstractModel):
  """Deep EBM."""

  def __init__(self, config: ml_collections.ConfigDict):
    # loads the data from model config data_path
    if config.get('data_path', None):
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
      try:
        model = pickle.load(open(os.path.join(path, 'params.pkl'), 'rb'))
      except:
        import pickle5
        model = pickle5.load(open(os.path.join(path, 'params.pkl'), 'rb'))
      with config.unlocked():
        config.params = flax.core.frozen_dict.freeze(model['params'])
      model_config = yaml.unsafe_load(
          open(os.path.join(path, 'config.yaml'), 'r')
      )
      config.update(model_config.model)
    else:
      raise ValueError("The provided path doesn't exist")
