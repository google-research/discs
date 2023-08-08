"""Config for er-test dataset."""

from discs.common import utils
from ml_collections import config_dict


def get_model_config(cfg_str):

  extra_cfg = utils.parse_cfg_str(cfg_str)
  model_config = dict(
      num_models=1,
      max_num_nodes=0,
      max_num_edges=0,
      num_instances=0,
      num_categories=3,
      shape=(0,),
      rand_type=extra_cfg['r'],
      penalty=0.0005,
      tolerance=0.03,
  )
  return config_dict.ConfigDict(model_config)
