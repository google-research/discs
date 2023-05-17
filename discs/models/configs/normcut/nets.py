"""Config for er-test dataset."""

from discs.common import utils
from ml_collections import config_dict


def get_model_config(cfg_str):

  extra_cfg = utils.parse_cfg_str(cfg_str)
  model_config = dict(
      max_num_nodes=0,
      max_num_edges=0,
      num_instances=0,
      num_categories=3,
      shape=(0,),
      rand_type=extra_cfg['r'],
      penalty=1.0,
      tolerance=0.03,
      stype='span',
  )
  return config_dict.ConfigDict(model_config)
