"""Graph generator."""

import os
from discs.graph_loader import maxcut_loader
from discs.graph_loader import mis_loader
import pdb

def get_graphs(config):
  """Get graph loader."""
  if config.model.name.startswith('maxcut'):
    return maxcut_loader.RandGraphGen(config.model.data_root, config.model)
  elif config.model.graph_type.startswith('ertest'):
    return mis_loader.ErTestGraphGen(config.model.data_root, config.model)
  elif config.model.graph_type.startswith('satlib'):
    return mis_loader.SatLibGraphGen(config.model.data_root, config.model)
  else:
    raise ValueError('Unknown graph type %s' % config.model.graph_type)
