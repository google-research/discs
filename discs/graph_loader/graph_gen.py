"""Graph generator."""

import os
from discs.graph_loader import maxcut_loader
from discs.graph_loader import mis_loader
import pdb

def get_graphs(config):
  """Get graph loader."""
  pdb.set_trace()
  if config.model.graph_type == 'optsicom':
    if config.model.rand_type == 'b':
      return maxcut_loader.OptsicomStatic(config.model.data_root)
    else:
      return maxcut_loader.OptsicomGen(config.model.rand_type)
  elif config.model.graph_type.startswith('ba'):
    return maxcut_loader.RandGraphGen(config.model.data_root, config.model)
  elif config.model.graph_type.startswith('er'):
    return maxcut_loader.RandGraphGen(config.model.data_root, config.model)
  elif config.model.graph_type.startswith('ertest'):
    return mis_loader.ErTestGraphGen(config.model.data_root, config.model)
  elif config.model.graph_type.startswith('satlib'):
    return mis_loader.SatLibGraphGen(config.model.data_root, config.model)

  else:
    raise ValueError('Unknown graph type %s' % config.model.graph_type)
