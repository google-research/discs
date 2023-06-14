"""Graph generator."""

import os
from discs.graph_loader import maxcut_loader
from discs.graph_loader import mis_loader
from discs.graph_loader import maxclique_loader
from discs.graph_loader import normcut_loader

def get_graphs(config):
  """Get graph loader."""
  if config.model.graph_type == 'optsicom':
    if config.model.rand_type == 'b':
      return maxcut_loader.OptsicomStatic(config.model.data_root)
    else:
      return maxcut_loader.OptsicomGen(config.model.rand_type)
  elif config.model.graph_type.startswith('ba'):
    return maxcut_loader.RandGraphGen(config.model.data_root, config.model)
  elif config.model.graph_type.startswith('ertest'):
    return mis_loader.ErTestGraphGen(config.model.data_root, config.model)
  elif config.model.graph_type.startswith('satlib'):
    return mis_loader.SatLibGraphGen(config.model.data_root, config.model)
  elif config.model.graph_type.startswith('er_density'):
    return mis_loader.ErDensityGraphGen(config.model.data_root, config.model)
  elif config.model.graph_type.startswith('er'):
    return maxcut_loader.RandGraphGen(config.model.data_root, config.model)
  elif config.model.graph_type == 'rb':
    return maxclique_loader.RBTestGraphGen(config.model.data_root, config.model)
  elif config.model.graph_type == 'twitter':
    return maxclique_loader.TwitterGraphs(config.model.data_root, config.model)
  elif config.model.graph_type == 'nets':
    return normcut_loader.ComputationGraphs(
        config.model.data_root, config.model)
  elif config.model.graph_type == 'gap_rand':
    return normcut_loader.RandGraphs(
        config.model.data_root, config.model)
  else:
    raise ValueError('Unknown graph type %s' % config.model.graph_type)
