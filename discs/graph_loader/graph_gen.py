"""Graph generator."""

import os
from discs.graph_loader import maxcut_loader


def get_graphs(config):
  """Get graph loader."""
  if config.model.graph_type.startswith('maxcut-'):
    data_folder = os.path.join(config.model.data_root, config.model.graph_type)
    return maxcut_loader.RandGraphGen(data_folder, config.model)
  else:
    raise ValueError('Unknown graph type %s' % config.model.graph_type)
