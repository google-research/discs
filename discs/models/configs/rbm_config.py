"""Config file for rbms."""

from ml_collections import config_dict


def get_config():
  c = dict(
          dataset='mnist',
          num_categories=2,
          num_hidden=200,
          num_visible=784,
          name='rbm',
          train=False,
          model_path='./discs/storage/models/rbm/'
          )
  c['model_path'] = c['model_path']+ c['dataset']+'-'+str(c['num_categories'])+'-'+str(c['num_hidden'])+'/rbm.pkl'
  c['shape'] = (c['num_visible'], )
  if c['num_hidden'] == 200:
      c['save_dir_name'] = 'rbm_lowtemp'
  else:
      c['save_dir_name'] = 'rbm_hightemp'
  return config_dict.ConfigDict(c)
