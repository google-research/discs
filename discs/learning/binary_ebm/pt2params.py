"""Convert torch ckpt to jax."""

from typing import Sequence
import os
import torch

from flax.core.frozen_dict import unfreeze
from flax import traverse_util
from discs.models import resnet
import jax
import jax.numpy as jnp
from absl import app
from absl import flags
from ml_collections import config_flags
import numpy as np
import pickle


_CONFIG = config_flags.DEFINE_config_file('config')
flags.DEFINE_string('pt_path', '', 'path for pt ckpt')
flags.DEFINE_string('save_root', '', 'root folder for results')
FLAGS = flags.FLAGS


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  config = _CONFIG.value
  pt = torch.load(FLAGS.pt_path, map_location=torch.device('cpu'))
  th_model = pt['ema_model']
  model_key = jax.random.PRNGKey(1)
  model = resnet.build_model(config)
  params = model.make_init_params(model_key)
  params = traverse_util.flatten_dict(unfreeze(params))
  assert len(params.keys()) + 1 == len(pt['ema_model'].keys())

  data_mean = None
  for key, val in th_model.items():
    val = val.numpy()
    if key == 'mean':
      data_mean = val.tolist()
    else:
      assert key.startswith('net.')
      key = key[4:]
      if key.startswith('net.'):
        _, layer, mod, p = key.split('.')
        if mod == 'shortcut_conv':
          param_mod = 'shortcut_conv'
        else:
          param_mod = 'Conv_%d' % (int(mod[-1]) - 1)
        assert p == 'weight' or p == 'bias'
        param_key = ('BasicBlock_%s' % layer,
                     param_mod,
                     'kernel' if p == 'weight' else 'bias')
      else:
        mod, p = key.split('.')
        assert p == 'weight' or p == 'bias'
        if mod == 'proj':
          param_key = ('Conv_0', 'kernel' if p == 'weight' else 'bias')
        else:
          assert mod == 'energy_linear'
          param_key = ('Dense_0', 'kernel' if p == 'weight' else 'bias')
      assert param_key in params
      if param_key[-1] == 'kernel':
        if len(val.shape) == 4:
          val = np.transpose(val, (2, 3, 1, 0))
        else:
          assert len(val.shape) == 2
          val = jnp.transpose(val, (1, 0))
      p_val = params[param_key]
      assert p_val.shape == val.shape
      params[param_key] = jnp.array(val)
  results = {}
  results['params'] = traverse_util.unflatten_dict(params)
  results['params']['data_mean'] = data_mean
  with open(os.path.join(FLAGS.save_root, 'params.pkl'), 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
  with open(os.path.join(FLAGS.save_root, 'config.yaml'), 'w') as f:
    f.write(config.to_yaml())


if __name__ == '__main__':
  app.run(main)
