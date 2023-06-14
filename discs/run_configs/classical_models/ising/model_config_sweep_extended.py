"""Config for rb job."""

from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='ising',
          sampler='path_auxiliary',
          sweep=[
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.mu': [0.25],
                  'model_config.lambdaa': [0.25],
                  'model_config.init_sigma': [0.75],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.mu': [0.30],
                  'model_config.lambdaa': [0.30],
                  'model_config.init_sigma': [0.90],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.mu': [0.35],
                  'model_config.lambdaa': [0.35],
                  'model_config.init_sigma': [1.05],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.mu': [0.40],
                  'model_config.lambdaa': [0.40],
                  'model_config.init_sigma': [1.20],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.mu': [0.45],
                  'model_config.lambdaa': [0.45],
                  'model_config.init_sigma': [1.35],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.mu': [0.5],
                  'model_config.lambdaa': [0.5],
                  'model_config.init_sigma': [1.5],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.mu': [0.55],
                  'model_config.lambdaa': [0.55],
                  'model_config.init_sigma': [1.65],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.mu': [0.60],
                  'model_config.lambdaa': [0.60],
                  'model_config.init_sigma': [1.80],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.mu': [0.65],
                  'model_config.lambdaa': [0.65],
                  'model_config.init_sigma': [1.95],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.mu': [0.70],
                  'model_config.lambdaa': [0.70],
                  'model_config.init_sigma': [2.10],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.mu': [0.75],
                  'model_config.lambdaa': [0.75],
                  'model_config.init_sigma': [2.25],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.mu': [0.80],
                  'model_config.lambdaa': [0.80],
                  'model_config.init_sigma': [2.40],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.mu': [0.85],
                  'model_config.lambdaa': [0.85],
                  'model_config.init_sigma': [2.55],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.mu': [0.90],
                  'model_config.lambdaa': [0.90],
                  'model_config.init_sigma': [2.70],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.mu': [0.95],
                  'model_config.lambdaa': [0.95],
                  'model_config.init_sigma': [2.85],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.mu': [1],
                  'model_config.lambdaa': [1],
                  'model_config.init_sigma': [3],
              },
              
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'model_config.mu': [0.25],
                  'model_config.lambdaa': [0.25],
                  'model_config.init_sigma': [0.75],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'model_config.mu': [0.30],
                  'model_config.lambdaa': [0.30],
                  'model_config.init_sigma': [0.90],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'model_config.mu': [0.35],
                  'model_config.lambdaa': [0.35],
                  'model_config.init_sigma': [1.05],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'model_config.mu': [0.40],
                  'model_config.lambdaa': [0.40],
                  'model_config.init_sigma': [1.20],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'model_config.mu': [0.45],
                  'model_config.lambdaa': [0.45],
                  'model_config.init_sigma': [1.35],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'model_config.mu': [0.5],
                  'model_config.lambdaa': [0.5],
                  'model_config.init_sigma': [1.5],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'model_config.mu': [0.55],
                  'model_config.lambdaa': [0.55],
                  'model_config.init_sigma': [1.65],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'model_config.mu': [0.60],
                  'model_config.lambdaa': [0.60],
                  'model_config.init_sigma': [1.80],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'model_config.mu': [0.65],
                  'model_config.lambdaa': [0.65],
                  'model_config.init_sigma': [1.95],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'model_config.mu': [0.70],
                  'model_config.lambdaa': [0.70],
                  'model_config.init_sigma': [2.10],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'model_config.mu': [0.75],
                  'model_config.lambdaa': [0.75],
                  'model_config.init_sigma': [2.25],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'model_config.mu': [0.80],
                  'model_config.lambdaa': [0.80],
                  'model_config.init_sigma': [2.40],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'model_config.mu': [0.85],
                  'model_config.lambdaa': [0.85],
                  'model_config.init_sigma': [2.55],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'model_config.mu': [0.90],
                  'model_config.lambdaa': [0.90],
                  'model_config.init_sigma': [2.70],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'model_config.mu': [0.95],
                  'model_config.lambdaa': [0.95],
                  'model_config.init_sigma': [2.85],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'model_config.mu': [1],
                  'model_config.lambdaa': [1],
                  'model_config.init_sigma': [3],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
              },
              
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.mu': [0.25],
                  'model_config.lambdaa': [0.25],
                  'model_config.init_sigma': [0.75],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.mu': [0.3],
                  'model_config.lambdaa': [0.3],
                  'model_config.init_sigma': [0.9],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.mu': [0.35],
                  'model_config.lambdaa': [0.35],
                  'model_config.init_sigma': [1.05],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.mu': [0.4],
                  'model_config.lambdaa': [0.4],
                  'model_config.init_sigma': [1.2],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.mu': [0.45],
                  'model_config.lambdaa': [0.45],
                  'model_config.init_sigma': [1.35],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.mu': [0.5],
                  'model_config.lambdaa': [0.5],
                  'model_config.init_sigma': [1.5],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.mu': [0.55],
                  'model_config.lambdaa': [0.55],
                  'model_config.init_sigma': [1.65],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.mu': [0.6],
                  'model_config.lambdaa': [0.6],
                  'model_config.init_sigma': [1.8],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.mu': [0.65],
                  'model_config.lambdaa': [0.65],
                  'model_config.init_sigma': [1.95],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.mu': [0.7],
                  'model_config.lambdaa': [0.7],
                  'model_config.init_sigma': [2.1],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.mu': [0.75],
                  'model_config.lambdaa': [0.75],
                  'model_config.init_sigma': [2.25],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.mu': [0.8],
                  'model_config.lambdaa': [0.8],
                  'model_config.init_sigma': [2.4],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.mu': [0.85],
                  'model_config.lambdaa': [0.85],
                  'model_config.init_sigma': [2.55],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.mu': [0.9],
                  'model_config.lambdaa': [0.9],
                  'model_config.init_sigma': [2.7],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.mu': [0.95],
                  'model_config.lambdaa': [0.95],
                  'model_config.init_sigma': [2.85],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.mu': [1],
                  'model_config.lambdaa': [1],
                  'model_config.init_sigma': [3],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
          ],
      )
  )
  return config
