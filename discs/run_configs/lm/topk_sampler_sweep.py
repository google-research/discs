"""Config for rb job."""

from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='text_infilling',
          sampler='path_auxiliary',
          sweep=[
              {
                  'exp_config.chain_length': [50],
                  'exp_config.use_topk': [True],
                  'exp_config.num_same_resample': [25],
                  'model_config.data_root': [
                      '/gcs/xcloud-shared/kgoshvadi/data/text_infilling_data/',
                  ],
                  'model_config.bert_model': [
                      '/gcs/xcloud-shared/kgoshvadi/data/text_infilling_models/bert-base-uncased/',
                  ],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
              },
              {
                  'exp_config.chain_length': [50],
                  'exp_config.use_topk': [True],
                  'exp_config.num_same_resample': [25],
                  'model_config.data_root': [
                      '/gcs/xcloud-shared/kgoshvadi/data/text_infilling_data/',
                  ],
                  'model_config.bert_model': [
                      '/gcs/xcloud-shared/kgoshvadi/data/text_infilling_models/bert-base-uncased/',
                  ],
                  'sampler_config.name': ['path_auxiliary', 'gwg', 'dmala'],
                  'sampler_config.balancing_fn_type': ['SQRT'],
              },
              {
                  'exp_config.chain_length': [50],
                  'exp_config.use_topk': [True],
                  'exp_config.num_same_resample': [25],
                  'model_config.data_root': [
                      '/gcs/xcloud-shared/kgoshvadi/data/text_infilling_data/',
                  ],
                  'model_config.bert_model': [
                      '/gcs/xcloud-shared/kgoshvadi/data/text_infilling_models/bert-base-uncased/',
                  ],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'sampler_config.balancing_fn_type': ['SQRT'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
          ],
      )
  )
  return config
