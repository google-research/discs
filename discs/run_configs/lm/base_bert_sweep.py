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
                  'config.experiment.chain_length': [50],
                  'config.experiment.use_topk': [True],
                  'config.experiment.num_same_resample': [25],
                  'model_config.data_root': [
                      '/gcs/xcloud-shared/kgoshvadi/data/text_infilling_data/',
                  ],
                  'model_config.bert_model': [
                      '/gcs/xcloud-shared/kgoshvadi/data/text_infilling_models/bert-base-uncased/',
                  ],
                  'sampler_config.name': ['randomwalk'],
              },
              {
                  'config.experiment.chain_length': [50],
                  'config.experiment.use_topk': [True],
                  'config.experiment.num_same_resample': [25],
                  'model_config.data_root': [
                      '/gcs/xcloud-shared/kgoshvadi/data/text_infilling_data/',
                  ],
                  'model_config.bert_model': [
                      '/gcs/xcloud-shared/kgoshvadi/data/text_infilling_models/bert-base-uncased/',
                  ],
                  'sampler_config.adaptive': [False],
                  'sampler_config.name': ['dmala', 'dlmc'],
                  'sampler_config.balancing_fn_type': ['SQRT'],
                  'sampler_config.step_size': [0.1, 0.2, 0.3, 0.4, 0.5],
              },
              {
                  'config.experiment.chain_length': [50],
                  'config.experiment.use_topk': [True],
                  'config.experiment.num_same_resample': [25],
                  'model_config.data_root': [
                      '/gcs/xcloud-shared/kgoshvadi/data/text_infilling_data/',
                  ],
                  'model_config.bert_model': [
                      '/gcs/xcloud-shared/kgoshvadi/data/text_infilling_models/bert-base-uncased/',
                  ],
                  'sampler_config.adaptive': [True],
                  'sampler_config.name': ['dmala', 'dlmc'],
                  'sampler_config.balancing_fn_type': ['SQRT'],
              },
              {
                  'config.experiment.chain_length': [50],
                  'config.experiment.use_topk': [True],
                  'config.experiment.num_same_resample': [25],
                  'model_config.data_root': [
                      '/gcs/xcloud-shared/kgoshvadi/data/text_infilling_data/',
                  ],
                  'model_config.bert_model': [
                      '/gcs/xcloud-shared/kgoshvadi/data/text_infilling_models/bert-base-uncased/',
                  ],
                  'sampler_config.adaptive': [True, False],
                  'sampler_config.name': ['path_auxiliary', 'gwg'],
                  'sampler_config.balancing_fn_type': ['SQRT'],
              },
          ],
      )
  )
  return config