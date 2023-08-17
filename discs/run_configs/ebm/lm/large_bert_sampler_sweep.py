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
                      '/gcs/xcloud-shared/kgoshvadi/data/text_infilling_models/bert-large-uncased/',
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
                      '/gcs/xcloud-shared/kgoshvadi/data/text_infilling_models/bert-large-uncased/',
                  ],
                  'sampler_config.adaptive': [True, False],
                  'sampler_config.name': ['dmala'],
              },
             {
                  'config.experiment.chain_length': [50],
                  'config.experiment.use_topk': [True],
                  'config.experiment.num_same_resample': [25],
                  'model_config.data_root': [
                      '/gcs/xcloud-shared/kgoshvadi/data/text_infilling_data/',
                  ],
                  'model_config.bert_model': [
                      '/gcs/xcloud-shared/kgoshvadi/data/text_infilling_models/bert-large-uncased/',
                  ],
                  'sampler_config.adaptive': [True],
                  'sampler_config.name': ['path_auxiliary'],
              },
             {
                  'config.experiment.chain_length': [50],
                  'config.experiment.use_topk': [True],
                  'config.experiment.num_same_resample': [25],
                  'model_config.data_root': [
                      '/gcs/xcloud-shared/kgoshvadi/data/text_infilling_data/',
                  ],
                  'model_config.bert_model': [
                      '/gcs/xcloud-shared/kgoshvadi/data/text_infilling_models/bert-large-uncased/',
                  ],
                  'sampler_config.adaptive': [ False],
                  'sampler_config.name': ['path_auxiliary'],
                  'sampler_config.num_flips': [2],
              },
              {
                  'config.experiment.chain_length': [50],
                  'config.experiment.use_topk': [True],
                  'config.experiment.num_same_resample': [25],
                  'model_config.data_root': [
                      '/gcs/xcloud-shared/kgoshvadi/data/text_infilling_data/',
                  ],
                  'model_config.bert_model': [
                      '/gcs/xcloud-shared/kgoshvadi/data/text_infilling_models/bert-large-uncased/',
                  ],
                  'sampler_config.name': [
                      'dlmc', 'gwg'
                  ],
                  'sampler_config.adaptive': [True],
              },
              {
                  'config.experiment.chain_length': [50],
                  'config.experiment.use_topk': [True],
                  'config.experiment.num_same_resample': [25],
                  'model_config.data_root': [
                      '/gcs/xcloud-shared/kgoshvadi/data/text_infilling_data/',
                  ],
                  'model_config.bert_model': [
                      '/gcs/xcloud-shared/kgoshvadi/data/text_infilling_models/bert-large-uncased/',
                  ],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'sampler_config.adaptive': [False],
                  'sampler_config.solver': ['interpolate'],
                  'sampler_config.n':[500],
              },
              {
                  'config.experiment.chain_length': [50],
                  'config.experiment.use_topk': [True],
                  'config.experiment.num_same_resample': [25],
                  'model_config.data_root': [
                      '/gcs/xcloud-shared/kgoshvadi/data/text_infilling_data/',
                  ],
                  'model_config.bert_model': [
                      '/gcs/xcloud-shared/kgoshvadi/data/text_infilling_models/bert-large-uncased/',
                  ],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'sampler_config.adaptive': [False],
                  'sampler_config.solver': ['euler_forward'],
                  'sampler_config.n': [10000],
              },
          ],
      )
  )
  return config