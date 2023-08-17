"""Config file for text infilling model."""
from ml_collections import config_dict


def get_config():
  model_config = dict(
      shape=(4,),
      num_categories=30522,
      name='text_infilling',
      bert_model='./text_infilling_models/bert-base-uncased',
      data_root='./text_infilling_data/',
      random_init_sample=False,
      num_of_sentences=10,
      min_sentence_len=15,
      max_sentence_len=25,
  )
  model_config['save_dir_name'] = 'text_infilling'
  return config_dict.ConfigDict(model_config)
