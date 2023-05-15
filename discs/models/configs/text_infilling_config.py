from ml_collections import config_dict


def get_config():
  model_config = dict(
      shape=(4,),
      num_categories=30522,
      name='text_infilling',
      bert_model='bert-base-uncased',
      data_root='./text_infilling_data/',
      random_init_sample=False,
  )
  model_config['save_dir_name'] = 'text_infilling_'+str(model_config['num_categories'])  
  return config_dict.ConfigDict(model_config)
