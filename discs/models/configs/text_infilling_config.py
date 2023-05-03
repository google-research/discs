from ml_collections import config_dict


def get_config():
  model_config = dict(
      shape=(3,),
      num_categories=30522,
      name='text_infilling',
      bert_model='bert-base-uncased',
      sentence="I am not afraid of storms, for I am learning how to sail my ship.",
      infill_pos=[14, 15, 16],
  )
  model_config['save_dir_name'] = 'text_infilling_'+str(model_config['num_categories'])  
  return config_dict.ConfigDict(model_config)