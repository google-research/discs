
color_map = {}
color_map['rwm'] = 'green'
color_map['fdl'] = 'gray'
color_map['pas'] = 'saddlebrown'
color_map['gwg'] = 'red'
color_map['bg-'] = 'orange'
color_map['dma'] = 'purple'
color_map['hb-'] = 'blue'


def get_color(sampler):
  if sampler[0:4] != 'dlmc':
    return color_map[sampler[0:3]]
  else:
    if sampler[0:5] == 'dlmcf':
      return 'gray'
  return 'pink'


def process_keys(dict_o_keys):
  if dict_o_keys['name'] == 'hammingball':
    dict_o_keys['name'] = 'hb-10-1'
  elif dict_o_keys['name'] == 'blockgibbs':
    dict_o_keys['name'] = 'bg-2'
  elif dict_o_keys['name'] == 'randomwalk':
    dict_o_keys['name'] = 'rwm'
  elif dict_o_keys['name'] == 'path_auxiliary':
    dict_o_keys['name'] = 'pas'

  if 'solver' in dict_o_keys:
    if dict_o_keys['solver'] == 'euler_forward':
      dict_o_keys['name'] = str(dict_o_keys['name']) + 'f'
    del dict_o_keys['solver']
    
  if 'approx_with_grad' in dict_o_keys:
    del dict_o_keys['approx_with_grad']

  if 'adaptive' in dict_o_keys:
    if dict_o_keys['adaptive'] == 'False':
      dict_o_keys['name'] = str(dict_o_keys['name']) + '-nA'
    del dict_o_keys['adaptive']
    if 'step_size' in dict_o_keys:
      dict_o_keys['name'] = str(dict_o_keys['name']) + dict_o_keys['step_size']
      del dict_o_keys['step_size']
    if 'n' in dict_o_keys:
      dict_o_keys['name'] = str(dict_o_keys['name']) + '-' + dict_o_keys['n']
      del dict_o_keys['n']
    if 'num_flips' in dict_o_keys:
      dict_o_keys['name'] = (
          str(dict_o_keys['name']) + '-' + dict_o_keys['num_flips']
      )
      del dict_o_keys['num_flips']

  if 'balancing_fn_type' in dict_o_keys:
    if 'name' in dict_o_keys:
      if dict_o_keys['balancing_fn_type'] == 'SQRT':
        dict_o_keys['name'] = str(dict_o_keys['name']) + '(s)'
      elif dict_o_keys['balancing_fn_type'] == 'RATIO':
        dict_o_keys['name'] = str(dict_o_keys['name']) + '(r)'
      elif dict_o_keys['balancing_fn_type'] == 'MIN':
        dict_o_keys['name'] = str(dict_o_keys['name']) + '(min)'
      elif dict_o_keys['balancing_fn_type'] == 'MAX':
        dict_o_keys['name'] = str(dict_o_keys['name']) + '(max)'
      del dict_o_keys['balancing_fn_type']
  return dict_o_keys
