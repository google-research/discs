color_map = {}
color_map['rwm'] = 'green'
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
  """Getting the name of the sampler based on its config.

  NEW_SAMPLER = In case of adding new sampler update this.
  """
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


def get_experiment_config(exp_config):
  exp_config = exp_config[1 + exp_config.find('_') :]
  keys = []
  values = []
  splits = str.split(exp_config, ',')
  for split in splits:
    key_value = str.split(split, '=')
    if len(key_value) == 2:
      key, value = key_value
      if value[0] == "'" and value[-1] == "'":
        value = value[1:-1]
      elif len(value) >= 2 and value[1] == '(':
        value = value[2:]
      if key != 'cfg_str':
        keys.append(str.split(key, '.')[-1])
        values.append(value)
  idx = exp_config.find('cfg_str')
  if idx != -1:
    keys.append('cfg_str')
    string = str.split(exp_config[len('cfg_str') + idx + 4 :], "'")[0]
    method = str.split(string, ',')[0]
    values.append(method)

  return dict(zip(keys, values))


def get_clusters_key_based(key, results_dict_list):
  """Clusters the experiments that are only different based on the key.

  Returns a list of list. Each list containing the cluster of experiment
  indeces.
  """
  results_index_cluster = []
  for i, result_dict in enumerate(results_dict_list):
    # the experiment doesn't have the key.
    if key not in result_dict:
      continue
    if len(results_index_cluster) == 0:
      results_index_cluster.append([i])
      continue

    found_match = False
    for j, cluster in enumerate(results_index_cluster):
      if get_diff_key(key, results_dict_list[cluster[0]], result_dict):
        found_match = True
        results_index_cluster[j].append(i)
        break
    if key in results_dict_list[i] and not found_match:
      results_index_cluster.append([i])

  return results_index_cluster


def process_ticks(x_ticks):
  x_ticks_new = []
  for i, tick in enumerate(x_ticks):
    if tick == "'SQRT'":
      x_ticks_new.append('$\\sqrt{t}$')
    elif tick == "'RATIO'":
      x_ticks_new.append('$\\frac{t}{t+1}$')
    elif tick == "'MIN'":
      x_ticks_new.append('1 \u2227 t')
    elif tick == "'MAX'":
      x_ticks_new.append('1 \u2228 t')
  return x_ticks_new


def sort_based_on_samplers(all_mapped_names):
  """NEW_SAMPLER = In case of adding new sampler, update this."""
  sampler_list = [
      'h',
      'b',
      'r',
      'gwg(s',
      'gwg(r',
      'gwg',
      'dmala-',
      'dmala(s',
      'dmala(r',
      'dmala',
      'pas-',
      'pas(s',
      'pas(r',
      'pas',
      'dlmcf-',
      'dlmcf(s',
      'dlmcf(r',
      'dlmcf',
      'dlmc-',
      'dlmc(s',
      'dlmc(r',
      'dlmc',
  ]
  for i, cluster_dict in enumerate(all_mapped_names):
    sampler_to_index = {}
    for key in cluster_dict.keys():
      if key in ['save_title', 'model']:
        continue
      for sampler_id, sampler in enumerate(sampler_list):
        if key.startswith(sampler):
          sampler_to_index[key] = sampler_id
          break
    sorted_sampler_to_index = {
        k: v
        for k, v in sorted(sampler_to_index.items(), key=lambda item: item[1])
    }
    sorted_keys_based_on_list = sorted_sampler_to_index.keys()
    sorted_res = {key: cluster_dict[key] for key in sorted_keys_based_on_list}
    sorted_res['save_title'] = cluster_dict['save_title']
    sorted_res['model'] = cluster_dict['model']
    all_mapped_names[i] = sorted_res

  return all_mapped_names


def get_diff_key(key_diff, dict1, dict2):
  """If dict1 and dict2 are only different in terms of key_diff."""
  for key in dict1.keys():
    if key in ['results', 'name']:
      continue
    if key not in dict2:
      return False
    if dict1[key] != dict2[key] and key != key_diff:
      return None
  return True
