"""TODO(kgoshvadi): DO NOT SUBMIT without one-line documentation for plot_ebm.

TODO(kgoshvadi): DO NOT SUBMIT without a detailed description of plot_ebm.
"""

import os

from absl import app
import matplotlib.pyplot as plt
from absl import flags
import pdb
import numpy as np

flags.DEFINE_string(
    'gcs_results_path',
    './ebm_mnist',
    'where results are being saved',
)
FLAGS = flags.FLAGS

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
    keys.append(str.split(key, '.')[-1])
    values.append(value)
  return dict(zip(keys, values))


def process_keys(dict_o_keys):
  if dict_o_keys['name'] == 'hammingball':
    dict_o_keys['name'] = 'hb-10-1'
  elif dict_o_keys['name'] == 'blockgibbs':
    dict_o_keys['name'] = 'bg-2'
  elif dict_o_keys['name'] == 'randomwalk':
    dict_o_keys['name'] = 'rmw'
  elif dict_o_keys['name'] == 'path_auxiliary':
    dict_o_keys['name'] = 'pas'

  if 'solver' in dict_o_keys:
    if dict_o_keys['solver'] == 'euler_forward':
      dict_o_keys['name'] = str(dict_o_keys['name']) + 'f'
    del dict_o_keys['solver']

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
        dict_o_keys['name'] = str(dict_o_keys['name'])
      elif dict_o_keys['balancing_fn_type'] == 'RATIO':
        dict_o_keys['name'] = str(dict_o_keys['name']) + '(r)'
      elif dict_o_keys['balancing_fn_type'] == 'MIN':
        dict_o_keys['name'] = str(dict_o_keys['name']) + '(min)'
      elif dict_o_keys['balancing_fn_type'] == 'MAX':
        dict_o_keys['name'] = str(dict_o_keys['name']) + '(max)'
      del dict_o_keys['balancing_fn_type']
  return dict_o_keys

def main(_):
  
  folders = os.listdir(FLAGS.gcs_results_path)
  folders = sorted(folders)
  image_paths = []
  y_ticks = []
  for folder in folders:
    subfolderpath = os.path.join(FLAGS.gcs_results_path, folder)
    res_dic = get_experiment_config(folder)
    res_dic = process_keys(res_dic)
    y_ticks.append(res_dic['name'])
    if res_dic['name'] in ['dlmc', 'hb-10-1']:
      chain = 1
    elif res_dic['name'] == 'bg-2':
      chain = 2
    elif res_dic['name'] in ['dlmcf', 'dmala']:
      chain = 3
    elif res_dic['name'] in ['gwg', 'pas']:
      chain = 0
    for i in range(10):
      image_file = os.path.join(subfolderpath, f'sample_{i}_of_chain_{chain}.jpeg')
      image_paths.append(image_file)

  # Calculate the total number of subplots required

  num_rows = 8
  num_cols = 10
  num_plots = num_rows*num_cols

  # Create the grid of subplots
  fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8), sharex=True, sharey=True)

  # Flatten the axes array to make it easier to iterate
  axes = axes.flatten()

  # Loop through the images and plot them in the corresponding subplot
  for i, image_path in enumerate(image_paths):
    # Read the image file
    image = plt.imread(image_path)

    # Plot the image in the current subplot
    axes[i].imshow(image)
    axes[i].axis('off')

    # Check if the current subplot is in the last row
    if i >= num_plots - num_cols:
      axes[i].axis('on')
      axes[i].set_xticks([])
      axes[i].set_xlabel(1+ i%num_cols, fontsize=14)
      
    # Check if the current subplot is in the last row
    if i % num_cols == 0:
      axes[i].axis('on')
      axes[i].set_yticks([])
      axes[i].set_ylabel(y_ticks[i//num_cols], fontsize=14)

  # Set x-label for the entire figure
  fig.text(0.55, -0.01, 'x1k Steps', ha='center', fontsize=16)
  
  # Set y-label for the entire figure
  fig.text(-0.02, 0.5, 'Samplers', va='center', rotation='vertical', fontsize=16)


  # # Adjust the spacing between subplots
  # fig.subplots_adjust(hspace=0.1, wspace=0.1) 
  
  
  # # Adjust the spacing between subplots
  fig.tight_layout()

  plt.show()

  plot_dir = f'./ebm_plots/{FLAGS.gcs_results_path}/'
  if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
  plt.savefig(
      f'{plot_dir}/res.png',
      bbox_inches='tight',
  )
  plt.savefig(
      f'{plot_dir}/res.pdf',
      bbox_inches='tight',
  )


if __name__ == '__main__':
  app.run(main)
