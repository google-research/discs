import csv
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os


def sort_dict(x):
  return {
      k: v
      for k, v in sorted(x.items(), key=lambda item: item[1], reverse=False)
  }


color_map = {}
color_map['ran'] = 'green'
color_map['dlm'] = 'pink'
color_map['pat'] = 'saddlebrown'
color_map['gwg'] = 'red'

def main():
  results_dir = './discs/'
  all_dirs = os.listdir(results_dir)

  for curr_dir in all_dirs:
    if curr_dir.startswith('results_'):
      curr_path = os.path.join(results_dir, curr_dir)
      # open the file in read mode
      filename = open(curr_path + '/results.csv', 'r')

      # creating dictreader object
      file = csv.DictReader(filename)

      samplers = []
      ess_ee = []
      ess_clock = []
      for col in file:
        model = col['model']
        samplers.append(col['sampler'])
        ess_ee.append(float(col['ESS_EE']) * 50000)
        ess_clock.append(float(col['ESS_T']))

      dict_ee = dict(zip(samplers, ess_ee))
      dict_ee = sort_dict(dict_ee)
      keys_ee = np.array(list(dict_ee.keys()))
      values_ee = np.array(list(dict_ee.values()))

      dict_clock = dict(zip(samplers, ess_clock))
      dict_clock = sort_dict(dict_clock)
      keys_clock = np.array(list(dict_clock.keys()))
      values_clock = np.array(list(dict_clock.values()))

      x_pos = 0.5 * np.arange(len(keys_ee))
      c = []
      for key in keys_ee:
          splits = str.split(key, '_')
          if splits[0][0] != 'a':
            alg = splits[0][0:3]
          else:
            alg = splits[1][0:3]
          c.append(color_map[alg])
      fig = plt.figure(figsize=(10, 6))
      plt.yscale('log')
      plt.bar(x_pos, values_ee, width=0.2, color=c)
      plt.xticks(x_pos, keys_ee)

      plt.grid()
      plt.xlabel('Samplers')
      plt.ylabel('ESS (high temp)')
      plt.title(f'ESS w.r.t. Energy Evaluations on {model}')
      plt.show()
      plt.savefig(curr_path + f'/EssEE_{model}.png')

      #########
      c = []
      for key in keys_clock:
          splits = str.split(key, '_')
          if splits[0][0] != 'a':
            alg = splits[0][0:3]
          else:
            alg = splits[1][0:3]
          c.append(color_map[alg])

      fig = plt.figure(figsize=(10, 6))
      plt.yscale('log')
      plt.bar(x_pos, values_clock, width=0.2, color=c)
      plt.xticks(x_pos, keys_clock)

      plt.grid()
      plt.xlabel('Samplers')
      plt.ylabel('ESS (high temp)')
      plt.title(f'ESS w.r.t. Wall Clock Time on {model}')
      plt.show()
      plt.savefig(curr_path + f'/EssClock_{model}.png')


main()
