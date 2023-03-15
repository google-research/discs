import pdb
import csv
import numpy as np
import matplotlib.pyplot as plt
import os


def sort_dict(dictionary, x_labels):
    new_dict = {}

    for label in x_labels:
        if label == 'a_randomwalk':
            updated_label = 'rwm'
        elif label == 'a_gwg(sqrt)':
            updated_label = 'gwg(s)'
        elif label == 'a_gwg(ratio)':
            updated_label = 'gwg(r)'
        elif label == 'a_path_auxiliary(ratio)':
            updated_label = 'pafs(r)'
        elif label == 'a_path_auxiliary(sqrt)':
            updated_label = 'pafs(s)'
        elif label == 'a_dlmc(sqrt)':
            updated_label = 'dlmc(s)'
        elif label == 'a_dlmc(ratio)':
            updated_label = 'dlmc(r)'
        else:
            print("label not present")
            continue
        if label in dictionary:
            new_dict[updated_label] =  dictionary[label]
            del dictionary[label]

    for k, v in dictionary.items():
        new_dict[k] = v
    return new_dict



color_map = {}
color_map['rwm'] = 'green'
color_map['dlm'] = 'pink'
color_map['paf'] = 'saddlebrown'
color_map['gwg'] = 'red'

def main():

  x_labels = ['a_randomwalk','a_gwg(sqrt)', 'a_gwg(ratio)', 'a_path_auxiliary(sqrt)', 'a_path_auxiliary(ratio)', 'a_dlmc(sqrt)', 'a_dlmc(ratio)']
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
      dict_ee = sort_dict(dict_ee, x_labels)
      keys_ee = np.array(list(dict_ee.keys()))
      values_ee = np.array(list(dict_ee.values()))

      dict_clock = dict(zip(samplers, ess_clock))
      dict_clock = sort_dict(dict_clock, x_labels)
      keys_clock = np.array(list(dict_clock.keys()))
      values_clock = np.array(list(dict_clock.values()))

      x_pos = 0.5 * np.arange(len(keys_ee))
      c = []
      for key in keys_ee:
          print(key)
          alg = key[0:3]
          c.append(color_map[alg])
      fig = plt.figure(figsize=(10, 6))
      plt.yscale('log')
      plt.bar(x_pos, values_ee, width=0.2, color=c)
      plt.xticks(x_pos, keys_ee)

      plt.grid()
      plt.xlabel('Samplers')
      plt.ylabel('ESS')
      plt.title(f'ESS w.r.t. Energy Evaluations on {model}')
      plt.show()
      plt.savefig(curr_path + f'/EssEE_{model}.png', bbox_inches='tight')

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
      plt.ylabel('ESS')
      plt.title(f'ESS w.r.t. Wall Clock Time on {model}')
      plt.show()
      plt.savefig(curr_path + f'/EssClock_{model}.png', bbox_inches='tight')


main()
