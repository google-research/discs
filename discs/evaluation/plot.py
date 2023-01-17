import csv
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os

results_dir = './discs/'
all_dirs = os.listdir(results_dir)

pdb.set_trace()
for curr_dir in all_dirs:
    if curr_dir.startswith('results_'):
      curr_path = os.path.join(results_dir, curr_dir)
      # open the file in read mode
      filename = open(curr_path+'/results.csv', 'r')
      
      # creating dictreader object
      file = csv.DictReader(filename)

      samplers = []
      ess_ee = []
      ess_clock = []
      for col in file:
        model = col['model']
        samplers.append( col['sampler'])
        ess_ee.append( float(col['ESS_EE'])*50000 )
        ess_clock.append(float(col['ESS_T']) )

      ess_ee = np.sort(np.array(ess_ee))
      ess_clock = np.sort(np.array(ess_clock))
      x_pos = 0.3 * np.arange(len(samplers))
      c = ['green', 'red', 'red', 'saddlebrown', 'saddlebrown', 'pink', 'pink'][0:len(x_pos)]
      fig = plt.figure(figsize = (10, 6))
      plt.yscale('log')
      plt.bar(x_pos, ess_ee, width = 0.2, color=c)
      plt.xticks(x_pos, samplers)

      plt.xlabel("Samplers")
      plt.ylabel("ESS (high temp)")
      plt.title(f'ESS w.r.t. Energy Evaluations on {model}')
      plt.show()
      plt.savefig('EssEE.png')

      #########

      fig = plt.figure(figsize = (10, 6))
      plt.yscale('log')
      plt.bar(x_pos, ess_clock, width = 0.2, color=c)
      plt.xticks(x_pos, samplers)
      
      plt.xlabel("Samplers")
      plt.ylabel("ESS (high temp)")
      plt.title(f'ESS w.r.t. Wall Clock Time on {model}')
      plt.show()
      plt.savefig('EssClock.png')
