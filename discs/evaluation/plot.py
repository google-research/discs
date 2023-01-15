import csv
import numpy as np
import matplotlib.pyplot as plt

# open the file in read mode
filename = open('./results.csv', 'r')
 
# creating dictreader object
file = csv.DictReader(filename)

samplers = []
ess_ee = []
ess_clock = []
for col in file:
  samplers.append( col['sampler'])
  ess_ee.append( col['ESS_EE'] )
  ess_clock.append(col['ESS_T'])

x_pos = 0.3 * np.arange(len(samplers))
c = np.arange(0, 1, 1/len(samplers))

fig = plt.figure(figsize = (10, 6))
plt.yscale('log')
plt.bar(x_pos, ess_ee, width = 0.2, color=c)
plt.xticks(x_pos, samplers)

plt.xlabel("Samplers")
plt.ylabel("ESS (high temp)")
plt.title("ESS w.r.t. Energy Evaluations on Bernoulli")
plt.show()
plt.savefig('EssEE.png')

#########

fig = plt.figure(figsize = (10, 6))
plt.yscale('log')
plt.bar(x_pos, ess_clock, width = 0.2, color=c)
plt.xticks(x_pos, samplers)
 
plt.xlabel("Samplers")
plt.ylabel("ESS (high temp)")
plt.title("ESS w.r.t. Wall Clock Time on Bernoulli")
plt.show()
plt.savefig('EssClock.png')

