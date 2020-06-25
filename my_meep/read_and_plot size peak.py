# %% reading in data
import sys
import os
import random
import math
import cmath
import numpy as np
import pickle
import traceback
import logging
from matplotlib import cm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from scipy import signal
b, a = signal.butter(2, 0.4, 'low', analog=False)

# parser = ArgumentParser(description='Process some integers.')
# parser.add_argument('-n', '--name', help='Give the name of the file to plot')
# args = parser.parse_args()
# f_name = args.name

# f_names = ['__hexagon_r_3.0_gap_3.0_xloc_0.0_fcen_0.8167_ff_0.5_particle_size_0.05_3.0_5distance_1.0_3.0_20',
# '__sphere_r_3.0_gap_3.0_xloc_0.0_fcen_0.8167_ff_0.5_particle_size_0.05_3.0_5distance_1.0_3.0_20',
# '__triangle_r_3.0_gap_3.0_xloc_0.0_fcen_0.8167_ff_0.5_particle_size_0.05_3.0_5distance_1.0_3.0_20',
# '__cube_r_3.0_gap_3.0_xloc_0.0_fcen_0.8167_ff_0.5_particle_size_0.05_3.0_5distance_1.0_3.0_20']

# datas = []
# for f_name in f_names:
#     dir = '/mnt/c/peter_abaqus/Summer-Research-Project/output/export/3/'
#     plot_f_name = dir + f_name + '.std'

#     with open(plot_f_name, 'rb') as f:
#         all_data = pickle.load(f)

#     param_name = all_data[0]
#     axis = [np.linspace(*ele) for ele in all_data[1]]
#     datas.append(all_data[2])

# # %%
# plt.plot(axis[0], data[:, :, 0])

# %% linear regression + peak location

from sklearn.linear_model import LinearRegression

reg =[]
x_pred = np.array([[0.5, 3]])

import matplotlib.cm as cm

colors = cm.rainbow(np.linspace(0, 1, 4))

for i, data in enumerate(datas):
    data_t = data[:, :, 0].transpose()
    peaks =[]
    for j in range(data_t.shape[1]):
        peak, _ = signal.find_peaks(data_t[:, j], height=0)
        if len(peak) >= 1:
            peaks.append(peak[0])
        else:
            peaks.append(0)

    # if peaks[-1] == 0:
    #     peaks = peaks[:3]
    
    # x = np.array([axis[0][1:]])
    # x = x.transpose()
    # reg.append(LinearRegression().fit(x[:len(peaks)-1], axis[1][peaks][1:]))
    # y_pred = reg[i].predict(x_pred.transpose())
    # plt.scatter(x[:len(peaks)-1], axis[1][peaks][1:], c=colors[i])
    # plt.plot(np.squeeze(x_pred), y_pred, c=colors[i])




shapes = ['hexagon', 'sphere', 'triangle', 'cube']
# for i, s in enumerate(shapes):
#     shapes[i] = s + ' y={:.2f}x + {:.2f}'.format(reg[i].coef_[0], reg[i].intercept_)

for i in range(data_t.shape[1]):
    plt.plot(axis[1], data_t[:, i])

for i in range(data_t.shape[1]):
    peak = peaks[i]
    plt.plot(axis[1][peak], data_t[:, i][peak], "x")

p_size = axis[0]
plt.title('Cube', fontsize=25)
plt.xlabel('Gap between particles', fontsize=20)
plt.ylabel('EM field mean strength', fontsize=20)
plt.legend(p_size, title = 'Particle size')

ax = plt.gca()
ax.tick_params(labelsize=18)

plt.savefig('/mnt/c/peter_abaqus/Summer-Research-Project/output/export/3' + '/hex_peak_size.png', dpi=400, bbox_inches='tight')

# %% plot fill factor vs temperature

# fill_factor = [0.57,
# 0.67,
# 0.77,
# 0.87]
# prod = [0.062695331,
# 0.07111506,
# 0.067728074,
# 0.069372269]

fill_factor =[0.63,
0.59,
0.58]

temp = np.array([95,
63,
50])-21

prod = [0.071169183,
0.054883161,
0.050084988]


color = 'tab:blue'
plt.plot(fill_factor, prod, "x", color=color)
plt.plot(fill_factor, prod, color=color)

ax1 = plt.gca()
plt.title('Absorption change', fontsize=25)
ax1.set_xlabel('fill factor', fontsize=20)
ax1.set_ylabel('EM absorption', fontsize=20, color=color)
# plt.legend('')
ax1.tick_params(labelsize=18)
ax1.tick_params(labelsize=18, axis='y', labelcolor=color)
ax1.set_ylim(bottom=0)

color = 'tab:red'
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('diff temp (°C)', fontsize=20, color=color)  # we already handled the x-label with ax1
ax2.plot(fill_factor, temp, color=color)
ax2.plot(fill_factor, temp, 'x', color=color)
ax2.tick_params(labelsize=18, axis='y', labelcolor=color)
ax2.set_ylim(bottom=0)

plt.savefig('/mnt/c/peter_abaqus/Summer-Research-Project/output/export/3' + '/ff_vs_em1.png', dpi=400, bbox_inches='tight')



# %% plot the normal distribution
mus = np.array([45, 100, 150])
sigmas = np.array([8.33, 8.5, 8.33])*2
sig_div_mean = sigmas/mus
print(sig_div_mean)

for mu, sigma in zip(mus, sigmas):
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.xlim((0, 200))
ranges = ['<75', '>75 <125', '>125']
line = [r.ljust(15, ' ')+ 'σ/x: {:0.2f}'.format(ele) for r, ele in zip(ranges, sig_div_mean)]
for l in line:
    print(len(l))
plt.legend(line, loc='upper right', fontsize=15)
sep = [75, 125]
for i in range(2):
    plt.plot([sep[i]]*2, [0, 0.05], '--b')

plt.title('particle size distribution', fontsize=25)
plt.xlabel('size', fontsize=20)
plt.ylabel('fraction', fontsize=20)
# plt.legend('')

ax = plt.gca()
ax.tick_params(labelsize=18)

plt.savefig('/mnt/c/peter_abaqus/Summer-Research-Project/output/export/3' + '/size_distri.png', dpi=400, bbox_inches='tight')

plt.show()


# %%
1

# %%
