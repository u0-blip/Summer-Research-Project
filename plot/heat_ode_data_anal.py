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
from matplotlib import colors

# %% plot fill factor vs temperature
import numpy as np
import matplotlib.pyplot as plt
# fill_factor = [0.57,
# 0.67,
# 0.77,
# 0.87]
# prod = [0.062695331,
# 0.07111506,
# 0.067728074,
# 0.069372269]

fill_factor = np.array([0.7279, 0.6876, 0.6844])

loss_factor = []
eps = []
for ff in fill_factor:
    eps_olivate = complex(7.27, 0.0685)
    eps_A = complex(1, 0)
    A = 2
    B = (1-3*ff)*eps_olivate-(2-3*ff)*eps_A
    C = -eps_A*eps_olivate
    res = (-B+np.sqrt(B**2-4*A*C))/(2*A)
    eps.append(res.real)
    loss_factor.append(res.imag)
    
eps, loss_factor = np.array(eps), np.array(loss_factor)

temp_olivine = np.array([[
    [21, 38, 57, 62, 73],
    [21, 29, 34, 40, 45],
    [21, 25, 33, 40, 44],
],[
    [21, 56, 76, 87, 95],
    [21, 42, 46, 54, 63], 
    [21, 36, 43, 46, 56],
]])

temp_diopside = np.array([

])


temp = np.array([95, 63, 56])-21
temp_diff = np.diff(temp)
E_field_squared = np.array([0.46354717, 0.44522147, 0.44359918])

E_field_pre_factor = E_field_squared*eps/(fill_factor)*loss_factor

const = np.mean(temp/E_field_pre_factor)

prod_temp = np.array([const*ele for ele in E_field_pre_factor])
prod_temp_per_min = (prod_temp)/4
prod_temp_over_time = []
for i in range(5):
    prod_temp_over_time.append(prod_temp_per_min*(i)+21)
prod_temp_over_time = np.array(prod_temp_over_time)

# color1 = 'tab:blue'
# color2 = 'tab:red'
# plt.plot(fill_factor, prod_temp, color = color1)
# ax1 = plt.gca()
# ax1.plot(fill_factor, temp, color=color2)
# plt.plot(fill_factor, prod_temp, "x", color = color1)
# ax1.plot(fill_factor, temp, "x", color=color2)
# plt.title('Heating rate difference', fontsize=20)
# ax1.set_xlabel('Fill factor', fontsize=20)
# ax1.set_ylabel('Temp change (°C)', fontsize=20)
# ax1.legend(('Simulation result', 'Experimental result'))
# ax1.tick_params(labelsize=18)
# ax1.tick_params(labelsize=18, axis='y')
# ax1.set_ylim(bottom=0, top=90)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# ax2.set_ylabel('Temp diff (°C)', fontsize=20, color=color)  # we already handled the x-label with ax1
# ax2.plot(fill_factor, temp, color=color)
# ax2.plot(fill_factor, temp, 'x', color=color)
# ax2.tick_params(labelsize=18, axis='y', labelcolor=color)
# ax2.set_ylim(bottom=0, top=120)

# plt.savefig('/mnt/c/peter_abaqus/Summer-Research-Project/output/export/3' + '/ff_vs_em2.png', dpi=400, bbox_inches='tight')
cmap = ['y','g','r']
for i in range(3):
    plt.plot(prod_temp_over_time.transpose()[i], color=cmap[i])


for i in range(3):
    plt.plot(temp_olivine[1][i], '*', color=cmap[i])


# plt.plot(prod_temp_over_time, '*')
# plt.plot(temp_olivine[1].transpose(), '+')
plt.xlabel('time (minutes)')
plt.ylabel('Temperature')
plt.title('Experimental temperature')
plt.legend(['<75um', '75-125um', '>125um'])
plt.ylim([15, 100])
plt.savefig('/mnt/c/peter_abaqus/Summer-Research-Project/output/export/3' + '/exp_linear_values_fit_const.png', dpi=400, bbox_inches='tight')

# %%
cmap = ['y','g','r']
for i in range(3):
    plt.plot(temp_olivine[1][i], color=cmap[i])


for i in range(3):
    plt.plot(temp_olivine[1][i], '+', color=cmap[i])

# for i in range(3):
#     plt.plot(temp_olivine[1][i], color=cmap[i])

# plt.plot(prod_temp_over_time, '*')
# plt.plot(temp_olivine[1].transpose(), '+')
plt.xlabel('time (minutes)')
plt.ylabel('Temperature')
plt.title('Simulated temperature')
plt.legend(['<75um', '75-125um', '>125um'])
plt.ylim([15, 100])
plt.savefig('/mnt/c/peter_abaqus/Summer-Research-Project/output/export/3' + '/exp_values.png', dpi=400, bbox_inches='tight')


# %% plot fill factor vs temperature
import numpy as np
import matplotlib.pyplot as plt
# fill_factor = [0.57,
# 0.67,
# 0.77,
# 0.87]
# prod = [0.062695331,
# 0.07111506,
# 0.067728074,
# 0.069372269]


loss_factor = []
eps = []
for ff in fill_factor:
    eps_olivate = complex(7.27, 0.0685)
    eps_A = complex(1, 0)
    A = 2
    B = (1-3*ff)*eps_olivate-(2-3*ff)*eps_A
    C = -eps_A*eps_olivate
    res = (-B+np.sqrt(B**2-4*A*C))/(2*A)
    eps.append(res.real)
    loss_factor.append(res.imag)
    
eps, loss_factor = np.array(eps), np.array(loss_factor)

temp_olivine = np.array([[
    [21, 38, 57, 62, 73],
    [21, 29, 34, 40, 45],
    [21, 25, 33, 40, 44],
],[
    [21, 56, 76, 87, 95],
    [21, 42, 46, 54, 63], 
    [21, 36, 43, 46, 56],
]])

temp_diopside = np.array([

])

temp_diff = np.diff(temp_olivine)

temp = np.array([95, 63, 56])-21

prod = E_field_squared*eps/(fill_factor)

const = np.mean(temp/prod)

prod_temp = np.array([const*ele for ele in prod])
prod_temp_per_min = (prod_temp)/4
prod_temp_over_time = []
for i in range(5):
    prod_temp_over_time.append(prod_temp_per_min*(i)+21)
prod_temp_over_time = np.array(prod_temp_over_time)

# %%
from scipy import optimize

heating_coef_start = 10
time_step = 0.01
ambient = 21
time = np.arange(0, 4, time_step)
heating_coefs = []
sim_plots = []
ff = 0

conduct_coef_optimal =[]
bracket = [0.02, 0.05]

for i in range(3):
    exp_temp = temp_olivine[1][i]
    ff = fill_factor[i]
    sol = optimize.minimize_scalar(get_mse_for_conduct_coef, bracket= bracket, method='brent')
    conduct_coef_optimal.append(sol.x)

for i in range(3):
    exp_temp = temp_olivine[1][i]
    ff = fill_factor[i]
    heating_coefs.append(get_heating_coef(conduct_coef_optimal[i], heating_coef_start))

def get_heat(conduct_coef, heating_coef):
    temperature = np.empty_like(time)
    temperature[0] = ambient
    for i, dt in enumerate(time):
        if i == len(time)-1:
            continue
        heat_equ = time_step*conduct_coef*(-temperature[i]*2+2*ambient)
        temperature[i+1] = temperature[i] + heat_equ + time_step*heating_coef
    return temperature[-1]-exp_temp[-1]

def get_heat_plot(conduct_coef, heating_coef):
    temperature = np.empty_like(time)
    temperature[0] = ambient
    for i, dt in enumerate(time):
        if i == len(time)-1:
            continue
        heat_equ = time_step*conduct_coef*(-temperature[i]*2+2*ambient)
        temperature[i+1] = temperature[i] + heat_equ + time_step*heating_coef
    return temperature

from functools import partial

def get_heating_coef(conduct_coef, heating_coef):
    bracket = [heating_coef-10, heating_coef+5000]
    get_heat_partial = partial(get_heat, conduct_coef)
    sol = optimize.root_scalar(get_heat_partial, x0=heating_coef, bracket= bracket, method='brentq')

    return sol.root

mse = []

def get_mse_for_conduct_coef(conduct_coef):
    temp_over_time = get_heat_plot(conduct_coef, get_heating_coef(conduct_coef, heating_coef_start))
    points_index = np.linspace(0, len(temp_over_time)-1, 5, dtype=np.int)
    mse = np.sqrt(np.mean((temp_over_time[points_index] - exp_temp)**2))
    return mse


heating_coefs = np.array(heating_coefs)
E_field_coef = np.mean(heating_coefs/E_field_pre_factor)
sim_heating_coefs = E_field_coef*E_field_pre_factor

for i in range(3):
    sim_plots.append(get_heat_plot(conduct_coef_optimal[i], heating_coefs[i]))

sim_plots = np.array(sim_plots)

print(conduct_coef_optimal, sim_heating_coefs)

cmap = ['y','g','r']

for i in range(3):
    plt.plot(time, sim_plots[i], color=cmap[i])


for i in range(3):
    plt.plot(temp_olivine[1][i], '*', color=cmap[i])


# plt.plot(prod_temp_over_time, '*')
# plt.plot(temp_olivine[1].transpose(), '+')
plt.xlabel('time (minutes)')
plt.ylabel('Temperature')
plt.title('Experimental temperature')
plt.legend(['<75um sim', '75-125um sim', '>125um sim', '<75um exp', '75-125um exp', '>125um exp'])
plt.ylim([15, 100])
plt.savefig('/mnt/c/peter_abaqus/Summer-Research-Project/output/export/3' + '/sim_ode_values_const.png', dpi=400, bbox_inches='tight')

# %% mse plot

# sim_heating_coefs = E_field_pre_factor
# sim_data_plot = [get_heat_plot]


# %%
E_field_coef
# %%
const
# %%
