# %% reading in data
import sys
import os
import random
import math
import cmath
import numpy as np
import pandas as pd
import pickle
import traceback
import logging
from matplotlib import cm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from sklearn.linear_model import LinearRegression
import scipy.stats as stats
import pickle
import os
import sys
os.chdir(os.path.dirname(os.getcwd()))
# from scipy import signal
# b, a = signal.butter(2, 0.4, 'low', analog=False)

# parser = ArgumentParser(description='Process some integers.')
# parser.add_argument('-n', '--name', help='Give the name of the file to plot')
# args = parser.parse_args()
# f_name = args.name

# from my_meep.gen_geo_helper import read_windows

f_names = [
    'triangle_r_0.5_gap_0.0_xloc_0.0_fcen_0.8167_ff_0.01_rt_0.0_2D',
    # 'triangle_r_0.5_gap_0.0_xloc_0.0_fcen_0.8167_ff_0.5_rt_0.0_2D',
    # 'triangle_r_0.5_gap_0.0_xloc_0.0_fcen_0.8167_ff_1.0_rt_0.0_2D',
    # 'triangle_r_0.5_gap_0.0_xloc_0.0_fcen_0.8167_ff_0.58_rt_0.0_2D',
    # 'triangle_r_0.5_gap_0.0_xloc_0.0_fcen_0.8167_ff_0.63_rt_0.0_2D',
    # 'triangle_r_0.5_gap_0.0_xloc_0.0_fcen_0.8167_ff_0.59_rt_0.0_2D'
    ]
# f_name = 'cube_r_0.5_gap_0.0_xloc_0.0_fcen_0.8167_ff_0.5_rt_0.0_fill_factor_0.01_0.5_16eps_rock_7'

for f_name in f_names:
    dir = '/mnt/c/peter_abaqus/Summer-Research-Project/output/export/3/'
    plot_f_name = dir + f_name + '.ez'

    # data = read_windows(plot_f_name)
    from visulization import viz_res, viz_struct

    with open(plot_f_name, 'rb') as f:
        ez_data, ez_trans, eps_data, config = pickle.load(f)

    viz_res(ez_data, ez_trans, eps_data, config)

# with open(plot_f_name, 'rb') as f:
#     all_data = pickle.load(f)


# # %% helper function

# def plot_one(axes, x, data, pred, title, color, label, legends):
#     axes.plot(x, data, '*', color=color)
#     axes.plot(x, pred, color=color)

#     axes.set_title(title, fontsize=25)
#     axes.set_xlabel(label[0], fontsize=20)
#     axes.set_ylabel(label[1], fontsize=20, color=color)
#     # plt.legend((line,), fontsize=10)
#     plt.legend(legends)
#     axes.tick_params(labelsize=18)
#     axes.tick_params(labelsize=18, axis='y', labelcolor=color)
#     axes.set_ylim(bottom=0)
#     axes.relim()
#     return axes
        
# def plot_multi(data):
#     # filt_data = []
#     # filt_data.append([signal.filtfilt(b, a, data[:, 0])])
#     # filt_data.append([signal.filtfilt(b, a, data[:, 1])])

#     plt_axis = 0
#     legend_axis = 0 if plt_axis else 1

#     if plt_axis == 0:
#         pass
#     else:
#         data = np.moveaxis(data, 0, 1)

#     plt.subplot(2, 1, 1)
#     plt.title('mean')
#     plt.ylabel('E^2/(m^2*s)')
#     plt.xlabel(param_name[plt_axis])
#     plt.plot(axis[plt_axis], data[:, :, 0])
#     plt.legend([str(ele) for ele in axis[legend_axis]], title=param_name[legend_axis])
#     plt.subplot(2, 1, 2)
#     plt.title('std')
#     plt.plot(axis[plt_axis], data[:, :, 1])

# def save_plot(name):
#     if name != '':
#         png_name = [ele.strip() for ele in f_name.split('_') if ele is not ''][1:]
#         png_name = '_'.join(png_name)
#         png_name = name + '_' + png_name
#     else:
#         png_name = f_name

#     plt.savefig(dir + png_name + '.png', dpi=400, bbox_inches='tight')

# if 'shape' not in all_data['param names']:
#     if len(axis) == 1:
#         plot_one(data)
#     elif len(axis) > 1:
#         plot_multi(data)
        
#     plt.subplots_adjust(hspace = 0.6)
#     save_plot('')

# else:
#     shape_pos = all_data['param names'].index('shape')
#     shapes = [int(ele) for ele in axis.pop(shape_pos)]
#     param_name.pop(0)

#     shapes_type = ['sphere', 'triangle', 'hexagon', 'cube']
#     og_data = data

#     for i in shapes:
#         data = np.take(og_data, i, 0)
#         figure = plt.figure()
#         if len(axis) == 1:
#             plot_one(data)

#         elif len(axis) > 1:
#             plot_multi(data)
#         plt.subplots_adjust(hspace = 0.6)
#         save_plot(shapes_type[i])
        

# # %%
# param_name = all_data['param names']
# title = ' '.join(param_name)

# axis = [np.linspace(*ele) for ele in all_data['iter vals']]
# area = all_data['area']
# data = all_data['std mean']

# data = np.nan_to_num(data)

# from scipy.optimize import curve_fit

# loss_factor = []

# for ff in axis[0]:
#     eps_olivate = complex(7.27, 0.0685)
#     eps_A = complex(1, 0)
#     A = 2
#     B = (1-3*ff)*eps_olivate-(2-3*ff)*eps_A
#     C = -eps_A*eps_olivate
#     res = (-B+np.sqrt(B**2-4*A*C))/(2*A)
#     eps = res.real
#     eps_i = res.imag
#     loss_factor.append(eps_i)
# loss_factor = np.array(loss_factor)
# absorption = data[:, 0]*loss_factor

# absorption = data[:, 0]
# x = axis[0]

# def sigmoid(x, L ,x0, k, b):
#     y = L / (1 + np.exp(-k*(x-x0)))+b
#     return (y)

# p0 = [max(absorption), np.median(x),1,min(absorption)] # this is an mandatory initial guess

# popt, pcov = curve_fit(sigmoid, x, absorption, p0, method='dogbox')

# fit = np.polyfit(x, absorption, 1)
# reg = np.poly1d(fit)
# y_pred = reg(x)
# factors = ['L', 'x0', 'k', 'b']

# line = '  '.join([fac + ':' + '{:.2e}'.format(ele) for ele, fac in zip(popt, factors)])

# # y_pred = sigmoid(x, *popt)

# color = 'tab:blue'
# color1 = 'tab:red'

# plt.plot(axis[0], y_pred, color=color)
# # plt.plot(np.linspace(0.01, 0.5, 16), y_pred1, color=color1)
# plt.plot(axis[0], absorption, "x", color=color)
# # plt.plot(np.linspace(0.01, 0.5, 16), absorption1, "x", color=color1)

# ax1 = plt.gca()
# plt.title('fill factor on EM absorption', fontsize=25)
# ax1.set_xlabel('fill factor', fontsize=20)
# ax1.set_ylabel('EM absorption', fontsize=20, color=color)
# plt.legend((line,), fontsize=10)
# # plt.legend(('eps = 7, 1', 'eps = 7, 7'))
# ax1.tick_params(labelsize=18)
# ax1.tick_params(labelsize=18, axis='y', labelcolor=color)
# ax1.set_ylim(bottom=0)

# plt.savefig('/mnt/c/peter_abaqus/Summer-Research-Project/output/export/3' + '/cube_mean.png', dpi=400, bbox_inches='tight')

# # %% Analyse the mean value of the EM field RMS value

# absorption = data[:, 0]
# x = axis[0]

# def sigmoid(x, L ,x0, k, b):
#     y = L / (1 + np.exp(-k*(x-x0)))+b
#     return (y)

# p0 = [max(absorption), np.median(x),1,min(absorption)] # this is an mandatory initial guess

# popt, pcov = curve_fit(sigmoid, x, absorption, p0, method='dogbox')

# fit = np.polyfit(x, absorption, 1)
# reg = np.poly1d(fit)
# y_pred = reg(x)
# factors = ['L', 'x0', 'k', 'b']

# line = '  '.join([fac + ':' + '{:.2e}'.format(ele) for ele, fac in zip(popt, factors)])

# # y_pred = sigmoid(x, *popt)

# color = 'tab:blue'
# color1 = 'tab:red'

# plt.plot(axis[0], y_pred, color=color)
# # plt.plot(np.linspace(0.01, 0.5, 16), y_pred1, color=color1)
# plt.plot(axis[0], absorption, "x", color=color)
# # plt.plot(np.linspace(0.01, 0.5, 16), absorption1, "x", color=color1)

# ax1 = plt.gca()
# plt.title('fill factor on EM absorption', fontsize=25)
# ax1.set_xlabel('Particle size', fontsize=20)
# ax1.set_ylabel('EM absorption', fontsize=20)
# # plt.legend(('eps = 7, 1', 'eps = 7, 7'))
# line = ' y={:.2e}x + {:.2e}'.format(reg[1], reg[0])
# plt.legend((line,), fontsize=10)
# ax1.tick_params(labelsize=18)
# ax1.tick_params(labelsize=18, axis='y')
# ax1.set_ylim(bottom=0)

# plt.savefig('/mnt/c/peter_abaqus/Summer-Research-Project/output/export/3' + '/cube_mean.png', dpi=400, bbox_inches='tight')


# # %% fit the seasonality
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# train = pd.DataFrame(data[:, 0])
# # train.index = x

# model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=25)
# model2 = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=25, damped=True)
# fit = model.fit()
# pred = fit.predict(0, len(x))
# fit2 = model2.fit()
# pred2 = fit2.predict(0, len(x))

# sse1 = np.sqrt(np.mean(np.square(train.values - pred.values)))
# sse2 = np.sqrt(np.mean(np.square(train.values - pred2.values)))

# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(train.index, train.values)
# ax.plot(train, label='truth')
# # ax.plot(pred, linestyle='--', color='#ff7823', label="w/o damping (RMSE={:0.2f}, AIC={:0.2f})".format(sse1, fit.aic))
# ax.plot(pred2, linestyle='--', color='#3c763d', label="damped (RMSE={:0.2e}, AIC={:0.2e})".format(sse2, fit2.aic))
# ax.legend()
# ax.set_title("Holt-Winter's Seasonal Smoothing")

# # %% fit the sine
# from scipy import optimize
# pi = np.pi

# def test_func(x, a, b, c, d, e, f, g, h):
#     return a * np.sin(2*pi*b * x + c) + d * np.sin(2*pi*e * x + f) + g + h*x

# params, params_covariance = optimize.curve_fit(test_func, x, absorption,
# p0=[2e-4, 0.412, 5, 2e-4, 2.4, 2, 3e-2, 0])

# print(params[:3])
# print(params[3:6])
# print(params[6:])

# plt.figure(figsize=(6, 4))
# plt.plot(x, test_func(x, *params))
# plt.plot(axis[0], y_pred, '--')
# plt.plot(x, absorption, '*')

# ax1 = plt.gca()
# plt.title('Particle size on EM absorption', fontsize=20)
# ax1.set_xlabel('Particle size', fontsize=20)
# ax1.set_ylabel('EM absorption', fontsize=20)
# # plt.legend(('eps = 7, 1', 'eps = 7, 7'))
# # line = ' y={:.2e}x + {:.2e}'.format(reg[1], reg[0])
# params[0] *= 1e4
# line = [' sin1: A:{:.2f}e-4 f:{:.2f} ɸ:{:.2f}'.format(*params[:3])
# ,' sin2:  A:{:.2e} f:{:.2f} ɸ:{:.2f}'.format(*params[3:6])
# , 'c: {:.2e}'.format(params[6])]
# line = '\n'.join(line)
# trend = 'k: {:.2e}'.format(params[7])
# # plt.legend((line, trend, 'data'), fontsize=12)
# plt.legend(('low freq fitting', 'trend', 'data'), fontsize=12)
# ax1.tick_params(labelsize=18)
# ax1.tick_params(labelsize=18, axis='y')
# # ax1.set_ylim(bottom=0)


# plt.savefig('/mnt/c/peter_abaqus/Summer-Research-Project/output/export/3' + '/sine fit.png', dpi=400, bbox_inches='tight')

# plt.show()

# # %% fourier decomposition
# from scipy.fft import fft
# # Number of sample points
# N = len(absorption)
# # sample spacing
# T = 4/N
# yf = fft(absorption-np.mean(absorption))
# xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
# import matplotlib.pyplot as plt
# plt.stem(xf, 2.0/N * np.abs(yf[0:N//2]))

# ax1 = plt.gca()
# plt.title('Frequency spectrum', fontsize=20)
# ax1.set_xlabel('f (Hz)', fontsize=20)
# ax1.set_ylabel('Energy', fontsize=20)
# # plt.legend(('eps = 7, 1', 'eps = 7, 7'))
# # line = ' y={:.2e}x + {:.2e}'.format(reg[1], reg[0])
# # line = [' sin1: A:{:.2e} λ:{:.2e} ɸ:{:.2e}'.format(*params[:3])
# # ,' sin2:  A:{:.2e} λ:{:.2e} ɸ:{:.2e}'.format(*params[3:6])
# # , 'const: {:.2e}'.format(*params[6:])]
# # line = '\n'.join(line)

# # plt.legend((line,), fontsize=10)

# ax1.tick_params(labelsize=18, axis='y')

# plt.savefig('/mnt/c/peter_abaqus/Summer-Research-Project/output/export/3' + '/freq.png', dpi=400, bbox_inches='tight')
# plt.show()


# # %%
