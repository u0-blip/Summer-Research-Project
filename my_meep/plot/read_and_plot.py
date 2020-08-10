import sys
import os
import random
from configs import *
import math
import cmath
import numpy as np
import pickle
import traceback
import logging
from matplotlib import cm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser(description='Process some integers.')
parser.add_argument('-n', '--name', help='Give the name of the file to plot')

args = parser.parse_args()
f_name = args.name


dir = '/mnt/c/peter_abaqus/Summer-Research-Project/output/export/3/'
output_file_name = dir + f_name + '.std'

with open(output_file_name, 'rb') as f:
    all_data = pickle.load(f)

param_name = all_data['param names']
title = ' '.join(param_name)
axis = []
axis = [np.linspace(*ele) for ele in all_data['iter vals']]
area = all_data['area']
data = all_data['std mean']

linear_regress = False
if linear_regress:
    to_fit = data[2][:, 0]
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import PolynomialFeatures
    polynomial_features= PolynomialFeatures(degree=2)
    x_poly = polynomial_features.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)
    y_poly_pred = model.predict(x_poly)

    plt.scatter(x, y, s=10)
    # sort the values of x before line plot
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
    x, y_poly_pred = zip(*sorted_zip)
    plt.plot(x, y_poly_pred, color='m')
    plt.show()


from scipy import signal
b, a = signal.butter(5, 0.4, 'low', analog=False)
data = np.nan_to_num(data)

# plt.plot(axis[0], area[axis[0]])
# plt.show()
def plot_one(data):
    axis = axis[0]
    filt_data = []
    
    filt_data.append(signal.filtfilt(b, a, data[:, 0]))
    filt_data.append(signal.filtfilt(b, a, data[:, 1]))

    plt.subplot(2, 1, 1)
    plt.title(title + ' mean')
    plt.scatter(axis, data[:, 0])
    plt.plot(axis, filt_data[0], 'm')
    plt.ylim(bottom = 0)
    plt.subplot(2, 1, 2)
    plt.title(title + ' std')
    plt.scatter(axis, data[:, 1])
    plt.plot(axis, filt_data[1], 'm')
    plt.ylim((0, 0.01))
        
def plot_multi(data):
    # filt_data = []
    # filt_data.append([signal.filtfilt(b, a, data[:, 0])])
    # filt_data.append([signal.filtfilt(b, a, data[:, 1])])

    plt_axis = 0
    legend_axis = 0 if plt_axis else 1

    if plt_axis == 0:
        pass
    else:
        data = np.moveaxis(data, 0, 1)

    plt.subplot(2, 1, 1)
    plt.title('mean')
    plt.ylabel('E^2/(m^2*s)')
    plt.xlabel(param_name[plt_axis])
    plt.plot(axis[plt_axis], data[:, :, 0])
    plt.legend([str(ele) for ele in axis[legend_axis]], title=param_name[legend_axis])
    plt.subplot(2, 1, 2)
    plt.title('std')
    plt.plot(axis[plt_axis], data[:, :, 1])

def save_plot(name):
    png_name = [ele.strip() for ele in f_name.split('_') if ele is not ''][1:]
    png_name = '_'.join(png_name)
    plt.savefig(dir + name + '_' + png_name + '.png', dpi=400, bbox_inches='tight')


if 'shape' not in all_data['param names']:
    if len(axis) == 1:
        axis = axis[0]
        filt_data = []
        print(data)
        filt_data.append(signal.filtfilt(b, a, data[:, 0]))
        filt_data.append(signal.filtfilt(b, a, data[:, 1]))

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.title(title + ' mean')
        plt.scatter(axis, data[:, 0])
        plt.plot(axis, filt_data[0], 'm')
        plt.ylim(bottom = 0)
        plt.subplot(2, 1, 2)
        plt.title(title + ' std')
        plt.scatter(axis, data[:, 1])
        plt.plot(axis, filt_data[1], 'm')
        plt.ylim((0, 0.01))
        plt.savefig(output_file_name + '.png', dpi=400, bbox_inches='tight')

    elif len(axis) > 1:
        # filt_data = []
        # filt_data.append([signal.filtfilt(b, a, data[:, 0])])
        # filt_data.append([signal.filtfilt(b, a, data[:, 1])])

        data = np.moveaxis(data, 0, 1)
        print(axis)


        plt.figure()
        plt.subplot(2, 1, 1)
        plt.title('mean')
        plt.plot(axis[1], data[:, :, 0])
        plt.legend([str(ele) for ele in axis[0]])
        plt.subplot(2, 1, 2)
        plt.title('std')
        plt.plot(axis[1], data[:, :, 1])
        plt.savefig(output_file_name + '.png', dpi=400, bbox_inches='tight')
else:
    shape_pos = all_data['param names'].index('shape')
    shapes = [int(ele) for ele in axis.pop(shape_pos)]
    param_name.pop(0)

    shapes_type = ['sphere', 'triangle', 'hexagon', 'cube']
    og_data = data

    for i in shapes:
        data = np.take(og_data, i, 0)
        figure = plt.figure()
        if len(axis) == 1:
            plot_one(data)

        elif len(axis) > 1:
            plot_multi(data)
        plt.subplots_adjust(hspace = 0.6)
        save_plot(shapes_type[i])
        