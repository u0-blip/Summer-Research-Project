import sys
import os
import numpy as np
from time import time
import pickle
import traceback
import logging
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import contextlib
import io
from functools import partial
from copy import deepcopy
import redis

sys.path.append(os.getcwd())

from my_meep.meep_funcs import read_windows, write_windows
from my_meep.animator import my_animate, my_rms_plot, plot_3d
from my_meep.helper import translate, plot_f_name, get_offset
from my_meep.configs import *

config = None

def process_data(ez_data, eps, config):
    res = config.getfloat('sim', 'resolution')
    if not config.getboolean('general', 'perform_mp_sim'):
        try:
            ez_data = read_windows(
                data_dir + project_name + '.mpout').transpose()
            ez_data = np.moveaxis(ez_data, -1, 0)
            eps = read_windows(data_dir + project_name + '.eps').transpose()
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.log(1, e)
    else:
        ez_data = ez_data.transpose()

    return ez_data, eps

def D2_plot(ez_data, ez_trans, eps, config, web):
    global cell_size
    res = config.getfloat('sim', 'resolution')
    X = np.arange(-cell_size[0]/2, cell_size[0]/2, 1/res)
    Y = np.arange(-cell_size[1]/2, cell_size[1]/2, 1/res)
    X, Y = np.meshgrid(X, Y)

    if np.max(eps) - np.min(eps) > 0.1:
        trans = translate(np.min(eps), np.max(eps), 0, 254)
        vtrans = np.vectorize(trans)
        eps = vtrans(eps).astype(np.uint8)
        import cv2
        eps_edges = cv2.Canny(eps, 10, 20).astype(np.bool)
    else:
        eps_edges = []
    eps_rock = eps >= 7.69

    if config.getboolean('visualization', 'transiant') and config.getboolean('general', 'perform_mp_sim'):
        ez_trans = ez_trans.transpose()
        ez_trans = np.moveaxis(ez_trans, -1, 0)
        my_animate(ez_trans, window=1)

    if config.getboolean('visualization', 'rms'):
        out_every = config.getfloat('sim', 'out_every')
        time_sim = config.getfloat('sim', 'time')
        cell_size = get_array('geo', 'cell_size', config)

        if len(ez_data.shape) == 3:
            start = int(cell_size[0]*2/out_every*3)

            # 3 is to ensure the slower wave in the medium fully propogate
            end = len(ez_data) - 1
            if start >= end:
                print('Time interval is not sufficient')
                start = end - 20

            print('Time period for RMS: ', [start, end])
            ez_data[-2, eps_edges] = np.max(ez_data)*len(ez_data)/20
            my_rms_plot(ez_data, 0, 'rms', [start, end])

        elif len(ez_data.shape) == 2:
            view_only_particles = config.getboolean(
                'visualization', 'view_only_particles')
            if view_only_particles:
                ez_data[np.logical_not(eps_rock)] = 0
            else:
                # ez_data[eps_edges] = np.max(ez_data)*0.8
                ez_data[eps_edges] = 5

            fig = plt.figure(figsize=(7, 6))

            cbar_scale = get_array('visualization', 'cbar_scale', config)

            if not view_only_particles:
                ax = plt.axes()
                graph = plt.pcolor(X, Y, ez_data,
                    vmin=0, vmax=0.5)
                cb = fig.colorbar(graph, ax=ax)
                cb.set_label(label='E^2 (V/m)^2',
                                size='xx-large', weight='bold')
                cb.ax.tick_params(labelsize=20)
            else:
                cbar_scale /= 6
                ax = fig.gca(projection='3d')
                trans = translate(-cell_size[0]/2,
                                    cell_size[0]/2, 0, ez_data.shape[0])
                ax_lim = [-1, 1, -2, 2]
                ax_index_lim = [int(trans(ele)) for ele in ax_lim]

                graph = ax.plot_surface(X[ax_index_lim[0]:ax_index_lim[1], ax_index_lim[2]:ax_index_lim[3]], Y[ax_index_lim[0]:ax_index_lim[1], ax_index_lim[2]
                                        :ax_index_lim[3]], ez_data[ax_index_lim[0]:ax_index_lim[1], ax_index_lim[2]:ax_index_lim[3]], cmap=cm.coolwarm, linewidth=0, antialiased=False)

            ax.tick_params(axis='both', which='major', labelsize=20)

            if not web: plt.savefig(plot_f_name(config) + '_rms.png', dpi=300, bbox_inches='tight')
            # plt.show()

def viz_res(ez_data, ez_trans, eps, _config):
    global config
    global cell_size

    web = False
    config = _config
    
    ez_data, eps = process_data(ez_data, eps, config)

    print('particle area is ', np.mean(eps))

    if len(eps.shape) == 2:
        D2_plot(ez_data, ez_trans, eps, config, web)
    else:
        offset, offset_index = get_offset(ez_data)
        plot_3d(ez_data, offset, offset_index)
        # plt.show()
        if not web: plt.savefig(plot_f_name(config) + '3D_rms.png', dpi=300, bbox_inches='tight')
    
    if web:
        r = redis.Redis(host='localhost', port=6379, db=0)
        bytes_image = io.BytesIO()
        plt.savefig(bytes_image, format='png')
        print('write figure')
        bytes_image.seek(0)
        r.set('RMS image', bytes_image.read())

def viz_struct(sim, sim_dim, eps_data, _config):
    global config
    config = _config
    
    if sim_dim == 2:
        plt.figure()
        sim.plot2D()
        plt.savefig(plot_f_name(config) + 'struct.png', dpi=300, bbox_inches='tight')
    else:
        offset, offset_index = get_offset(eps_data)
        plot_3d(eps_data, offset, offset_index)


if __name__ == '__main__':
    ez_data = read_windows('/mnt/c/peter_abaqus/Summer-Research-Project/output/export/' + 'cube_r_1.0_gap_4.5_xloc_3.8_fcen_0.8167_ff_0.5_3D.ez')
    ez_data_2d = read_windows('/mnt/c/peter_abaqus/Summer-Research-Project/output/export/' + '__cube_r_1.0_gap_0.0_xloc_0.0_fcen_0.8167_ff_0.5_rt_0.0_2D.ez')

    all_mean = []

    ez_data = ez_data[45:285, 45:285, 45:285]
    ez_data_2d = ez_data_2d[45:285, 45:285]

    ez_data_mean = np.mean(ez_data)

    # for i in range(len(ez_data)):

    #     diff = ez_data[ :,i, :] - ez_data_2d
    #     # diff = np.divide(diff, ez_data_2d)
    #     diff = abs(diff)

    #     mean = np.mean(diff)
    #     all_mean.append(mean/ez_data_mean*100)

    # plt.plot(all_mean)
    # plt.ylim([0, 100])
    # plt.title('Relative error compare to EZ 3D', fontsize=25)
    # plt.xlabel('X axis location', fontsize=20)
    # plt.ylabel('relative error value (%)', fontsize=20)
    # ax = plt.gca()
    # ax.tick_params(labelsize=20)
    # plt.savefig(plot_f_name() + 'percentage_error.png',
    #                         dpi=400, bbox_inches='tight')

    ez_data *= 1.0/ez_data.max()
    ez_data_2d *= 1.0/ez_data_2d.max()

    i = 107
    ez_data = ez_data[ :,i, :]
    diff = ez_data - ez_data_2d
    diff = np.divide(diff, ez_data)
    diff = abs(diff)

    fig = plt.figure()
    graph = plt.pcolor(diff, vmin=0, vmax=100)
    ax = plt.gca()
    cb = fig.colorbar(graph, ax=ax)
    cb.ax.tick_params(labelsize=20)
    cb.set_label(label='Percentage Error',
                    size='xx-large', weight='bold')
    plt.title('Relative error compare to EZ 3D', fontsize=25)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    ax = plt.gca()
    ax.tick_params(labelsize=18)
    plt.savefig(plot_f_name() + 'rms_error.png',
                            dpi=400, bbox_inches='tight')


    fig = plt.figure()
    graph = plt.pcolor(ez_data_2d)
    ax = plt.gca()
    cb = fig.colorbar(graph, ax=ax)
    plt.title('ez_data 2d')

    fig = plt.figure()
    graph = plt.pcolor(ez_data)
    ax = plt.gca()
    cb = fig.colorbar(graph, ax=ax)
    plt.title('ez_data')

    plt.show()