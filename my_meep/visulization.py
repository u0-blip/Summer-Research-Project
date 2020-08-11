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
import cv2

sys.path.append(os.getcwd())

from my_meep.gen_geo_helper import read_windows, write_windows
from my_meep.animator import my_animate, my_rms_plot, plot_3d
from my_meep.helper import Translate, output_file_name, get_offset
from my_meep.config.configs import get_array
from my_meep.config.config_variables import *



        
class Save_img():
    """ 
    depend on whether the plotting is web, the funtion will attempt to pipe the image to the right direction 
    """
    def __init__(self, config):
        self.rdb = redis.Redis(host='localhost', port=6379, db=0)
        self.web = config.getboolean('web', 'web')
        self.config = config

    def __call__(self, name):
        if self.web:
            bytes_image = io.BytesIO()
            plt.savefig(bytes_image, format='png')
            bytes_image.seek(0)
            self.rdb.set(name, bytes_image.read())
        else:
            # save to local dir
            plt.savefig(output_file_name(self.config) + name + '.png', dpi=300, bbox_inches='tight')

class Plot_res():
    """
    The class will both plot structure and RMS
    depend on the shape of the eps data and electric field data, it plots the corresponding correct graphes
    """
    def get_ez_from_file(self):
        if not self.config.getboolean('general', 'perform_mp_sim'):
            ez_data = read_windows(data_dir + project_name + '.mpout').transpose()
            self.ez_data = np.moveaxis(ez_data, -1, 0)
            self.eps = read_windows(data_dir + project_name + '.eps').transpose()

    def get_eps_edge(self):
        if np.max(self.eps) - np.min(self.eps) > 0.1:
            self.translate = Translate(np.min(self.eps), np.max(self.eps), 0, 254)
            vtrans = np.vectorize(self.translate)
            self.eps = vtrans(self.eps).astype(np.uint8)
            self.eps_edges = cv2.Canny(self.eps, 10, 20).astype(np.bool)
        else:
            self.eps_edges = None
        
    def get_configs(self):
        self.web = self.config.getboolean('web', 'web')
        self.res = self.config.getfloat('sim', 'resolution')
        self.out_every = self.config.getfloat('sim', 'out_every')
        self.time_sim = self.config.getfloat('sim', 'time')
        self.cbar_scale = get_array('visualization', 'cbar_scale', self.config)
        self.view_only_particles = self.config.getboolean('visualization', 'view_only_particles')

    def get_conture_axis(self):
        X = np.arange(-cell_size[0]/2, cell_size[0]/2, 1/self.res)
        Y = np.arange(-cell_size[1]/2, cell_size[1]/2, 1/self.res)
        self.X, self.Y = np.meshgrid(X, Y)
        

    def __init__(self, result_manager, sim, eps_data):


        self.ez_data = result_manager.ez_data.transpose()
        self.eps=result_manager.eps.transpose()
        self.eps_rock = result_manager.eps_rock.transpose()
        self.ez_data_particle_only = result_manager.ez_data_particle_only.transpose()
        self.config = result_manager.config

        self.sim=sim
        self.eps_data=eps_data

        self.get_ez_from_file()

        self.ez_dim = len(self.ez_data.shape)
        self.eps_dim = len(self.eps_data.shape)
        self.get_configs()
        self.get_eps_edge()
        self.get_conture_axis()
        self.save_img = Save_img(self.config)
        
    def structure_plot(self):
        if sim_dim == 2:
            plt.figure()
            self.sim.plot2D()
        else:
            offset, offset_index = get_offset(self.eps_data)
            plot_3d(self.eps_data, offset, offset_index)

    def transient_3d(self):
        ez_trans = self.ez_trans
        ez_trans = np.moveaxis(ez_trans, -1, 0)
        my_animate(ez_trans, window=1)

    def static_3d(self):
        offset, offset_index = get_offset(self.ez_data)
        plot_3d(self.ez_data, offset, offset_index)

    def transient_2d(self):
        start = int(cell_size[0]*2/self.out_every*3)

        # 3 is to ensure the slower wave in the medium fully propogate
        end = len(self.ez_data) - 1
        if start >= end:
            print('Time interval is not sufficient')
            start = end - 20

        print('Time period for RMS: ', [start, end])
        self.ez_data[-2, self.eps_edges] = np.max(self.ez_data)*len(self.ez_data)/20
        my_rms_plot(self.ez_data, 0, 'rms', [start, end])

    def add_particle_edge(self):
        self.ez_data[self.eps_edges] = 5

    def static_2d(self, ez_data):
        fig = plt.figure(figsize=(7, 6))
        ax = plt.axes()
        graph = plt.pcolor(self.X, self.Y, ez_data, vmin=0, vmax=0.01)
        cb = fig.colorbar(graph, ax=ax)
        cb.set_label(label='E^2 (V/m)^2', size='xx-large', weight='bold')
        cb.ax.tick_params(labelsize=20)        
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.title('Fill Factor is ' + self.config.get('geo','fill_factor'), fontsize=20)

    def static_2d_all(self):
        self.static_2d(self.ez_data)

    def static_2d_particle(self):
        self.static_2d(self.ez_data_particle_only)

    def static_2d_particle_contour(self):
        fig = plt.figure(figsize=(7, 6))

        self.cbar_scale /= 6
        ax = fig.gca(projection='3d')
        trans = self.translate(-cell_size[0]/2,
                            cell_size[0]/2, 0, self.ez_data_particle_only.shape[0])
        ax_lim = [-1, 1, -2, 2]
        ax_index_lim = [int(trans(ele)) for ele in ax_lim]

        graph = ax.plot_surface(self.X[ax_index_lim[0]:ax_index_lim[1], ax_index_lim[2]:ax_index_lim[3]], self.Y[ax_index_lim[0]:ax_index_lim[1], ax_index_lim[2] :ax_index_lim[3]], self.ez_data_particle_only[ax_index_lim[0]:ax_index_lim[1], ax_index_lim[2]:ax_index_lim[3]], cmap=cm.coolwarm, linewidth=0, antialiased=False)    
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.title('Fill Factor is ' + self.config.get('geo','fill_factor'), fontsize=20)

    def __call__(self):
        if self.config.getboolean('visualization', 'structure'):
            self.structure_plot()
            self.save_img('structure')
            
        ez_dim = self.ez_dim
        eps_dim = self.eps_dim

        if ez_dim == 4 and eps_dim == 3:
            print('trans 3d')
            self.transient_3d()
        elif ez_dim == 3 and eps_dim == 3:
            print('static 3d')
            self.static_3d()
        elif ez_dim == 3 and eps_dim == 2:
            print('trans 2d')
            self.transient_2d()
        elif ez_dim == 2 and eps_dim == 2:
            if self.view_only_particles:
                print('static 2d particles')
                self.add_particle_edge()
                self.static_2d_particle()
            else:
                print('static 2d')
                self.add_particle_edge()
                self.static_2d_all()

        self.save_img('rms')


# if __name__ == '__main__':
#     ez_data = read_windows('/mnt/c/peter_abaqus/Summer-Research-Project/output/export/' + 'cube_r_1.0_gap_4.5_xloc_3.8_fcen_0.8167_ff_0.5_3D.ez')
#     ez_data_2d = read_windows('/mnt/c/peter_abaqus/Summer-Research-Project/output/export/' + '__cube_r_1.0_gap_0.0_xloc_0.0_fcen_0.8167_ff_0.5_rt_0.0_2D.ez')

#     all_mean = []

#     ez_data = ez_data[45:285, 45:285, 45:285]
#     ez_data_2d = ez_data_2d[45:285, 45:285]

#     ez_data_mean = np.mean(ez_data)

#     # for i in range(len(ez_data)):

#     #     diff = ez_data[ :,i, :] - ez_data_2d
#     #     # diff = np.divide(diff, ez_data_2d)
#     #     diff = abs(diff)

#     #     mean = np.mean(diff)
#     #     all_mean.append(mean/ez_data_mean*100)

#     # plt.plot(all_mean)
#     # plt.ylim([0, 100])
#     # plt.title('Relative error compare to EZ 3D', fontsize=25)
#     # plt.xlabel('X axis location', fontsize=20)
#     # plt.ylabel('relative error value (%)', fontsize=20)
#     # ax = plt.gca()
#     # ax.tick_params(labelsize=20)
#     # plt.savefig(output_file_name(self.config) + 'percentage_error.png',
#     #                         dpi=400, bbox_inches='tight')

#     ez_data *= 1.0/ez_data.max()
#     ez_data_2d *= 1.0/ez_data_2d.max()

#     i = 107
#     ez_data = ez_data[ :,i, :]
#     diff = ez_data - ez_data_2d
#     diff = np.divide(diff, ez_data)
#     diff = abs(diff)

#     fig = plt.figure()
#     graph = plt.pcolor(diff, vmin=0, vmax=100)
#     ax = plt.gca()
#     cb = fig.colorbar(graph, ax=ax)
#     cb.ax.tick_params(labelsize=20)
#     cb.set_label(label='Percentage Error',
#                     size='xx-large', weight='bold')
#     plt.title('Relative error compare to EZ 3D', fontsize=25)
#     plt.xlabel('x', fontsize=20)
#     plt.ylabel('y', fontsize=20)
#     ax = plt.gca()
#     ax.tick_params(labelsize=18)
#     plt.savefig(output_file_name(self.config) + 'rms_error.png',
#                             dpi=400, bbox_inches='tight')


#     fig = plt.figure()
#     graph = plt.pcolor(ez_data_2d)
#     ax = plt.gca()
#     cb = fig.colorbar(graph, ax=ax)
#     plt.title('ez_data 2d')

#     fig = plt.figure()
#     graph = plt.pcolor(ez_data)
#     ax = plt.gca()
#     cb = fig.colorbar(graph, ax=ax)
#     plt.title('ez_data')

#     plt.show()