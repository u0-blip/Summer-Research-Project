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
import pickle
import json

sys.path.append(os.getcwd())

from my_meep.gen_geo_helper import read_windows, write_windows
from my_meep.animator import my_animate, my_rms_plot, plot_3d
from my_meep.helper import Translate, output_file_name, get_offset
from my_meep.config.configs import get_array
from my_meep.config.config_variables import *

import mpld3
from mpld3 import plugins, utils


r = redis.Redis(port=6379, host='localhost', db=0)
        
class Save_img():
    """ 
    depend on whether the plotting is web, the funtion will attempt to pipe the image to the right direction 
    """
    def __init__(self, config, current_user_id):
        self.web = config.getboolean('web', 'web')
        self.config = config
        self.current_user_id = current_user_id

    def __call__(self, name):
        if self.web:
            # bytes_image = io.BytesIO()
            # plt.savefig(bytes_image, format='png')
            # bytes_image.seek(0)
            # r.set(str(self.current_user_id) + name, bytes_image.read())
            if 'rms' in name:
                return 

            fig = plt.gcf()
            plot_string = mpld3.fig_to_html(fig, d3_url=None, mpld3_url=None, no_extras=False, template_type='general', figid=None, use_http=False)
            r.set('user_' + str(self.current_user_id) + '_plot_' + name, plot_string)
        else:
            # save to local dir
            plt.savefig(output_file_name(self.config) + name + '.png', dpi=300, bbox_inches='tight')
        plt.close()

class Plot_res():
    """
    The class will both plot structure and RMS
    depend on the shape of the eps data and electric field data, it plots the corresponding correct graphes
    """

    def __init__(self, result_manager, Simulation, current_user_id):
        self.ez_data = result_manager.ez_data.transpose()
        self.eps=result_manager.eps.transpose()
        self.eps_rock = result_manager.eps_rock.transpose()
        self.ez_data_particle_only = result_manager.ez_data_particle_only.transpose()
        self.config = result_manager.config
        self.Simulation=Simulation
        self.current_user_id = current_user_id
        self.cell_size = conf.get_array('Geometry', 'cell_size', self.config)

        self.ez_dim = len(self.ez_data.shape)
        self.eps_dim = len(self.eps.shape)
        r.set('user_' + str(self.current_user_id) + '_plot_eps', json.dumps(self.eps.tolist()))
        
        self.get_configs()
        self.get_eps_edge()
        self.get_conture_axis()
        self.save_img = Save_img(self.config, current_user_id)
        r.set('user_' + str(self.current_user_id) + '_plot_rms_particle_only', json.dumps(self.ez_data_particle_only.tolist()))
        r.set('user_' + str(self.current_user_id) + '_plot_rms_particle_only_log', json.dumps(np.log(self.ez_data_particle_only).tolist()))
        # r.set('user_' + str(self.current_user_id) + '_plot_rms_eps', json.dumps(self.eps.tolist()))

    def get_eps_edge(self):
        if np.max(self.eps) - np.min(self.eps) > 0.1:
            self.translate = Translate(np.min(self.eps), np.max(self.eps), 0, 254)
            vtrans = np.vectorize(self.translate)
            self.eps = vtrans(self.eps).astype(np.uint8)
            self.eps_edges = cv2.Canny(self.eps, 50, 150).astype(np.bool)
            # r.set('user_' + str(self.current_user_id) + '_plot_rms_eps_edge', json.dumps(self.eps_edges.tolist()))
        else:
            self.eps_edges = None
        
    def get_configs(self):
        self.web = self.config.getboolean('web', 'web')
        self.res = self.config.getfloat('Simulation', 'resolution')
        self.out_every = self.config.getfloat('Simulation', 'out_every')
        self.time_sim = self.config.getfloat('Simulation', 'time')
        self.cbar_scale = get_array('Visualization', 'cbar_scale', self.config)
        self.view_only_particles = self.config.getboolean('Visualization', 'view_only_particles')

    def get_conture_axis(self):
        X = np.arange(-self.cell_size[0]/2, self.cell_size[0]/2, 1/self.res)
        Y = np.arange(-self.cell_size[1]/2, self.cell_size[1]/2, 1/self.res)
        self.X, self.Y = np.meshgrid(X, Y)
        r.set('user_' + str(self.current_user_id) + '_plot_rms_xy', json.dumps([self.X[0, :].tolist(), self.Y[:, 0].tolist()]))

    def structure_plot(self):
        sim_dim = self.config.getint('Simulation', 'dimension')
        if sim_dim == 2:
            plt.figure(figsize=[4,4])
            self.Simulation.plot2D()
        else:
            offset, offset_index = get_offset(self.eps)
            plot_3d(self.eps, offset, offset_index, self.config)

    def transient_3d(self):
        ez_trans = self.ez_trans
        ez_trans = np.moveaxis(ez_trans, -1, 0)
        my_animate(ez_trans, window=1)

    def static_3d(self):
        offset, offset_index = get_offset(self.ez_data)
        plot_3d(self.ez_data, offset, offset_index, self.config)

    def transient_2d(self):
        start = int(self.cell_size[0]*2/self.out_every*3)

        # 3 is to ensure the slower wave in the medium fully propogate
        end = len(self.ez_data) - 1
        if start >= end:
            print('Time interval is not sufficient')
            start = end - 20

        print('Time period for RMS: ', [start, end])
        self.ez_data[-2, self.eps_edges] = np.max(self.ez_data)*len(self.ez_data)/20
        my_rms_plot(self.ez_data, 0, 'rms', [start, end])

    def add_particle_edge(self):
        if self.eps_edges is not None:
            self.ez_data[self.eps_edges] = 5


    def static_2d(self, ez_data):
        pass
        # fig = plt.figure(figsize=(4,4))
        # ax = plt.axes()
        # graph = plt.pcolor(self.X, self.Y, ez_data, vmin=0, vmax=0.01)
        # cb = fig.colorbar(graph, ax=ax)
        # cb.set_label(label='E^2 (V/m)^2', size='xx-large', weight='bold')
        # cb.ax.tick_params(labelsize=20)        
        # ax.tick_params(axis='both', which='major', labelsize=20)
        # plt.title('Fill Factor is ' + self.config.get('Geometry','fill_factor'), fontsize=20)

    def static_2d_all(self):
        r.set('user_' + str(self.current_user_id) + '_plot_rms_block', json.dumps(self.ez_data.tolist()))
        r.set('user_' + str(self.current_user_id) + '_plot_rms_block_log', json.dumps(np.log(self.ez_data).tolist()))

    def static_2d_particle(self):
        pass
        # r.set('user_' + str(self.current_user_id) + '_plot_rms_particle_only', json.dumps(self.ez_data_particle_only.tolist()))

    def static_2d_particle_contour(self):
        fig = plt.figure(figsize=(7, 6))

        self.cbar_scale /= 6
        ax = fig.gca(projection='3d')
        trans = self.translate(-self.cell_size[0]/2,
                            self.cell_size[0]/2, 0, self.ez_data_particle_only.shape[0])
        ax_lim = [-1, 1, -2, 2]
        ax_index_lim = [int(trans(ele)) for ele in ax_lim]

        graph = ax.plot_surface(self.X[ax_index_lim[0]:ax_index_lim[1], ax_index_lim[2]:ax_index_lim[3]], self.Y[ax_index_lim[0]:ax_index_lim[1], ax_index_lim[2] :ax_index_lim[3]], self.ez_data_particle_only[ax_index_lim[0]:ax_index_lim[1], ax_index_lim[2]:ax_index_lim[3]], cmap=cm.coolwarm, linewidth=0, antialiased=False)    
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.title('Fill Factor is ' + self.config.get('Geometry','fill_factor'), fontsize=20)

    def __call__(self):
        if self.config.getboolean('Visualization', 'structure'):
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
            self.add_particle_edge()
            self.static_2d_particle()
            self.static_2d_all()

        self.save_img('rms')

