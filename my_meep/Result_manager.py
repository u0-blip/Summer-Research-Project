import cmath
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io

from my_meep.config.configs import get_array
from my_meep.config.config_variables import *
from my_meep.helper import Translate, output_file_name
from my_meep.gen_geo_helper import read_windows, write_windows
import redis

class Result_manager():
    @staticmethod
    def get_area(eps, config):
        eps_const = config.getfloat('geo', 'eps')
        eps_rock = eps >= eps_const
        area = np.sum(eps_rock)
        area = area/np.prod(eps.shape)
        return area

    def get_roi_res_block(self):
        """ only take the center 3x3 area as region of interest """
        res_block = [-3, 3, -3, 3]
        translate = Translate(-cell_size[0]/2, cell_size[1]/2, 0, self.eps.shape[0])
        res_block_index = [int(translate(ele)) for ele in res_block]
        
        res_block_mat = np.zeros_like(self.eps)
        res_block_mat[res_block_index[0]:res_block_index[1], res_block_index[2]:res_block_index[3]] = 1
        self.res_block_mat = res_block_mat

    def get_config(self):
        self.cell_size = get_array('geo', 'cell_size', self.config)
        self.eps_const = self.config.getfloat('geo', 'eps')
        self.dim = self.config.getfloat('sim', 'dimension')
        self.res = self.config.getfloat('sim', 'resolution')
        self.sim_type = self.config.get('sim', 'type') 

    def get_eps_rock(self):
        """ get the particle area of the rock """
        if self.sim_type == 'effective medium':
            self.eps_rock = self.res_block_mat.astype(np.bool)
        else:
            self.eps_rock = self.eps >= self.eps_const
            self.eps_rock = np.logical_and(self.eps_rock, self.res_block_mat)

        self.ez_data_particle_only[np.logical_not(self.eps_rock)] = 0
        
    def __init__(self, ez_data, eps, config):
        self.ez_data = ez_data
        self.ez_data_particle_only = np.copy(ez_data)
        self.eps = eps
        self.config = config

        self.get_config()
        self.get_roi_res_block()
        self.get_eps_rock()

    def result_statistics(self):
        """
        return the mean, std and area informaiton about the result
        the area means the amount of area the rock take up
        """

        mean = np.trapz(self.ez_data_particle_only, axis=0, dx=1/self.res)
        self.mean = np.trapz(mean, axis=0, dx=1/self.res)
        if self.dim == 3:
            self.mean = np.trapz(self.mean, axis=0, dx=1/self.res)

        self.std = np.std(self.ez_data_particle_only[self.eps_rock])

        if verbals:
            print('mean', self.mean, 'std', self.std)
            
        return self.mean, self.std


def write_res(config, data, var_descrip_str):
    """
    write result to file or redis database base on the configuration
    """
    if not config.getboolean('web','web'):
        with pd.ExcelWriter(output_file_name(config) + '_' + var_descrip_str + '.xlsx') as writer:  
            data[0].to_excel(writer, sheet_name='mean')
            data[1].to_excel(writer, sheet_name='std')
            data[2].to_excel(writer, sheet_name='area')
    else:
        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        data[0].to_excel(writer, sheet_name='mean')
        data[1].to_excel(writer, sheet_name='std')
        data[2].to_excel(writer, sheet_name='area')
        writer.save()
        output.seek(0)

        r = redis.Redis(port=6379, host='localhost')
        r.set('Current result', output.read())
        output.close()
