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
import json
import numpy as np

r = redis.Redis(port=6379, host='0.0.0.0', db=0)

class Result_manager():
    @staticmethod
    def get_area(eps, config):
        eps_background = get_array('Geometry', 'eps', config)[0]
        eps_rock = eps >= eps_background
        area = np.sum(eps_rock)
        area = area/np.prod(eps.shape)
        return area

    def get_roi_res_block(self):
        """ only take the center 3x3 area as region of interest """

        res_block = [-3, 3, -3, 3]
        translate = Translate(-self.cell_size[0]/2, self.cell_size[1]/2, 0, self.eps.shape[0])
        res_block_index = [int(translate(ele)) for ele in res_block]
        
        res_block_mat = np.zeros_like(self.eps)
        res_block_mat[res_block_index[0]:res_block_index[1], res_block_index[2]:res_block_index[3]] = 1
        self.res_block_mat = res_block_mat

    def get_config(self):
        self.cell_size = get_array('Geometry', 'cell_size', self.config)
        eps = get_array('Geometry', 'eps', self.config)
        eps_real = []
        self.eps_background = eps[0]
        for i, ele in enumerate(eps):
            if i % 2 == 0: eps_real.append(ele)
        np.sort(eps_real)
        bk_index = np.searchsorted(eps_real, self.eps_background)
        self.eps_background_tolerance = 0.5
        if bk_index == 0:
            self.eps_background_lower = self.eps_background - self.eps_background_tolerance
            self.eps_background_upper = (self.eps_background + eps_real[bk_index+1])/2
        elif bk_index == len(eps_real) - 1:
            self.eps_background_lower = (self.eps_background + eps_real[bk_index-1])/2
            self.eps_background_upper = self.eps_background + self.eps_background_tolerance
        else:
            self.eps_background_lower = (self.eps_background + eps_real[bk_index-1])/2
            self.eps_background_upper = (self.eps_background + eps_real[bk_index+1])/2

        self.dim = self.config.getfloat('Simulation', 'dimension')
        self.res = self.config.getfloat('Simulation', 'resolution')
        self.sim_type = self.config.get('Simulation', 'sim_types') 

    def get_eps_rock(self):
        """ get the particle area of the rock """
        if self.sim_type == 'effective medium':
            self.eps_rock = self.res_block_mat.astype(np.bool)
        else:
            self.eps_rock = np.logical_or(self.eps < self.eps_background_lower, self.eps > self.eps_background_upper)
            self.eps_rock = np.logical_and(self.eps_rock, self.res_block_mat)

        self.ez_data_particle_only[np.logical_not(self.eps_rock)] = self.ez_min
        
    def __init__(self, ez_data, eps, config, current_user_id):
        self.ez_data = ez_data
        self.ez_min = np.min(ez_data)
        self.ez_data_particle_only = np.copy(ez_data)
        self.eps = eps

        # fig = plt.figure()
        # ax = plt.axes()
        # graph = plt.pcolor(self.eps)
        # cb = fig.colorbar(graph, ax=ax)
        # plt.show()
        # exit()
        
        self.config = config

        self.get_config()
        self.get_roi_res_block()
        self.get_eps_rock()
        
        with io.BytesIO() as output:
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            ez_data_pd = pd.DataFrame(ez_data)
            ez_data_pd.to_excel(writer, sheet_name='E field data')
            writer.save()
            output.seek(0)
            r.set(str(current_user_id) + 'field_result', output.read())

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

        verbals = self.config.getboolean('General', 'verbals')
        if verbals:
            print('mean', self.mean, 'std', self.std)
            
        return self.mean, self.std

def create_info_sheet(data, param_iterated):
    param_sweep_value = []
    for sweep in [data[0].columns, data[0].index]:
        if len(sweep) > 1:
            param_sweep_value.append(sweep)

    information = pd.DataFrame(param_sweep_value)
    if len(param_iterated) != 0:
        information.index = param_iterated
    else:
        information.append(['No param is sweeped'])
    return information

def to_excel(data, information, writer):
    start_row = 0
    space = 5

    names = ['Mean', 'Parameters', 'Standard deviation', 'Actual Fill Factor']
    info_data = [information, *data]
    
    pd.DataFrame([]).to_excel(writer, sheet_name='data',startrow=0 , startcol=0)

    for name, datum in zip(names, info_data):
        worksheet = writer.sheets['data']
        worksheet.write(start_row, 0, name)
        datum.to_excel(writer, sheet_name='data',startrow=start_row+1 , startcol=0)
        start_row += len(datum.index) + space

    writer.save()

def write_res(config, data, var_descrip_str, current_user_id, param_iterated):
    """
    write result to file or redis database base on the configuration
    """
    information = create_info_sheet(data, param_iterated)
    if not config.getboolean('web','web'):
        with pd.ExcelWriter(output_file_name(config) + '_' + var_descrip_str + '.xlsx') as writer: 
            to_excel(data, information, writer)
    else:
        with io.BytesIO() as output:
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            to_excel(data, information, writer)
            output.seek(0)

            r = redis.Redis(port=6379, host='0.0.0.0', db=0)
            r.set(str(current_user_id) + 'mean_result', output.read())
