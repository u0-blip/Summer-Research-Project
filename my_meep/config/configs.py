import configparser
import numpy as np
import os
from copy import deepcopy
import cmath
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
from itertools import product



if 'config' not in globals():
    config = configparser.ConfigParser()
    if os.name == 'nt':
        config.read('C:\peter_abaqus\Summer-Research-Project\my_meep\config\sim.ini')
        data_dir = config.get('process_inp', 'data')
    elif os.name == 'posix':
        config.read('/mnt/c/peter_abaqus/Summer-Research-Project/my_meep/config/sim checker test.ini')
        # config.read('/mnt/c/peter_abaqus/Summer-Research-Project/meep/sim checker fill factor.ini')
        data_dir = config.get('process_inp', 'posix_data')

# config_og = deepcopy(config)


def get_array(section, name, config, type = np.float):
    val = config.get(section, name)
    val = np.fromstring(val, dtype = type, sep=',')
    return val


class Iter_vars:
    ele = []
    index = np.array([])
    nums = 0

    def __init__(self, params, param_names):
        self.param_names = param_names
        self.params = params
        for i, ele in enumerate(self.params):
            if ele[-1] > 1:
                self.ele.append(ele)
                self.index = np.append(self.index, i).astype(np.int)
        self.ele = np.array(self.ele)
        self.get_vars_size()
        self.nums = len(self.ele)

        # data = {
        #     'param names': [param_names[i] for i in iter_vars_index], 
        #     'iter vals': iter_vars_ele, 
        #     'std mean' : mean_stds,
        #     'area' : areas
        #     }
        
    def get_vars_size(self):
        if len(self.ele) >= 1:
            self.size = self.ele[:, -1].astype(np.int)
        else:
            self.size = [1]

    def get_index(self, dimension):
        if self.nums == 0 or dimension >= self.nums:
            return [0]
        else:
            return np.linspace(self.ele[dimension, 0], self.ele[dimension, 1], int(self.ele[dimension, 2]))

    def __str__(self):
        var_descrip_str = ''
        if len(self.ele) != 0:
            p = self.ele[0]
            i = self.index[0]
            for i, p in zip(self.index, self.ele):
                var_descrip_str += self.param_names[i] + '_' + str('_'.join([str(p) for p in self.params[i]]))
        return var_descrip_str

class Config_manager:
    def __init__(self, config):
        self.config = config

    def break_down_config(self):
        """ 
        take in the configuration and output a list of configurations to be ran
        """

        shape_str = [ele.strip() for ele in self.config.get('geo', 'shape_types').split(',')]

        self.params = []
        self.param_names = ['shape', 'fill_factor', 'particle_size', 'x_loc', 'distance', 'std', 'rotation']
        param_cat = ['geo', 'geo', 'geo', 'geo', 'geo', 'geo', 'geo', 'geo']


        for cat, name in zip(param_cat, self.param_names):
            temp = list(get_array(cat, name, self.config))
            temp[-1] = int(temp[-1])
            self.params.append(temp)

        self.total = np.prod([ele[-1] for ele in self.params])

        configs = []
        for param_val in product(*[np.linspace(*ele) for ele in self.params]):
            for i, val in enumerate(param_val):
                if self.param_names[i] == 'shape':
                    self.config.set(param_cat[i], self.param_names[i], shape_str[int(val)])
                else:
                    self.config.set(param_cat[i], self.param_names[i], str(val))
            configs.append(deepcopy(self.config))
        return configs

    def sort_res(self, mean_std_area_value):
        iter_vars = Iter_vars(self.params, self.param_names)
        
        areas = np.array([])
        means = np.array([])
        stds = np.array([])
        for mean, std, area in mean_std_area_value:
            areas = np.append(areas, area)
            means = np.append(means, mean)
            stds = np.append(stds, std)

        data = [means, stds, areas]
        for i, d in enumerate(data):
            data[i] = d.reshape(*iter_vars.size)
            data[i] = pd.DataFrame(data[i])
            data[i].colums = iter_vars.get_index(1)
            data[i].index = iter_vars.get_index(0)
        
        return data, str(iter_vars)
      