import configparser
import argparse
import numpy as np
import os
from copy import deepcopy

config = configparser.ConfigParser()


if os.name == 'nt':
    config.read('C:\peter_abaqus\Summer-Research-Project\my_meep\sim.ini')
    data_dir = config.get('process_inp', 'data')
elif os.name == 'posix':
    config.read('/mnt/c/peter_abaqus/Summer-Research-Project/my_meep/sim checker test.ini')
    # config.read('/mnt/c/peter_abaqus/Summer-Research-Project/meep/sim checker fill factor.ini')
    data_dir = config.get('process_inp', 'posix_data')

# config_og = deepcopy(config)

def get_array(section, name, config, type = np.float):
    val = config.get(section, name)
    val = np.fromstring(val, dtype = type, sep=',')
    return val

file_name = config.get('process_inp', 'posix_data') + config.get('process_inp', 'project_name') + '.mpout'


per_sim = config.getboolean('general', 'perform_mp_sim')    
# fcen = eval(config.get('source', 'fcen')) # center frequency of CW source (wavelength is 1 μm)
sim_dim = config.getint('sim', 'dimension')
cell_size = get_array('geo', 'cell_size', config)
vor_size = get_array('vor', 'size', config)
vor_center = get_array('vor', 'center', config)
type_s = config.get('sim', 'type')
project_name = config.get('process_inp', 'project_name')
