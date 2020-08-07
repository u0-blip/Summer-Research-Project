import cmath
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io

from my_meep.config.configs import *
from my_meep.helper import Translate, plot_f_name
from my_meep.gen_geo_helper import read_windows, write_windows

def process_meep_arr(arr=None):
    if arr == None:
        arr = read_windows(
            data_dir + config.get('process_inp', 'project_name')+'.mpout')

    clean = get_ms(arr, 5)
    print(clean.shape)
    print(np.amax(clean))
    write_windows(clean, data_dir +
                  config.get('process_inp', 'project_name')+'.clean')


def get_area(eps_data):
    eps_rock = eps_data > 7.69
    area = np.sum(eps_rock)
    return area/np.prod(eps_data.shape)

def get_roi_res_block(eps):
    """ only take the center 3x3 area as region of interest """
    res_block = [-3, 3, -3, 3]
    translate = Translate(-cell_size[0]/2, cell_size[1]/2, 0, eps.shape[0])
    res_block_index = [int(translate(ele)) for ele in res_block]
    
    res_block_mat = np.zeros_like(eps)
    res_block_mat[res_block_index[0]:res_block_index[1], res_block_index[2]:res_block_index[3]] = 1
    return res_block_mat


def result_statistics(ez_data, eps):
    """
    return the mean, std and area informaiton about the result
    the area means the amount of area the rock take up
    """
    get_array('geo', 'cell_size', config)
    eps_const = config.getfloat('geo', 'eps')
    dim = config.getfloat('sim', 'dimension')
    res = config.getfloat('sim', 'resolution')
    sim_type = config.get('sim', 'sim_types') 

    eps_rock = eps >= eps_const
    res_block_mat = get_roi_res_block(eps)

    if sim_type == 'checker':
        eps_rock = np.logical_and(eps_rock, res_block_mat)

    elif sim_type == 'effective medium':
        # eps_rock = np.logical_and((eps > 7.69), res_block_mat)
        eps_rock = res_block_mat.astype(np.bool)

    ez_data[np.logical_not(eps_rock)] = 0

    mean = np.trapz(ez_data, axis=0, dx=1/res)
    mean = np.trapz(mean, axis=0, dx=1/res)
    if dim == 3:
        mean = np.trapz(mean, axis=0, dx=1/res)

    std = np.std(ez_data[eps_rock])

    if verbals:
        print('mean', mean, 'std', std)
        
    return mean, std

def get_ms(arr, rms_interval):
    """
    return the mean square value of the E field
    """

    rms_shape = np.array(arr.shape)
    rms_shape[0] = np.floor(rms_shape[0]/rms_interval)
    rms_arr = np.zeros(rms_shape.astype(np.int))
    for i in range(0, rms_shape[0]):
        slice = arr[i*rms_interval:(i+1)*rms_interval]
        slice = slice**2
        int_slice = romb(slice, dx=1, axis=0)/(rms_interval-1)*10e26
        # rms_arr[i, :, :, :] = np.sqrt(np.mean(slice, axis=0))
        rms_arr[i, :, :, :] = int_slice
    return rms_arr


def write_res(web_config, data, var_descrip_str):
    """
    write result to file or redis database base on the configuration
    """
    if not web_config:
        with pd.ExcelWriter(plot_f_name() + '_' + var_descrip_str + '.xlsx') as writer:  
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

        r = redis.Redis(port=6793, host='localhost')
        r.set('Current result', output.read())
        output.close()
