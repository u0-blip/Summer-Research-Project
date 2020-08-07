import cmath
import numpy as np
import pandas as pd
from time import time
import pickle
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_vor(config):
    """ get the voronoi geometry file """
    if config.get('sim', 'type') == 'voronoi':
        if config.getboolean('general', 'gen_vor'):
            _, complete_vor, geo = b_voronoi(to_out_geo=True)
        else:
            with open(config.get('process_inp', 'posix_data') + config.get('process_inp', 'project_name') + '.vor', 'rb') as f:
                _, complete_vor = pickle.load(f)

        vor = complete_vor

        if verbals:
            now = time()
            print('Created voronoi geo, time: ' + str(now - last))
    else:
        vor = None
    return vor
