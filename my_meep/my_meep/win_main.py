import cmath
import numpy as np
import pandas as pd
from time import time
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from gen_voronoi.gmsh_create import voronoi_create
from gen_voronoi.process_inp_file import processing, abaqusing
from gen_voronoi.bounded_voronoi import Gen_vor
from my_meep.config.configs import get_array
from my_meep.config.config_variables import *
from Result_manager import process_meep_arr

def log_time(info):
    pass
def win_main():
    if config.getboolean('General', 'gen_gmsh'):
        voronoi_create(display=True, to_out_f=True, to_mesh=True, in_geo=None)

        if verbals:
            now = time()
            print('Create input file, time: ' + str(now - last))
            last = now

    if config.getboolean('General', 'process_inp'):
        if verbals:
            last = time()
        processing()
        if verbals:
            now = time()
            print('Process input file, time: ' + str(now - last))
            last = now

    if config.getboolean('General', 'sim_abq'):
        if verbals:
            before = time()
        abaqusing()
        if verbals:
            after = time()
            elapsed = after-before
            print('Time for Abaqusing is ' + str(elapsed))

    if config.getboolean('General', 'clean_array'):
        if verbals:
            before = time()
        process_meep_arr()
        if verbals:
            after = time()
            elapsed = after-before
            print('Time for clean array is ' + str(elapsed))
