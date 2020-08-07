import cmath
import numpy as np
import pandas as pd
from time import time
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from gen_geo.gmsh_create import voronoi_create
from gen_geo.process_inp_file import processing, abaqusing
from gen_geo.bounded_voronoi import b_voronoi, bounding_box
from my_meep.config.configs import *
from process_res import process_meep_arr

def win_main():
    if config.getboolean('general', 'gen_gmsh'):
        voronoi_create(display=True, to_out_f=True, to_mesh=True, in_geo=None)

        if verbals:
            now = time()
            print('Create input file, time: ' + str(now - last))
            last = now

    if config.getboolean('general', 'process_inp'):
        if verbals:
            last = time()
        processing()
        if verbals:
            now = time()
            print('Process input file, time: ' + str(now - last))
            last = now

    if config.getboolean('general', 'sim_abq'):
        if verbals:
            before = time()
        abaqusing()
        if verbals:
            after = time()
            elapsed = after-before
            print('Time for Abaqusing is ' + str(elapsed))

    if config.getboolean('general', 'clean_array'):
        if verbals:
            before = time()
        process_meep_arr()
        if verbals:
            after = time()
            elapsed = after-before
            print('Time for clean array is ' + str(elapsed))
