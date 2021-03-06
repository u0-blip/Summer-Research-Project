import math
import cmath
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import my_meep.gen_geo_helper as gen_geo_helper
from my_meep.config.configs import get_array
from my_meep.config.config_variables import *
import meep as mp

def gaussian_beam(sigma, k, x0):
    def _gaussian_beam(x):
        return cmath.exp(1j*2*math.pi*k.dot(x-x0)-(x-x0).dot(x-x0)/(2*sigma**2))
    return _gaussian_beam

def source_wrapper(config):
    s = get_array('Source', 'size', config)
    center = get_array('Source', 'center', config)
    dim = config.getfloat('Simulation', 'dimension')
    sim_dim = config.getint('Simulation', 'dimension')
    if sim_dim == 2:
        s[0] = 0

    size = gen_geo_helper.np2mp(s)
    
    center = gen_geo_helper.np2mp(center)
    if dim == 3:
        size.z = size.y

    mode = config.get('Source', 'mode')
    fwidth = config.getfloat('Source', 'fwidth')
    fcen = config.getfloat('Source', 'fcen')

    if not config.getboolean('Simulation', 'calc_flux'):
        pre_source = mp.ContinuousSource(
            fcen, fwidth=fwidth*fcen, is_integrated=True)
    else:
        pre_source = mp.GaussianSource(fcen, fwidth=fwidth*fcen)
    if mode == 'normal':
        return [
            mp.Source(src=pre_source,
                      component=mp.Ez,
                      center=center,
                      size=size)
        ]

    elif mode == 'gaussian':
        tilt_angle = math.radians(config.getfloat(
            'Source', 'tilt_angle'))  # angle of tilted beam
        k = mp.Vector3(x=2).rotate(mp.Vector3(z=1), tilt_angle).scale(fcen)
        sigma = config.getfloat('Source', 'sigma')  # beam width
        # src_pt = mp.Vector3(y=4) # if you change center, you have to change phase and weird shit like that

        return [
            mp.Source(src=pre_source,
                      component=mp.Ez,
                      center=center,
                      size=size,
                      amp_func=gaussian_beam(sigma, k, center))
        ]
    else:
        return None
