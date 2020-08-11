import math
import cmath
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import my_meep.gen_geo_helper as gen_geo_helper
import meep as mp
from my_meep.config.configs import get_array
from my_meep.config.config_variables import *
from my_meep.Sim_manager import Sim_manager

def index2coord(index, size_arr, size_geo):
    index = (index/size_arr - 0.5)*size_geo
    return index

class Gen_geo:
    """ a class that dealt with generating different geometric shape """
    
    @staticmethod
    def cut(cell_size, center, geo, eps=None, mat=None):
        if eps is not None:
            mat = mp.Medium(epsilon=eps)
        for i in range(len(center)):
            geo.append(
                mp.Block(
                    cell_size,
                    center=gen_geo_helper.np2mp(center[i]),
                    material=mat
                )
            )

    def apply_matellic(self):
        """ adding matellic boundary to the geometry """
        s = cell_size-0.5
        if sim_dim == 2:
            s[2] = 0

        self.geo.append(mp.Block(cell_size, material=mp.metal))
        self.geo.append(mp.Block(s, material=mp.air))


    def __init__(self, vor, config):
        self.particle_size = config.getfloat('geo', 'particle_size')
        self.shape = config.get('geo', 'shape')
        self.ff = config.getfloat('geo', 'fill_factor')
        self.eps = config.getfloat('geo', 'eps')
        self.eps_i = config.getfloat('geo', 'eps_i')
        self.wcen = config.getfloat('source', 'fcen')*0.34753
        self.config = config

        self.radius = gen_geo_helper.get_rad(self.particle_size, config)

        geo = []
        if sim_dim == 2:
            cell_size[2] = 0

        if config.getboolean('boundary', 'metallic'):
            self.apply_matellic()

        if shape_type == 'checker':
            # this is to cut the gap between the source and shape
            cut_gap = mp.Vector3(-2, 0, 0)
            self.cut(cell_size, [cut_gap], geo, eps=1)
            coords = gen_geo_helper.gen_checker(config)
            self.gen_geo(geo, coords)
        elif shape_type == 'shape':
            coords = gen_geo_helper.get_coord(self.radius, config)
            self.gen_geo(geo, coords)
        elif shape_type == 'effective medium':
            self.gen_effective_medium_geo(geo)
        else:
            geo = vor
        self.cut_surrounding(geo)
        self.geo = geo

    def __call__(self):
        return self.geo

    def cut_surrounding(self, geo):
        center = np.array([
                [9, 0, 0],
                [0, 9, 0],
                [-9, 0, 0],
                [0, -9, 0],
                [0, 0, 9],
                [0, 0, -9]
            ])

        center[0, 0] = 8
        self.cut(cell_size, center, geo, eps=1)

    def gen_effective_medium_geo(self, geo):

        eps_olivate = complex(7.27, 0.0685)
        eps_A = complex(1, 0)
        A = 2
        B = (1-3*self.ff)*eps_olivate-(2-3*self.ff)*eps_A
        C = -eps_A*eps_olivate
        res = (-B+np.sqrt(B**2-4*A*C))/(2*A)
        eps = res.real
        eps_i = res.imag

        mat = mp.Medium(epsilon=eps, D_conductivity=2*math.pi*self.wcen*eps_i/eps)

        self.cut(mp.Vector3(8, 8, 0), [[-1, 0, 0]], geo, mat = mat)


    def gen_geo(self, geometry, coords):
        if particle_size_t == 'gaussian' and shape_type=='checker':
            radius = gen_geo_helper.gaussian_size(len(coords), self.config)
            radius = [gen_geo_helper.get_rad(ele, self.config) for ele in radius]
        else:
            radius = [radius]*len(coords)


        if self.shape == "cube":
            for i, coord in enumerate(coords):
                geometry.append(
                    mp.Block(
                        [radius[i]]*3, 
                        center = coord,
                        material=mp.Medium(epsilon=7.69, D_conductivity=2*math.pi*0.42*2.787/3.4)
                        )
                )
        elif self.shape == "sphere":
            for i, coord in enumerate(coords):
                geometry.append(
                    mp.Sphere(
                        radius = radius[i], 
                        center = coord,
                        material=mp.Medium(epsilon=7.69, D_conductivity=2*math.pi*0.42*2.787/3.4)
                        )
                )
        elif self.shape == "ellipsoid":
            for i, coord in enumerate(coords):
                geometry.append(
                    mp.Ellipsoid(
                        radius[i], 
                        center = coord,
                        material=mp.Medium(epsilon=7.69, D_conductivity=2*math.pi*0.42*2.787/3.4)
                        )
                )
        elif self.shape == "hexagon" or self.shape == 'triangle':
            for i, coord in enumerate(coords):
                p_coords = gen_geo_helper.get_polygon_coord(coord, radius[i], self.config)
                geometry.append(
                    mp.Prism(
                        vertices = p_coords,
                        height = 1, 
                        axis = mp.Vector3(0,0,1),
                        material=mp.Medium(epsilon=7.69, D_conductivity=2*math.pi*0.42*2.787/3.4)
                        )
                )
        
        return geometry
