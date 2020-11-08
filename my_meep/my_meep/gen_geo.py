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
    def cut(cell_size, center, Geometry, eps=None, mat=None):
        if mat is None:
            mat = mp.Medium(epsilon=eps)
        for i in range(len(center)):
            Geometry.append(
                mp.Block(
                    cell_size,
                    center=gen_geo_helper.np2mp(center[i]),
                    material=mat
                )
            )

    def apply_matellic(self):
        """ adding matellic boundary to the geometry """
        s = self.cell_size-0.5
        if self.sim_dim == 2:
            s[2] = 0

        self.Geometry.append(mp.Block(self.cell_size, material=mp.metal))
        self.Geometry.append(mp.Block(s, material=mp.air))


    def __init__(self, vor, config):
        self.particle_size = config.getfloat('Geometry', 'particle_area_or_volume')
        print('particle size', self.particle_size)
        self.shape = config.get('Geometry', 'shape')
        self.ff = config.getfloat('Geometry', 'fill_factor')
        self.fcen = config.getfloat('Source', 'fcen')
        self.solid_center = get_array( 'Geometry', 'solid_center', config)
        self.solid_size = get_array('Geometry', 'solid_size', config)
        self.cell_size = get_array('Geometry', 'cell_size', config)
        self.sim_dim = config.getint('Simulation', 'dimension')
        self.radius = gen_geo_helper.get_rad(self.particle_size, config)
        self.sim_type = config.get('Simulation', 'sim_types')
        self.config = config

        self.eps_value = get_array('Geometry', 'eps', self.config)
        self.eps_real = []
        self.eps_imag = []
        self.eps_complex = []
        self.eps_background = self.eps_value[0]
        for i, ele in enumerate(self.eps_value):
            if i % 2 == 0: 
                self.eps_real.append(ele)
                self.eps_complex.append(complex(self.eps_value[i], self.eps_value[i+1]))
            else: self.eps_imag.append(ele)
        self.section_assign = get_array('Geometry', 'section', self.config, type=np.int)

        Geometry = []

        if config.getboolean('boundary', 'metallic'):
            self.apply_matellic()

        if self.sim_type == 'checker':
            # this is to cut the gap between the Source and shape
            cut_gap = mp.Vector3(-2, 0, 0)
            eps = self.eps_real[self.section_assign[0]]
            eps_i = 2*math.pi*self.fcen*self.eps_imag[self.section_assign[0]]/eps
            matrix_mat = mp.Medium(epsilon=eps, D_conductivity=eps_i)
            self.cut(self.cell_size, [cut_gap], Geometry, mat=matrix_mat)
            coords = gen_geo_helper.gen_checker(config)
            self.gen_geo(Geometry, coords)
        elif self.sim_type == 'shape':
            coords_arr = get_array('Geometry', 'particle_location', self.config)
            num_particles = self.config.getint('Geometry', 'num_particles')
            coords = []
            for i in range(num_particles):
                coords.append(mp.Vector3(coords_arr[i*3], coords_arr[i*3+1], coords_arr[i*3+2]))
            self.gen_geo(Geometry, coords)

        elif self.sim_type == 'effective medium':
            self.gen_effective_medium_geo(Geometry)
        elif self.sim_type == 'voronoi':
            Geometry = vor
        if self.sim_type in ['checker', 'effective medium']:
            self.cut_surrounding(Geometry)
        self.Geometry = Geometry

    def __call__(self):
        return self.Geometry

    def cut_surrounding(self, Geometry):
        centerX = np.array([
                [1, 0, 0],
                [-1, 0, 0],
            ])
        centerY = np.array([
                [0, 1, 0],
                [0, -1, 0],
            ])
        centerZ = np.array([
                [0, 0, 1],
                [0, 0, -1]
            ])
        centerX = (self.cell_size[0]/2 + self.solid_size[0]/2)*centerX + self.solid_center
        centerY = (self.cell_size[1]/2 + self.solid_size[1]/2)*centerY + self.solid_center
        centerZ = (self.cell_size[2]/2 + self.solid_size[2]/2)*centerZ + self.solid_center
        
        self.cut(self.cell_size, centerX, Geometry, eps=self.eps_background)
        self.cut(self.cell_size, centerY, Geometry, eps=self.eps_background)
        self.cut(self.cell_size, centerZ, Geometry, eps=self.eps_background)

    def gen_effective_medium_geo(self, Geometry):
        eps_matrix = self.eps_complex[self.section_assign[0]]
        eps_rock = self.eps_complex[self.section_assign[1]]
        A = 2
        B = (1-3*self.ff)*eps_rock-(2-3*self.ff)*eps_matrix
        C = -eps_matrix*eps_rock
        res = (-B+np.sqrt(B**2-4*A*C))/(2*A)
        eps = res.real
        eps_i = 2*math.pi*self.fcen*res.imag/eps

        mat = mp.Medium(epsilon=eps, D_conductivity=eps_i)

        self.cut(mp.Vector3(8, 8, 0), [[-1, 0, 0]], Geometry, mat = mat)


    def gen_geo(self, geometry, coords):
        particle_size_t = self.config.get('Geometry','particle_size_t')
        sim_type = self.config.get('Simulation', 'sim_types')
        particle_size = get_array('Geometry', 'particle_area_or_volume', self.config)
        radius = particle_size[0]/2
        if particle_size_t == 'gaussian' and sim_type=='checker':
            radius = gen_geo_helper.gaussian_size(len(coords), self.config)
            radius = [gen_geo_helper.get_rad(ele, self.config) for ele in radius]
        else:
            radius = [radius]*len(coords)

        if sim_type =='checker':
            num_mat = len(self.section_assign)
            print('section assign', self.section_assign)
            parts_ass = np.random.randint(1, num_mat, (len(coords)))

        for i, coord in enumerate(coords):
            if sim_type =='checker':
                assign = parts_ass[i]
            elif sim_type == 'shape':
                assign = i

            eps = self.eps_real[self.section_assign[assign]]
            eps_i = 2*math.pi*self.fcen*self.eps_imag[self.section_assign[assign]]/eps

            mat = mp.Medium(epsilon=eps, D_conductivity=eps_i)
            if self.shape == "cube":
                geometry.append(
                    mp.Block(
                        [radius[i]]*3, 
                        center = coord,
                        material=mat
                        )
                )
            elif self.shape == "sphere":
                geometry.append(
                    mp.Sphere(
                        radius = radius[i], 
                        center = coord,
                        material=mat
                        )
                )
            elif self.shape == "ellipsoid":
                geometry.append(
                    mp.Ellipsoid(
                        radius[i], 
                        center = coord,
                        material=mat
                        )
                )
            elif self.shape == "hexagon" or self.shape == 'triangle':
                p_coords = gen_geo_helper.get_polygon_coord(coord, radius[i], self.config)
                geometry.append(
                    mp.Prism(
                        vertices = p_coords,
                        height = 1, 
                        axis = mp.Vector3(0,0,1),
                        material=mat
                        )
                )
        
        return geometry
