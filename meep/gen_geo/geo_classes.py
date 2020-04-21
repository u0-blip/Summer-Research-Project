import meep as mp
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
from numpy import cos, sin
import json
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import time
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

from gen_geo import bounded_voronoi
from gen_geo import convex_hull
from gen_geo.geo_classes import *

mat1 = mp.Medium(epsilon=7.69, D_conductivity=2*math.pi*0.42*2.787/3.4)
mat2 = mp.Medium(epsilon=7.1)

class voronoi_geo:
    num_mat = 2
    eps_mat = [mat1, mat2]

    def __init__(self, num_seeds, size_cell = 1, vor = None):
        if vor == None:
            self.num_seeds = num_seeds
            self.points = (np.random.rand(self.num_seeds, 3) - 0.5)*size_cell
            self.vor = Voronoi(self.points)
        else:
            self.num_seeds = len(vor.points)
            self.vor = vor
            self.points = vor.points
        self.vertices = self.vor.vertices
        self.random_ass()

    def random_ass(self):
        self.parts_ass = np.random.randint(0, self.num_mat, (self.num_seeds))
        self.parts_eps = [self.eps_mat[self.parts_ass[i]] for i in range(self.num_seeds)]

class checker_geo:
    num_div = 10
    num_seed = num_div**3
    points = np.zeros((num_div**3, 3))

    p_range = np.array([1.0, 1.0, 1.0])
    
    num_parts = 2
    eps_val = [1, 10]

    def __init__(self):
        self.checker_pattern()

    def checker_pattern(self):
        for i in range(self.num_div):
            for j in range(self.num_div):
                for k in range(self.num_div):
                    index = np.array([i,j,k])
                    self.points[index, :] = index2coord(index, np.array([self.num_div, self.num_div, self.num_div]), self.p_range)
        # this will produce checker pattern
        self.parts_ass = np.array([1 if i%2 else 0 for i in range(self.num_seed)])
        self.parts_eps = [self.eps_val[self.parts_ass[i]] for i in range(self.num_seed)]
