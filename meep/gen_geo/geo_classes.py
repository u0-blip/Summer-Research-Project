import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
from numpy import cos, sin
import json
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from time import time
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

import os
if os.name == 'posix':
    import meep as mp

from gen_geo import bounded_voronoi
from gen_geo import convex_hull

if os.name == 'posix': 
    mat1 = mp.Medium(epsilon=7.69, D_conductivity=2*math.pi*0.42*2.787/3.4)
    mat2 = mp.Medium(epsilon=7.1)

class voronoi_geo:
    num_mat = 2
    if os.name == 'posix': eps_mat = [mat1, mat2]

    def __init__(self, vor, box):
        self.num_seeds = len(vor.regions)
        self.vor = vor
        self.points = np.array(vor.og_points)
        self.vertices = self.vor.vertices
        self.random_ass()
        self.bounding_box = box

    def random_ass(self):
        self.parts_ass = np.random.randint(0, self.num_mat, (self.num_seeds))
        if os.name == 'posix': 
            self.parts_eps = [self.eps_mat[self.parts_ass[i]] for i in range(self.num_seeds)]
        else:
            self.parts_eps = None
    
    def inbox(self, coord):
        b_box = self.bounding_box
        x_in = np.logical_and(b_box[0, 0] <= coord[0], coord[0] <= b_box[0, 1])
        y_in = np.logical_and(b_box[1, 0] <= coord[1], coord[1] <= b_box[1, 1])
        z_in = np.logical_and(b_box[2, 0] <= coord[2], coord[2] <= b_box[2, 1])
        return np.logical_and(np.logical_and(x_in, y_in), z_in)

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
