import numpy as np
from my_meep.config.configs import get_array
from my_meep.config.config_variables import *

import os
if os.name == 'posix':
    import meep as mp
from numpy import pi

import sys
sys.path.append("..")

import math


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin[0:2]
    px, py = point[0:2]

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return np.array([qx, qy, point[2]])

def np2mp(arr):
    res = []
    for point in arr:
        res.append(mp.Vector3(*point))
    return res

def get_polygon_coord(center, radius, config):
    shape = config.get('Geometry', 'shape')
    a = np.sqrt(3)/2*radius
    if shape == 'hexagon':
        points = np.array([
            [-radius/2, a, 0], 
            [radius/2, a, 0],
            [radius, 0, 0],
            [radius/2, -a, 0],
            [-radius/2, -a, 0],
            [-radius, 0, 0]
            ])
    elif shape == 'triangle':
        points = np.array([
            [0, radius, 0],
            [a, -radius/2, 0],
            [-a, -radius/2, 0]
        ])


    for i, p in enumerate(points):
        points[i] = rotate([0, 0, 0], p, config.getfloat('Geometry', 'rotation')/180*np.pi)
    points += center
    return np2mp(points)

