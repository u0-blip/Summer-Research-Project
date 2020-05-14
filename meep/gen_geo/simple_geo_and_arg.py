import numpy as np
import argparse

import os
if os.name == 'posix':
    import meep as mp
from numpy import pi

import sys
sys.path.append("..")

from configs import config

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


num_particles = config.getint('geo', 'num_particles')


def get_coord(dist):
    if num_particles == 1:
        return [mp.Vector3(coordsconfig.getfloat('geo','dist_to_source'), 0, 0)]
    else:
        return [mp.Vector3(0., dist/2., 0.),  mp.Vector3(0., -dist/2,  0.)]


def get_polygon_coord(center):
    radius = config.getfloat('geo','particle_radius')
    a = np.sqrt(3)/2*radius
    shape = config.get('geo', 'shape')
    if shape == 'hexagon':
        points = np.array([
            [-radius/2, a, 0], 
            [radius/2, a, 0],
            [ radius, 0, 0],
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
        points[i] = rotate([0, 0, 0], p, eval(config.get('geo', 'rotation')))

    og_points = points

    vertices = []
    for i in range(len(center)):
        p = og_points + center[i]
        vertices.append([])
        for pp in p:
            vertices[i].append(mp.Vector3(pp[0], pp[1], pp[2]))

    return vertices

