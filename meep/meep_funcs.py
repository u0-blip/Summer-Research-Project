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


size_crystal_base = [0.1, 0.1, 0.1]
num_crystal = 200
air = mp.Medium(epsilon=1.0)

def closest_node(node, nodes):
    # nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)

    
def my_mat(coord):
    if coord[0] <  0.25 and coord[0] > -0.25 :
        return mp.Medium(epsilon=10.5)
    else:
        return mp.Medium(epsilon=1)

def pass_vor(func, vor):
    def wrapper(*args, **kwargs):
        return func(vor, *args, **kwargs)
    return wrapper

def my_eps(my_vor, coord):
    acoord = np.abs(coord)
    if (acoord[0] < 0.5 and acoord[1] < 0.5 and acoord[2] < 0.5 ):
        return my_vor.parts_eps[closest_node([coord[0],coord[1],coord[2]], my_vor.points)]
    else:
        return air

def my_ass(my_vor, coord):
    acoord = np.abs(coord)
    if (acoord[0] < 0.5 and acoord[1] < 0.5 and acoord[2] < 0.5 ):
        return my_vor.parts_ass[closest_node([coord[0],coord[1],coord[2]], my_vor.points)]
    else:
        return -1

def index2coord(index, size_arr, size_geo):
    index = (index/size_arr - 0.5)*size_geo
    return index

def vis(sim):
    sim.init_sim()
    eps_data = sim.get_epsilon()

    from mayavi import mlab
    s = mlab.contour3d(eps_data, colormap="YlGnBu")
    mlab.show()

def write_windows(arr, file_name):
    if type(arr) is not np.ndarray:
        arr = np.array(arr)
    with open(file_name, 'wb') as f:
        arr.transpose().astype('<f8').tofile(f)
    with open(file_name + '.meta', 'wb') as f:
        np.array(arr.shape).transpose().astype('<f8').tofile(f)

def read_windows(file_name, shape = None):
    if shape == None:
        shape = np.flip(np.fromfile(file_name + '.meta', np.float).astype(np.int), axis=0)
    return np.fromfile(file_name, np.float).reshape(shape).transpose()

def gen_part_size(num_crystal, size_crystal_base, weibull = True):
    a = 5. # shape of weibull distribution
    size_crystal_change = np.random.weibull(a, (num_crystal, 3))
    
    if weibull:
        size_crystal_base *= size_crystal_change
            
    size = []
    for i in range(num_crystal):
        size.append(mp.Vector3(*size_crystal[i, :]))
        
    return size

def gen_part_loc(num_crystal, size_solid = None, use_normal = False):
    if(size_solid == None):
        size_solid = [1, 1, 1]
    if use_normal:
        mean= (0, 0, 0)
        cov = [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]
        loc = np.random.multivariate_normal(mean, cov, (num_crystal))
    else:
        loc = np.random.uniform(-size_solid[0]/2, size_solid[0]/2, (num_crystal, 3))
    return loc

def gen_part_rot(num_crystal):
    theta = np.empty(num_crystal)
    for i in range(3):
        theta[i] = np.random.uniform(0, 2*np.pi, num_crystal)
    return theta

def gen_particle_geo(loc, theta_x, theta_y, theta_z):
    R = np.empty((num_crystal, 3, 3))

    Rx_matrix = np.empty((num_crystal, 3, 3))
    Ry_matrix = np.empty((num_crystal, 3, 3))
    Rz_matrix = np.empty((num_crystal, 3, 3))

    for i in range(num_crystal):
        Rx_matrix[i, :, :] = np.array([[1, 0, 0],
                       [0, cos(theta_x[i]), -sin(theta_x[i])], 
                      [0, sin(theta_x[i]), cos(theta_x[i])]])

        Ry_matrix[i, :, :] = np.array([[cos(theta_y[i]), 0, sin(theta_y[i])], 
                      [0, 1, 0],
                      [-sin(theta_y[i]), 0, cos(theta_y[i])]])

        Rz_matrix[i, :, :] = np.array([[cos(theta_z[i]), -sin(theta_z[i]), 0],
                     [sin(theta_z[i]), cos(theta_z[i]), 0],
                     [0, 0, 1]])

        R[i, :, :] = np.matmul(np.matmul(Ry_matrix[i, :, :], Rx_matrix[i, :, :]), Rz_matrix[i, :, :])


    og_x = np.array([[1, 0, 0] for i in range(num_crystal)])
    og_y = np.array([[0, 1, 0] for i in range(num_crystal)])
    og_z = np.array([[0, 0, 1] for i in range(num_crystal)])

    Rx_vector = np.empty((num_crystal, 3))
    Ry_vector = np.empty((num_crystal, 3))
    Rz_vector = np.empty((num_crystal, 3))

    for i in range(num_crystal):
        Rx_vector[i, :] = np.matmul(R[i, :, :], og_x[i, :])
        Ry_vector[i, :] = np.matmul(R[i, :, :], og_y[i, :])
        Rz_vector[i, :] = np.matmul(R[i, :, :], og_z[i, :])

    geometry = [solid_region,]

    for i in range(num_crystal):
        if (np.abs(loc[i, 0]) < size_solid[0] - size_crystal_base[0]/2 and 
        np.abs(loc[i, 1]) < size_solid[1] - size_crystal_base[1]/2 and 
        np.abs(loc[i, 2]) < size_solid[2] - size_crystal_base[2]/2):
            geometry.append(mp.Block(
                size_crystal[i],
                center = mp.Vector3(loc[i, 0], loc[i, 1], loc[i, 2]),
                e1 = Rx_vector[i, :],
                e2 = Ry_vector[i, :],
                e3 = Rz_vector[i, :],
                material=mp.Medium(epsilon=10.5)))
    return geometry

def out_para_geo(file_name, num_crystal, size_solid_l, size_crystal_l, loc, theta):
    to_write = [num_crystal, size_solid_l, size_crystal_l, loc, theta]
    for i in range(len(to_write)):
        if type(to_write[i]) is not int and type(to_write[i]) is not list:
            to_write[i] = to_write[i].tolist()
    with open(file_name, 'w') as f:
        json.dump(to_write, f)

#\\ad.monash.edu\home\User045\dche145\Documents\Abaqus\geometry_shapes
#\\Client\D$\source\working_with_meep

#output geometric files
def out_num_geo(file_name, geo_data_obj, range_geo=None, range_index = None):
    if range_index == None:
        range_index = [100, 100, 100]
    if range_geo == None:
        range_geo = [1.0, 1.0, 1.0]
    out_geo = np.zeros((range_index))
    for i in range(range_index[0]):
        for j in range(range_index[1]):
            for k in range(range_index[2]):
                coord = index2coord(np.array((i,j,k),dtype=float), range_index, range_geo)
                out_geo[i,j,k] = geo_data_obj.parts_eps[closest_node(coord, geo_data_obj.points)]
    out_geo = out_geo.transpose().astype('<f4')
    with open(file_name, 'wb') as f:
        out_geo.tofile(f)
    print('file ' + file_name+' shape: ')
    print(out_geo.shape)

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def write_geo_for_field(vertices):
    raise Exception('unimplemented')



def index2coord(index, size_arr, size_geo):
    index = (index/size_arr - 0.5)*size_geo
    return index




def create_simple_geo(geometry, coords, shape, size_solid, prism_height=0, prism_axis=(0,0,1)):
    if shape == "cube":
        for i,coord in enumerate(coords):
            geometry.append(
                mp.Block(
                    size_solid, 
                    center = coord,
                    material=mp.Medium(epsilon=7.69, D_conductivity=2*math.pi*0.42*2.787/3.4)
                    )
            )
    elif shape == "sphere":
        for i,coord in enumerate(coords):
            geometry.append(
                mp.Sphere(
                    radius = size_solid, 
                    center = coord,
                    material=mp.Medium(epsilon=7.69, D_conductivity=2*math.pi*0.42*2.787/3.4)
                    )
            )
    if shape == "ellipsoid":
        for i,coord in enumerate(coords):
            geometry.append(
                mp.Ellipsoid(
                    size_solid, 
                    center = coord,
                    material=mp.Medium(epsilon=7.69, D_conductivity=2*math.pi*0.42*2.787/3.4)
                    )
            )
    if shape == "hexagon" or shape == 'triangle':
        for i, coord in enumerate(coords):
            geometry.append(
                mp.Prism(
                    vertices = coord,
                    height = prism_height, 
                    axis = prism_axis,
                    material=mp.Medium(epsilon=7.69, D_conductivity=2*math.pi*0.42*2.787/3.4)
                    )
            )
    return geometry