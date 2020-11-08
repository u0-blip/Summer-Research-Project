from my_meep.config.configs import get_array
from my_meep.config.config_variables import *
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.spatial
import sys
import pickle
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from gen_voronoi.convex_hull import *
from gen_voronoi.geo_classes import voronoi_geo
import redis
r = redis.Redis(port=6379, host='localhost', db=0)
sys.path.append('..')

class Gen_vor():
    def __init__(self, config):
        self.eps = sys.float_info.epsilon
        self.config =  config
        self.vor_size = get_array('Geometry', 'solid_size', self.config)
        self.vor_center = get_array('Geometry', 'solid_center', self.config)
        self.bounding_box = np.expand_dims(self.vor_size, 1)
        self.bounding_box = np.concatenate((-self.bounding_box/2, self.bounding_box/2), axis=1)
        self.bounding_box = (self.bounding_box.transpose() + self.vor_center).transpose()

    def in_box(self, seed_points):
        x_in = np.logical_and(self.bounding_box[0, 0] <= seed_points[:, 0], seed_points[:, 0] <= self.bounding_box[0, 1])
        y_in = np.logical_and(self.bounding_box[1, 0] <= seed_points[:, 1], seed_points[:, 1] <= self.bounding_box[1, 1])
        z_in = np.logical_and(self.bounding_box[2, 0] <= seed_points[:, 2], seed_points[:, 2] <= self.bounding_box[2, 1])

        return np.logical_and(np.logical_and(x_in, y_in), z_in)

    def maxDist(self, points):
        dist = 0
        for i in range(len(points)):
            for j in range(i):
                if dist < np.linalg.norm(points[i] - points[j]):
                    dist  = np.linalg.norm(points[i] - points[j])
        return dist

    def generateBoundedVor(self, seed_points):
        # Select seed_points inside the bounding box
        #i = in_box(seed_points, self.bounding_box)
        # Mirror points
        # points_center = seed_points[i, :]

        points_center = seed_points
        points_left = np.copy(points_center)
        points_right = np.copy(points_center)
        points_down = np.copy(points_center)
        points_up = np.copy(points_center)
        points_front = np.copy(points_center)
        points_back = np.copy(points_center)
        
        points_left[:, 0] = self.bounding_box[0, 0] - (points_left[:, 0] - self.bounding_box[0, 0])
        points_right[:, 0] = self.bounding_box[0, 1] + (self.bounding_box[0, 1] - points_right[:, 0])
        points_down[:, 1] = self.bounding_box[1, 0] - (points_down[:, 1] - self.bounding_box[1, 0])
        points_up[:, 1] = self.bounding_box[1, 1] + (self.bounding_box[1, 1] - points_up[:, 1])
        points_back[:, 2] = self.bounding_box[2, 0] - (points_back[:, 2] - self.bounding_box[2, 0])
        points_front[:,2] = self.bounding_box[2, 1] + (self.bounding_box[2, 1] - points_front[:, 2])

        points = np.copy(points_center)
        points = np.append(points, np.append(points_left, points_right, axis=0), axis=0)
        points = np.append(points, np.append(points_down, points_up, axis=0), axis=0)
        points = np.append(points, np.append(points_back, points_front, axis=0), axis=0)

        # Compute Voronoi
        vor = sp.spatial.Voronoi(points)

        # Filter regions
        regions = []
        for region in vor.regions:
            flag = True
            for index in region:
                if index == -1:
                    flag = False
                    break
                else:
                    x = vor.vertices[index, 0]
                    y = vor.vertices[index, 1]
                    z = vor.vertices[index, 2]
                    if not(self.bounding_box[0, 0] - 100*self.eps <= x and x <= self.bounding_box[0, 1] + 100*self.eps and
                        self.bounding_box[1, 0] - 100*self.eps <= y and y <= self.bounding_box[1, 1] + 100*self.eps and 
                        self.bounding_box[2, 0] - 100*self.eps <= z and z <= self.bounding_box[2, 1] + 100*self.eps):
                        flag = False
                        break
            if region != [] and flag:
                regions.append(region)


        vor.og_points = points_center
        vor.regions = regions
        return vor


    def del_bad_poly_ratio(self, vor, hull):
        seed_of_regions_to_keep = []
        for i, region in enumerate(vor.regions):
            dist = self.maxDist(vor.vertices[region, :])
            ratio = dist/(hull[i].volume**(1/3.))
            # print(ratio)
            if ratio <= 2.2:
                seed_of_regions_to_keep.append(vor.og_points[i])

        # print('points is ')
        # print(len(seed_of_regions_to_keep))
        return seed_of_regions_to_keep


    def del_points_too_close(self, vor):
        # merge points doesn't actually work because the geometry will have empty regions
        merge_points = []

        # for i_region in range(len(vor.regions)):
        for i_region in range(len(vor.regions)):
            region = vor.regions[i_region]
            for i in range(len(region)):
                for j in range(i):
                    dist = np.linalg.norm(vor.vertices[region[i]] - vor.vertices[region[j]])
                    if dist < 5*10e-5:
                        in_sets = False
                        for sets in merge_points:
                            if region[i] in sets or region[j] in sets:
                                sets.add(region[i])
                                sets.add(region[j])
                                in_sets = True
                                break
                        if not in_sets:
                            merge_points.append({region[i], region[j]})

        at_boundary = lambda point: (
            point[0] < self.bounding_box[0, 0] + sys.float_info.epsilon 
        or point[0] > self.bounding_box[0, 1] - sys.float_info.epsilon 
        or point[1] < self.bounding_box[1, 0] + sys.float_info.epsilon 
        or point[1] > self.bounding_box[1, 1] - sys.float_info.epsilon
        or point[2] < self.bounding_box[2, 0] + sys.float_info.epsilon
        or point[2] > self.bounding_box[2, 1] - sys.float_info.epsilon
        )

        merge_to_points = []

        for point_set in merge_points:
            point_list = list(point_set)
            breaked = False
            for i, index_point in enumerate(point_list):
                point = vor.vertices[index_point]
                if at_boundary(point):
                    merge_to_points.append(point_list.pop(i))
                    breaked = True
                    break
            if not breaked:
                merge_to_points.append(point_list.pop(0))
                # merge_to_points = [sets.pop() for sets in merge_points]
        # print(merge_to_points)

        for i, pair in enumerate(merge_points):
            for p in pair:
                vor.vertices[p] = vor.vertices[merge_to_points[i]]
                # print(vor.vertices[p])
        # print(merge_points)

        # indexToDel = np.sort(np.array(list(indexToDel)))

        # for region in vor.regions:
        #     for i_region_point in range(len(region)):
        #         for i_points_set, points_set in enumerate(merge_points):
        #             if region[i_region_point] in points_set:
        #                 # print('sub ' + str(region[i_region_point]) + ' to ' + str(merge_to_points[i_points_set]))
        #                 region[i_region_point] = merge_to_points[i_points_set]
        #                 break
        return vor

    def del_polygon_too_narrow(self, vor):

        # vor = generateBoundedVor(self.seed_points) 

        hull_seed_points = []
        hull = []

        hull_seed_points = [vor.vertices[region] for region in vor.regions]
        hull, faces = get_conv_hull(hull_seed_points)

        self.seed_points = self.del_bad_poly_ratio(vor, hull)

        vor = self.generateBoundedVor(self.seed_points) 

        return vor

    def plot_vor(self, vor_vertices, regions):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot vertices
        for region in regions:
            vertices = vor_vertices[region, :]
            ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], 'go')

        ax.plot(self.seed_points[:,0], self.seed_points[:,1], self.seed_points[:,2], 'yo')

        plt.show()


    def b_voronoi(self, to_out_geo = True):

        n_towers = self.config.getint('Geometry', 'num_particles_vor')
        np.random.seed(self.config.getint('Geometry', 'rand_seed'))
        sim_dim = self.config.getint('Simulation', 'dimension')
        
        self.seed_points = np.random.rand(n_towers, 3) - 0.5
        if sim_dim == 2:
            self.seed_points[:, 2] = 0
        # print(self.seed_points)

        for i in range(3):
            self.seed_points[:, i] *= self.vor_size[i] 
            self.seed_points[:, i] += self.vor_center[i]

        vor = self.generateBoundedVor(self.seed_points) 

        # vor = self.del_polygon_too_narrow(vor)

        # vor = self.del_points_too_close(vor)

        hull_seed_points = [vor.vertices[region] for region in vor.regions]

        hull, faces = get_conv_hull(hull_seed_points)

        # finishing up and writing points to the file
        for i in range(len(hull_seed_points)):
            hull_seed_points[i] = hull_seed_points[i].tolist()

        unique_edge_list, face_index_list = del_useless_edges(vor, hull)

        geo = [hull_seed_points, unique_edge_list, face_index_list]

        complete_vor = voronoi_geo(vor=vor, box = self.bounding_box, config=self.config)
        
        if to_out_geo: 
            r.set('vor_geo', pickle.dumps(geo))
            r.set('vor_vor', pickle.dumps([vor, complete_vor]))
            r.set('vor_partass', pickle.dumps(complete_vor.parts_ass))
        
        print('created ' + str(len(vor.regions)) + ' polygons')
        return vor, complete_vor, geo
