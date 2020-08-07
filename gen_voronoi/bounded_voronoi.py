from my_meep.config.configs import *
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.spatial
import sys
import pickle
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from .convex_hull import *
from .geo_classes import voronoi_geo

sys.path.append('..')

eps = sys.float_info.epsilon

bounding_box = np.expand_dims(vor_size, 1)
bounding_box = np.concatenate((-bounding_box/2, bounding_box/2), axis=1)
bounding_box = (bounding_box.transpose() + vor_center).transpose()

def in_box(towers, bounding_box):
    x_in = np.logical_and(bounding_box[0, 0] <= towers[:, 0], towers[:, 0] <= bounding_box[0, 1])
    y_in = np.logical_and(bounding_box[1, 0] <= towers[:, 1], towers[:, 1] <= bounding_box[1, 1])
    z_in = np.logical_and(bounding_box[2, 0] <= towers[:, 2], towers[:, 2] <= bounding_box[2, 1])

    return np.logical_and(np.logical_and(x_in, y_in), z_in)

def maxDist(points):
    dist = 0
    for i in range(len(points)):
        for j in range(i):
            if dist < np.linalg.norm(points[i] - points[j]):
                dist  = np.linalg.norm(points[i] - points[j])
    return dist

def generateBoundedVor(towers, bounding_box):
    # Select towers inside the bounding box
    #i = in_box(towers, bounding_box)
    # Mirror points
    # points_center = towers[i, :]

    points_center = towers
    points_left = np.copy(points_center)
    points_right = np.copy(points_center)
    points_down = np.copy(points_center)
    points_up = np.copy(points_center)
    points_front = np.copy(points_center)
    points_back = np.copy(points_center)

    points_left[:, 0] = bounding_box[0, 0] - (points_left[:, 0] - bounding_box[0, 0])
    points_right[:, 0] = bounding_box[0, 1] + (bounding_box[0, 1] - points_right[:, 0])
    points_down[:, 1] = bounding_box[1, 0] - (points_down[:, 1] - bounding_box[1, 0])
    points_up[:, 1] = bounding_box[1, 1] + (bounding_box[1, 1] - points_up[:, 1])
    points_back[:, 2] = bounding_box[2, 0] - (points_back[:, 2] - bounding_box[2, 0])
    points_front[:,2] = bounding_box[2, 1] + (bounding_box[2, 1] - points_front[:, 2])

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
                if not(bounding_box[0, 0] - 100*eps <= x and x <= bounding_box[0, 1] + 100*eps and
                       bounding_box[1, 0] - 100*eps <= y and y <= bounding_box[1, 1] + 100*eps and 
                       bounding_box[2, 0] - 100*eps <= z and z <= bounding_box[2, 1] + 100*eps):
                    flag = False
                    break
        if region != [] and flag:
            regions.append(region)


    vor.og_points = points_center
    vor.regions = regions
    return vor


def del_bad_poly_ratio(vor, hull):
    seed_of_regions_to_keep = []
    for i, region in enumerate(vor.regions):
        dist = maxDist(vor.vertices[region, :])
        ratio = dist/(hull[i].volume**(1/3.))
        # print(ratio)
        if ratio <= 2.2:
            seed_of_regions_to_keep.append(vor.og_points[i])

    # print('points is ')
    # print(len(seed_of_regions_to_keep))
    return seed_of_regions_to_keep


def del_points_too_close(vor):
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
        point[0] < bounding_box[0, 0] + sys.float_info.epsilon 
    or point[0] > bounding_box[0, 1] - sys.float_info.epsilon 
    or point[1] < bounding_box[1, 0] + sys.float_info.epsilon 
    or point[1] > bounding_box[1, 1] - sys.float_info.epsilon
    or point[2] < bounding_box[2, 0] + sys.float_info.epsilon
    or point[2] > bounding_box[2, 1] - sys.float_info.epsilon
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

def del_polygon_too_narrow(vor, bounding_box):

    # vor = generateBoundedVor(v_seed_points, bounding_box) 

    hull_seed_points = []
    hull = []

    hull_seed_points = [vor.vertices[region] for region in vor.regions]
    hull, faces = get_conv_hull(hull_seed_points)

    v_seed_points = del_bad_poly_ratio(vor, hull)

    vor = generateBoundedVor(v_seed_points, bounding_box) 

    return vor

def plot_vor(vor_vertices, regions):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot vertices
    for region in regions:
        vertices = vor_vertices[region, :]
        ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], 'go')

    # Compute and plot centroids
    # centroids = []
    # for region in vor.filtered_regions:
    #     vertices = vor.vertices[region + [region[0]], :]
    #     centroid = centroid_region(vertices)
    #     centroids.append(list(centroid[0, :]))
    #     ax.plot(centroid[:, 0], centroid[:, 1], 'r.')

    # Plot ridges
    # for region in regions:
    #     if region != [] and -1 not in region:
    #         vertices = vor_vertices[region + [region[0]], :]
    #         ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], 'k-')

    ax.plot(towers[:,0], towers[:,1], towers[:,2], 'yo')
    # print(vor.filtered_regions)

    plt.show()


def b_voronoi(to_out_geo = True):

    n_towers = config.getint('vor', 'num_particle')
    np.random.seed(config.getint('vor', 'rand_seed'))

    v_seed_points = np.random.rand(n_towers, 3) - 0.5

    for i in range(3):
        v_seed_points[:, i] *= vor_size[i] 
        v_seed_points[:, i] += vor_center[i]

    vor = generateBoundedVor(v_seed_points, bounding_box) 

    vor = del_polygon_too_narrow(vor, bounding_box)

    vor = del_points_too_close(vor)

    hull_seed_points = [vor.vertices[region] for region in vor.regions]

    hull, faces = get_conv_hull(hull_seed_points)

    # finishing up and writing points to the file
    for i in range(len(hull_seed_points)):
        hull_seed_points[i] = hull_seed_points[i].tolist()

    unique_edge_list, face_index_list = del_useless_edges(vor, hull)

    geo = [hull_seed_points, unique_edge_list, face_index_list]

    complete_vor = voronoi_geo(vor=vor, box = bounding_box)
    
    if to_out_geo: 
        with open(data_dir + project_name + '.geo', 'wb') as f:
            pickle.dump(geo, f)
        with open(data_dir + project_name + '.vor', 'wb') as f:
            pickle.dump([vor, complete_vor], f)
        with open(data_dir + project_name +'.partass', 'wb') as f:
            pickle.dump(complete_vor.parts_ass, f)
    
    print('created ' + str(len(vor.regions)) + ' polygons')
    return vor, complete_vor, geo

if __name__ == "__main__":
    n_towers = 30
    vor = b_voronoi(n_towers, 15)
    