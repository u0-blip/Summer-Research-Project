import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.spatial
import sys
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

eps = sys.float_info.epsilon



def in_box(towers, bounding_box):
    x_in = np.logical_and(bounding_box[0] <= towers[:, 0], towers[:, 0] <= bounding_box[1])
    y_in = np.logical_and(bounding_box[2] <= towers[:, 1], towers[:, 1] <= bounding_box[3])
    z_in = np.logical_and(bounding_box[4] <= towers[:, 2], towers[:, 2] <= bounding_box[5])

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

    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points_back[:, 2] = bounding_box[4] - (points_back[:, 2] - bounding_box[4])
    points_front[:,2] = bounding_box[5] + (bounding_box[5] - points_front[:, 2])

    points = points_center
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
                if not(bounding_box[0] - 3*eps <= x and x <= bounding_box[1] + 3*eps and
                       bounding_box[2] - 3*eps <= y and y <= bounding_box[3] + 3*eps and 
                       bounding_box[4] - 3*eps <= z and z <= bounding_box[5] + 3*eps):
                    flag = False
                    break
        if region != [] and flag:
            regions.append(region)


    vor.filtered_points = points_center
    vor.regions = regions
    return points, vor

def delPointsBadRatio(points, vor):
    for region in vor.regions:
        dist = maxDist(vor.vertices[region, :])
    
def voronoi(towers, bounding_box):

    # while delPointsBadRatio(points, vor):
    #     points, vor = generateBoundedVor(points, bounding_box)

    points, vor = generateBoundedVor(towers, bounding_box) 

    indexToDel = set()
    dontAdd  = set()
    for i in range(len(vor.vertices)):
        for j in range(i):
            dist = np.linalg.norm(vor.vertices[i] - vor.vertices[j])
            if dist < 0.05:
                if i not in dontAdd:
                    indexToDel.add(i)
                    dontAdd.add(j)
    indexToDel = np.sort(np.array(list(indexToDel)))
    print('points deleted ' + str(len(indexToDel)))

    for i in range(len(indexToDel)):
        
        index = indexToDel[len(indexToDel) - i -1]
        # vor.vertices  = np.delete(vor.vertices, index, 0)
        for i, r in enumerate(vor.regions):
            r = np.array(r)
            not_delete = np.logical_not(r == index)
            # print(r)
            # print(not_delete)
            vor.regions[i] = r[not_delete]
    
    return vor

def centroid_regionBackup(vertices):
    # Polygon's signed area
    A = 0
    # Centroid's x
    C_x = 0
    # Centroid's y
    C_y = 0
    Cz = 0
    for i in range(0, len(vertices) - 1):
        s = (vertices[i, 0] * vertices[i + 1, 1] - vertices[i + 1, 0] * vertices[i, 1])
        A = A + s
        C_x = C_x + (vertices[i, 0] + vertices[i + 1, 0]) * s
        C_y = C_y + (vertices[i, 1] + vertices[i + 1, 1]) * s
        Cz = Cz 
    A = 0.5 * A
    C_x = (1.0 / (6.0 * A)) * C_x
    C_y = (1.0 / (6.0 * A)) * C_y
    return np.array([[C_x, C_y]])

def centroid_region(verticesSinglePoly):
    verticesSinglePoly = np.array(verticesSinglePoly)
    centroid  = np.sum(verticesSinglePoly, axis=0)/verticesSinglePoly.shape[0]
    return centroid


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


if __name__ == "__main__":
    n_towers = 30
    np.random.seed(112)
    towers = np.random.rand(n_towers, 3)

    bounding_box = np.array([0., 1., 0., 1., 0., 1.]) # [x_min, x_max, y_min, y_max]
    vor = voronoi(towers, bounding_box)
