import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import pickle

def centroid_region(verticesSinglePoly):
    verticesSinglePoly = np.array(verticesSinglePoly)
    centroid  = np.sum(verticesSinglePoly, axis=0)/verticesSinglePoly.shape[0]
    return centroid

def plot_hull(points, hull, plotIndex=None):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if type(points[0]) is not list and type(points[0]) is not np.ndarray:
        for s in hull.simplices:
            s = np.append(s, s[0])  # Here we cycle back to the first coordinate
            ax.plot(points[s, 0], points[s, 1], points[s, 2], "r-")

        ax.plot(points.T[0], points.T[1], points.T[2], "ko") 

    else:
        if(plotIndex == None):
            ran = range(len(points))
        else:
            ran = plotIndex
        for i in ran:
            points[i] = np.array(points[i])
            for s in hull[i].simplices:
                s = np.append(s, s[0])  # Here we cycle back to the first coordinate
                ax.plot(points[i][s, 0], points[i][s, 1], points[i][s, 2], "r-")

            ax.plot(points[i].T[0], points[i].T[1], points[i].T[2], "ko") 
            temp = centroid_region(points[i])
            ax.plot([temp[0]], [temp[1]], [temp[2]], 'go')

    

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def get_conv_hull(points, name_points = 'polygon1.csv', name_hull = 'polygon1-hull.csv'):
    # points = np.array([[0, 0, 0], [1, 1, 2] ,[1.5, 0.5, 1], [1.5, -0.5, 3], [1.25, 0.3, -1], [1, 0, 2], [1.25, -0.3, -1], [1, -1, 3]])
    
    if type(points[0]) is not list and type(points[0]) is not np.ndarray:
        hull = ConvexHull(points)
        faces = hull.simplices
    else:
        hull = []
        faces = []
        for i, polygon in enumerate(points):
            hull.append(ConvexHull(polygon))
            faces.append((hull[i].simplices + 1).tolist())
    # np.savetxt(name_points, points, delimiter=",")
    # np.savetxt(name_hull, faces+1, delimiter=",")

    for i in range(len(points)):
        points[i] = points[i].tolist()

    with open(name_points, 'wb') as f:
        pickle.dump(points, f, protocol=2)
    with open(name_hull, 'wb') as f:
        pickle.dump(faces, f, protocol=2)

    return hull

if __name__ == "__main__":
    points = np.random.rand(30, 2)   # 30 random points in 2-D
    hull = ConvexHull(points)

    import matplotlib.pyplot as plt
    plt.plot(points[:,0], points[:,1], 'o')
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

    plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
    plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
    plt.show()