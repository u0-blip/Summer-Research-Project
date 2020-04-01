import numpy as np

def get_coord(dist, num = 2):
    return [[dist/2., 0., 0.],  [-dist/2,  0., 0.]]

def get_prism_coord(dist, radius, num=2):
    center = np.array([[dist/2., 0., 0.], [-dist/2,  0., 0.]])
    a = np.sqrt(3)/2*radius
    prism_points = np.array([[-radius/2, a], [radius/2, a],[ radius,0],[-radius,0],[-radius/2, -a], [radius/2, -a]])
    prism_points = np.concatenate((prism_points, np.zeros((6,1))), 1)
    prism_points = np.expand_dims(prism_points, 0)
    prism_points = np.concatenate((prism_points + center[0], prism_points+center[1]), 0)
    
    vertices = [[] for i in range(num)]
    for i in range(num):
        for j in range(prism_points.shape[1]):
            vertices[i].append(mp.Vector3(*prism_points[i,j,:]))
    return vertices
