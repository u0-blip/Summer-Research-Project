import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import pickle
import sys
import itertools

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


def get_conv_hull(points):
    # points = np.array([[0, 0, 0], [1, 1, 2] ,[1.5, 0.5, 1], [1.5, -0.5, 3], [1.25, 0.3, -1], [1, 0, 2], [1.25, -0.3, -1], [1, -1, 3]])
    
    if type(points[0]) is not list and type(points[0]) is not np.ndarray:
        hull = ConvexHull(points)
        faces = hull.simplices+1
    else:
        hull = []
        faces = []
        for i in range(len(points)):
            hull.append(ConvexHull(points[i]))
            faces.append((hull[i].simplices + 1).tolist())
    # np.savetxt(name_points, points, delimiter=",")
    # np.savetxt(name_hull, faces+1, delimiter=",")

    return hull, faces

def get_edges(shape_vertices):
    index = 0
    while index < len(shape_vertices):
        if index < len(shape_vertices) - 1:
            yield np.sort([shape_vertices[index], shape_vertices[index+1]])
        elif index == len(shape_vertices) - 1:
            yield np.sort([shape_vertices[-1], shape_vertices[0]])
        index += 1


def boarder_shape(s1, s2):
    for i1, e1 in enumerate(get_edges(s1)):
        for i2, e2 in enumerate(get_edges(s2)):
            if (np.sort(e1) == np.sort(e2)).all():
                # if (e1 == e2).all():
                #     inversed = False
                # else:
                #     inversed = True
                return [i1, i2]
    return [-1, -1]

def boarder_shape_b(s1, s2):
    for i1, e1 in enumerate(get_edges(s1)):
        for i2, e2 in enumerate(get_edges(s2)):
            if (np.sort(e1) == np.sort(e2)).all():
                # if (e1 == e2).all():
                #     inversed = False
                # else:
                #     inversed = True
                return True
    return False

def merge_shape(s1, s2):
    if type(s1) is np.ndarray:
        s1 = s1.tolist()
    pos = boarder_shape(s1, s2)
    if pos[1] == 0:
        s1.insert(pos[0] + 1, s2[2])
    elif pos[1] == 1:
        s1.insert(pos[0] + 1, s2[0])
    elif pos[1] == 2:
        s1.insert(pos[0] + 1, s2[1])
    else:
        # # print(str(s1) + ' ' + str(s2))
        return False
    return s1

sameShape = lambda s1, s2: s1[0] in s2 and s1[1] in s2 and s1[2] in s2

def co_plane(s1, s2, vertices):
    pos = boarder_shape(s1, s2)
    p1_i, p2_i, p3_i = s1[0], s1[1], s1[2]

    

    if pos == [-1, -1]:
        return False
    if pos[1] == 0:
        p_i = s2[2]
    elif pos[1] == 1:
        p_i = s2[0]
    elif pos[1] == 2:
        p_i = s2[1]

    p1 = vertices[p1_i]
    p2 = vertices[p2_i]
    p3 = vertices[p3_i]
    p = vertices[p_i]
    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1
    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp
    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = -np.dot(cp, p3)
    # # print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))
    # equation of plane is: a*x + b*y + c*z = 0 # 
    # checking if the 4th point satisfies 
    # the above equation 
    
    
    if(abs(a * p[0] + b * p[1] + c * p[2] + d) <= 10* sys.float_info.epsilon): 
        return True
    else: 
        return False 

def put_in_set(i, sets_list):
    in_sets = False
    for sets in sets_list:
        if i in sets:
            in_sets = True
    if not in_sets:
        sets_list.append({i})

def del_useless_edges(vor, hull):
    unique_edge_list = [[] for i in range(len(hull))]
    face_index_list = [[] for i in range(len(hull))]


    # for macro_i in range(353,354):
    for macro_i in range(len(hull)):
        # i = 0
        # # print(hull[0].simplices)
        hull_vertices = hull[macro_i].points
        
        tri_shapes = hull[macro_i].simplices
        poly_shapes = []

        # plot_hull(hull_vertices, hull, plotIndex=[0])

        # print('tri shape ' + str(tri_shapes))
        for i in range(len(tri_shapes)):
            for j in range(i):
                if co_plane(tri_shapes[i], tri_shapes[j], hull_vertices):
                    in_sets = False
                    for sets in poly_shapes:
                        if i in sets or j in sets:
                            sets.add(i)
                            sets.add(j)
                            in_sets = True
                            break
                    if not in_sets:
                        poly_shapes.append({i, j})
        mark_for_del = []
        mark_for_del_keep = []

        for i in range(len(poly_shapes)):
            for j in range(i):
                if len(poly_shapes[i].intersection(poly_shapes[j])) is not 0:
                    in_sets = False
                    for sets in mark_for_del:
                        if i in sets or j in sets:
                            sets.add(i)
                            sets.add(j)
                            in_sets = True
                            break
                    if not in_sets:
                        mark_for_del.append({i, j})

        for delete in mark_for_del:
            mark_for_del_keep.append(delete.pop())
            
        # print('mark for delete ' + str(mark_for_del))
        # print('mark for delete keep ' + str(mark_for_del_keep))
        # for i in range(len(mark_for_del)):
        #     mark_for_del[i] = mark_for_del[i].tolist()

        for i, delete_set in enumerate(mark_for_del):
            for delete in sorted(list(delete_set), reverse = True):
                poly_shapes[mark_for_del_keep[i]] = poly_shapes[mark_for_del_keep[i]].union(poly_shapes[delete])
        poly_shapes_kept = []

        delete_list = []
        for i in range(len(mark_for_del)):
            delete_list += list(mark_for_del[i])

        # get a list of not kept list
        for i in range(len(poly_shapes)):
            if i not in delete_list:
                poly_shapes_kept.append(poly_shapes[i])
                
        poly_shapes = poly_shapes_kept
        # print('poly shape now: ' + str(poly_shapes))

        # put the rest of triangular shape into the sets
        for i in range(len(tri_shapes)):
            put_in_set(i, poly_shapes)

        # print('poly_shape ' + str(macro_i) + ': ' + str(poly_shapes))

        merge_to_shape = [tri_shapes[sets.pop()] for sets in poly_shapes]
        
        for i in range(len(merge_to_shape)):
            tri_left = list(poly_shapes[i])
            for j in range(len(tri_left)):
                index_to_pop = 0
                while not (boarder_shape_b(merge_to_shape[i], tri_shapes[tri_left[index_to_pop]])) and index_to_pop < len(tri_left):
                    index_to_pop += 1
                merge_to_shape[i] = merge_shape(
                    merge_to_shape[i], 
                    tri_shapes[tri_left.pop(index_to_pop)]
                )
            # merge_to_shape[i] = np.array(vor.regions[macro_i])[merge_to_shape[i]].tolist()
            merge_to_shape[i] = np.array(merge_to_shape[i]).tolist()
        face_edge_list = []
        
        for i, shapes in enumerate(merge_to_shape):
            face_edge_list.append([])
            for j, edge in enumerate(get_edges(shapes)):
                unique_edge_list[macro_i].append(edge)
                face_edge_list[i].append(edge)
        unique_edge_list[macro_i] = np.unique(unique_edge_list[macro_i], axis=0).tolist()


        for i, face in enumerate(face_edge_list):
            face_index_list[macro_i].append([])
            for j, edge in enumerate(face):
                for k, unique_edge in enumerate(unique_edge_list[macro_i]):
                    if (edge == unique_edge).all():
                        face_index_list[macro_i][i].append(k) #= np.where(unique_edge_list[macro_i] == edge)[0]
                        break
                    
    # face_index_list = rid_dup_face(face_index_list, unique_edge_list)

    return unique_edge_list, face_index_list

def unroll_2(list_to_unroll):
    cum_unroll = 0


def rid_dup_face(face_index_list, unique_edge_list):
    """using center and norm to uniquely identify a face, get rid of it at other places."""

    # face_index_list = unroll_2(face_index_list)
    num_edge_per_face = []
    for i in range(len(face_index_list)):
        num_edge_per_face.append([])
        for j in range(len(face_index_list[i])):
            num_edge_per_face[i].append(len(face_index_list[i][j]))

    len_num_edge = np.cumsum([len(item) for item in num_edge_per_face])

    flatten = lambda l: [item for sublist in l for item in sublist]


    unrolled = flatten(num_edge_per_face)

    unrolled = np.array(unrolled)
    min = np.min(unrolled)
    max = np.max(unrolled)
    bins = np.arange(min, max+1)
    bins_index = [[] for i in range(len(bins))]

    for i in range(len(unrolled)):
        bins_index[unrolled[i]-min].append(i)

    # for bin in bins_index:


    print(bins_index)
    # num_edge_per_face = np.array(num_edge_per_face)
    # args = np.argsort(unrolled)


    return face_index_list
    # dup_sets = []
    # for i in range(len(node)):
    #     for j in range(i):
    #         if node[i] == node[j]:
    #             # print(str(i) + ' is the same as ' + str(j))
    #             dup_sets.append([i,j])
                
    # unique_sets = []

    # for i in range(len(dup_sets)):
    #     found = False
    #     for j, sets in enumerate(unique_sets):
    #         if dup_sets[i][0] in sets or dup_sets[i][1] in sets:
    #             sets.add(dup_sets[i][0])
    #             sets.add(dup_sets[i][1])
    #             found = True
    #             break
    #     if found:
    #         found = False
    #     else:
    #         unique_sets.append({dup_sets[i][0], dup_sets[i][1]})
    
    # hashed_nodes = set()



    # replaced_with = []
    # for sets in unique_sets:
    #     replaced_with.append(sets.pop())

    # list_nodes = []
    # index_for_sets = []

    # for i, s in enumerate(unique_sets):
    #     for n in s:
    #         list_nodes.append(n)
    #         index_for_sets.append(i)
    #         hashed_nodes.add(n)

    # list_np = np.sort(np.array(list_nodes))
    # sorted_arg = np.argsort(list_nodes)
    
    
    # for v in volumes:
    #     for ele in v:
    #         for i, e in enumerate(ele[1:]):
    #             e_1base = e -1
    #             j = i + 1
    #             if e_1base not in hashed_nodes:
    #                 continue
    #             index = np.searchsorted(list_np, e_1base, side='left')
    #             # print(list_np)
    #             # print(e)
    #             # print()
    #             if e_1base == list_np[index]:
    #                 # print(ele[j])
    #                 # print(replaced_with[index_for_sets[sorted_arg[index]]])
    #                 # print()
    #                 ele[j] = replaced_with[index_for_sets[sorted_arg[index]]] + 1


if __name__ == "__main__":
    # points = np.random.rand(30, 2)   # 30 random points in 2-D
    # hull = ConvexHull(points)

    # import matplotlib.pyplot as plt
    # plt.plot(points[:,0], points[:,1], 'o')
    # for simplex in hull.simplices:
    #     plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

    # plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
    # plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
    # plt.show()
    # points = np.array([[1, 1, -2],[3.5, 1.5, -11.5],[5, 5, -30], [3, 4.5, -22]])


    # points = np.array([[0.00820046851664,0.0857238532724,-0.0210645300889], [0.109327384468, 0.229168345257, 0.100517249346], [-0.21736658251, 0.0628641540209, -0.0509073122494], [0.0489275187761, 0.287231837685, 0.144667047312], [-0.439516382443, 0.276741226475, 0.111735276199]])
    # co_plane([0, 2, 3], [4, 2, 3], points)
    # result = 2.454633718507182e-16

    # print(merge_shape([1, 0, 7], [0, 7, 9]))
    pass