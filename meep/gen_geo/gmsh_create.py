from copy import copy
import pickle
from time import time 
import sys
import numpy as np

sys.path.append(r'C:\Users\dche145\Downloads\gmsh-4.5.5-Windows64-sdk\gmsh-4.5.5-Windows64-sdk\lib')

import gmsh


def isCollinear(p0, p1, p2):
	p0 = np.array(p0)
	p1 = np.array(p1)
	p2 = np.array(p2)
	e1 = p0 - p1
	e2 = p0 - p2
	x = np.cross(e1, e2)
	
	# if sum(abs(x)) > 1e-6:
	# 	return False

	dist1 = np.linalg.norm(e1)
	dist2 = np.linalg.norm(e2)

	avg_dist = (dist1+dist2)/2

	# # print(sum(abs(x)))
	if sum(abs(x))/avg_dist < 5e-6: #and (dist1 < 0.05 or dist2 < 0.05:
		# # print('before : ' + str(sum(abs(x))) + 'after : ' + str(sum(abs(x))/avg_dist))
		return True
	else:
		# # print('wrong : ' + str(dist1) + ' ' + str(dist2))
		return False

def get_datum(vertices):
	geo_p = []
	for polygon in vertices:
		geo_pp = []
		for point in polygon:
			geo_pp.append(gmsh.model.occ.addPoint(point[0], point[1], point[2]))
		geo_p.append(geo_pp)
	return geo_p

def in_record(pointOn, records):
	# pointOn is multiple points that define multiple edges that define faces
	# records is a list of pointOn
	# The order of pointOn and the order it is in the records might be different
	
	for i, record in enumerate(records):
		if len(record) != len(pointOn):
			continue
		same = True
		for line in pointOn:
			if line not in record:
				same = False
		if same:
			return i
	return -1

def connect_line(edge_list, geo_p):
	# edge_pointOn = []

	# edge_creation_record = []
	
	# for edge in edge_list:
	# 	index1 = edge[0]+1 if edge[0] < 999 else edge[0]+2
	# 	index2 = edge[1]+1 if edge[0] < 999 else edge[1]+2
	# 	points=((d1[index1], d1[index2]),)

	# 	recorded_point = [vertices[edge[0]], vertices[edge[1]]]

	# 	recorded = in_record(recorded_point, edge_creation_record)
	# 	edge_creation_record.append(recorded_point)

	# 	if recorded == -1:
	# 		edge_feature = p.WirePolyLine(points=points, mergeType=IMPRINT, meshable=ON)
	# 		for edge_obj_in_part in p.edges:
	# 			if edge_obj_in_part.featureName == edge_feature.name:
	# 				edge_pointOn.append(edge_obj_in_part.pointOn)
	# 				break
	# 	else:
	# 		edge_pointOn.append(edge_pointOn[recorded])
	# # this return a list of edge point on that can uniquely identify the edge
	# return edge_pointOn
	geo_l = []
	for i, polygon in enumerate(edge_list):
		geo_ll = []
		for edge in polygon:
			line = gmsh.model.occ.addLine(geo_p[i][edge[0]], geo_p[i][edge[1]])
			geo_ll.append(line)
		geo_l.append(geo_ll)
	return geo_l

def cover_face(geo_l, face_index_list):
	# edges = p.edges

	# # for some reason, Abaqus refuse to create face if the face already exist
	# # even tho it would happily do it for edges
	# # so what we need to do is to test whether the edges have already be created
	# # use the otherone to put it the pointOn
	# # Once go past the face, the cell creation shouldn't be a big problem
	# face_creation_record_pointOn = []
	# face_pointOn = []

	geo_f = []
	for i, polygon in enumerate(face_index_list):
		geo_ff = []
		for face in polygon:	
			face_loop = []
			for edge in face:
				face_loop.append(geo_l[i][edge])
			loop = gmsh.model.occ.addCurveLoop(face_loop)
			surface = gmsh.model.occ.addPlaneSurface([loop])
			geo_ff.append(surface)
		geo_f.append(geo_ff)
	return geo_f
			

	# 	recorded = in_record(face_creation_pointOn, face_creation_record_pointOn)

	# 	face_creation_record_pointOn.append(face_creation_pointOn)
	# 	if recorded == -1:
	# 		face_feature = p.CoverEdges(edgeList = face_edge_obj, tryAnalytical=True)
	# 		for face_obj_in_part in p.faces:
	# 			if face_obj_in_part.featureName == face_feature.name:
	# 				face_pointOn.append(face_obj_in_part.pointOn)
	# 	else:
	# 		face_pointOn.append(face_pointOn[recorded])
	# # this return a list of face point on that can uniquely identify the face
	# return face_pointOn

def put_in_set(i, sets_list):
	in_sets = False
	for sets in sets_list:
		if i in sets:
			in_sets = True
	if not in_sets:
		sets_list.append({i})

def getCells(face_pointOn, cell_index_list):
	keptCell = []

	for i, cell1 in enumerate(cell_index_list):
		keptCell.append({i})
		for j in range(i):
			cell2 = cell_index_list[j]
			sameface = False
			for k, faceIndex1 in enumerate(cell1):
				for p, faceIndex2 in enumerate(cell2):
					if face_pointOn[faceIndex1][0] == face_pointOn[faceIndex2][0]:
						cell1.pop(k)
						cell2.pop(p)
						sameface = True
						break
				if sameface:
					break
			if sameface:
				in_sets = False
				for sets in keptCell:
					if i in sets or j in sets:
						sets.add(i)
						sets.add(j)
						in_sets = True
						break
				if not in_sets:
					keptCell.append({i, j})
	# print(keptCell)
		# for index in cell1:
		# 	face = face_pointOn[index][0]
		# 	if face not in uniqueFace:
		# 		uniqueFace.append(face)
		# 	else:
		# 		repeatFaceIndex.append(index)
		# 		k = index
		# 		for j, cellLen in enumerate(cell_index_list):
		# 			k -= len(cellLen)
		# 			if k <= 0:
		# 				break
		# 		in_sets = False
		# 		for sets in keptCell:
		# 			if i in sets or j in sets:
		# 				sets.add(i)
		# 				sets.add(j)
		# 				in_sets = True
		# 				break
		# 		if not in_sets:
		# 			keptCell.append({i, j})

	# keptCell = [set([1]), set([1,2,3]), set([2,8,9]), set([4,23,32]), set([2]), set([3]), set([4])]
	mark_for_del = []
	mark_for_del_keep = []

	for i in range(len(keptCell)):
		for j in range(i):
			if len(keptCell[i].intersection(keptCell[j])) is not 0:
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

	for i, delete_set in enumerate(mark_for_del):
			for delete in sorted(list(delete_set), reverse = True):
				keptCell[mark_for_del_keep[i]] = keptCell[mark_for_del_keep[i]].union(keptCell[delete])

	keptCell_kept = []

	delete_list = []
	for i in range(len(mark_for_del)):
		delete_list += list(mark_for_del[i])

	# get a list of not kept list
	for i in range(len(keptCell)):
		if i not in delete_list:
			keptCell_kept.append(keptCell[i])
			
	keptCell = keptCell_kept
	# print(keptCell)
	newFaceIndex = []
	for cell in keptCell:
		facePoints = set()
		for index in cell:
			facePoints = facePoints.union(facePoints, set(cell_index_list[index]))
		newFaceIndex.append(list(facePoints))
	return newFaceIndex

def fill_cell(geo_f):
	volumes = []
	for i, cell in enumerate(geo_f):
		# surf_loop = []
		# for face in cell:
		# 	surf_loop.append(geo_f[i][face])
		geo_surf_loop = gmsh.model.occ.addSurfaceLoop(cell)
		vol = gmsh.model.occ.addVolume([geo_surf_loop])
		volumes.append(vol)
	return volumes
	# cell_index_list = getCells(face_pointOn, cell_index_list)

	# for cell in cell_index_list:
	# 	faces = p.faces
	# 	# print('cell is ' + str(cell))
	# 	cell_face_obj = []
	# 	for face_index in cell:	
	# 		# print('face point on: ' + str(face_pointOn[face_index]))
	# 		for face_obj_in_part in faces:
	# 			if face_obj_in_part.pointOn[0] == face_pointOn[face_index][0]:
	# 				cell_face_obj.append(face_obj_in_part)
	# 				break
	# 	# print(cell_face_obj)
	# 	p.AddCells(faceList = cell_face_obj)
	

start = time()
previous = time()

def track_time(i):
	global now
	global previous
	now = time()
	elapsed = now - start
	interval = now - previous
	previous = now
	print('Creating the ' + str(i*100) + 'th polygon.')
	print('Total elapsed time: ' + str(elapsed))
	print('Elapsed time interval: ' + str(interval))


def create_in_abq():
	data = []
	convexHull = []

	parts_list = []
	section_list = [range(300, 400)]

	# for i in range(353,354):
	for i, section in enumerate(section_list):
	# for i in range(len(vertices)):
		# create coordinate reference and supress
		part_name = 'Section-'+str(i)
		p = mdb.models['Model-1'].Part(name=part_name, dimensionality=THREE_D, type=DEFORMABLE_BODY)

		# v = vertices[i]
		# e = unique_edge_point_list[i]
		# f = face_index_list[i]
		v = []
		e = []
		f = []
		c = []
		running_vertices_index_count = 0
		running_edge_index_count = 0
		running_face_index_count = 0

		for part in section:
			v = v + vertices[part]
			e = e + [[point_index + running_vertices_index_count for point_index in point_index_pair] for point_index_pair in unique_edge_point_list[part]]

			f = f + [[face_index + running_edge_index_count for face_index in face_index_group]for face_index_group in face_index_list[part]]

			c = c + [range(running_face_index_count, running_face_index_count + len(face_index_list[part]))]
			running_vertices_index_count += len(vertices[part])
			running_edge_index_count += len(unique_edge_point_list[part])
			running_face_index_count += len(face_index_list[part])
		
		get_datum(p, v)
		edge_pointOn = connect_line(p, e, v)
		face_pointOn = cover_face(p, edge_pointOn, f)
		fill_cell(p, face_pointOn, c)

		if i%1 == 0:
			track_time(i)

		# mdb.saveAs(pathName='C:/peter_abaqus/Summer-Research-Project/abaqus_working_space/voronoi_500')
		
	# cell_index_list = [[1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15]]
	# for i in range(len(cell_index_list)):
	# 	for j in range(len(cell_index_list[i])):
	# 		cell_index_list[i][j] -= 1

	# face_pointOn = [[1.1],[2.2],[3.3],[4.4],[5.4],
	# 				[6.4], [1.1],[8.8],[9.9], [4.4],
	# 					[11.1],[12.2],[13.3],[14.4],[15.5]]
	# print(getCells(face_pointOn, cell_index_list))

	# execfile('C:/peter_abaqus/Summer-Research-Project/abaqus_macro/abq_create_polygon.py', __main__.__dict__)

def get_simple_data():
	cube1 = np.array([
		[0, 0, 0],
		[0, 0, 1],
		[0, 1, 0],
		[0, 1, 1],
		[1, 0, 0],
		[1, 0, 1],
		[1, 1, 0],
		[1, 1, 1]
	])
	cube2 = cube1.copy()
	for ele in cube2:
		ele[0] = ele[0] + 1
		
	line1 = np.array([
		[0, 1],
		[0, 4],
		[1, 5],
		[5, 4],
		[5, 7],
		[7, 3],
		[1, 3],
		[6, 7],
		[4, 6],
		[2, 6],
		[3, 2],
		[0, 2]
	])
	line2 = line1 #+ len(cube1)
	
	face1 = np.array([
		[0, 1, 3, 2],
		[3, 8, 7, 4],
		[2, 4, 5, 6],
		[0, 11, 10, 6],
		[1, 8, 9, 11],
		[9, 7, 5, 10]
	])
	face2 = face1 #+ len(line1)

	bd1 = np.array([0, 1, 2, 3, 4, 5])
	bd2 = bd1 #+ len(face1)

	vertices = [cube1, cube2]
	unique_edge_point_list = [line1, line2]
	face_index_list = [face1, face2]

def simple_test():
	gmsh.initialize(sys.argv)
	gmsh.fltk.initialize()

	gmsh.option.setNumber("General.Terminal", 1)

	gmsh.model.add("square with cracks")
	surf1 = 1
	surf2 = 2
	gmsh.model.occ.addRectangle(0, 0, 0, 1, 1, surf1)
	gmsh.model.occ.addRectangle(0, 0, 0, -1, -1, surf2)

	g1 = gmsh.model.addPhysicalGroup(2, [1])
	g2 = gmsh.model.addPhysicalGroup(2, [2])
	# pt1 = gmsh.model.occ.addPoint(0.2, 0.2, 0)
	# pt2 = gmsh.model.occ.addPoint(0.4, 0.4, 0)
	# line1 = gmsh.model.occ.addLine(pt1, pt2)
	# pt3 = gmsh.model.occ.addPoint(0.6, 0.1, 0)
	# pt4 = gmsh.model.occ.addPoint(0.1, 0.3, 0)
	# line2 = gmsh.model.occ.addLine(pt3, pt4)

	# o, m = gmsh.model.occ.fragment([(2,surf1)], [(1,line1), (1,line2)])
	gmsh.model.occ.synchronize()

	# m contains, for each input entity (surf1, line1 and line2), the child entities
	# (if any) after the fragmentation, as lists of tuples. To apply the crack
	# plugin we group all the intersecting lines in a physical group

	# new_surf = m[0][0][1]
	# new_lines = [item[1] for sublist in m[1:] for item in sublist]

	# gmsh.model.addPhysicalGroup(2, [new_surf], 100)
	# gmsh.model.addPhysicalGroup(1, new_lines, 101)
	pnts = gmsh.model.getBoundary([(2,surf2)], True, True, True)
	print(type(pnts[0]))
	gmsh.model.mesh.setSize(pnts, 0.2)
	gmsh.model.mesh.generate(2)

	# gmsh.plugin.setNumber("Crack", "PhysicalGroup", 101)
	# gmsh.plugin.run("Crack")
	gmsh.write('gmsh_out/crack.bdf')
	gmsh.fltk.run()

	gmsh.finalize()

def abq_create(mesh_size = 0.5, display = False, ingeo = 'Voronoi_500_seed_15.geo', out_f='gmsh_test.inp'):

	# gmsh.initialize(sys.argv)
	gmsh.initialize()
	if display: gmsh.fltk.initialize()
	# gmsh.option.setNumber("General.Terminal", 1)
	gmsh.model.add("Voronoi geo")

	with open(r'C:\\peter_abaqus\\Summer-Research-Project\\meep\\' + ingeo, 'rb') as f:
		data = pickle.load(f)

	vertices, unique_edge_point_list, face_index_list = data

	

	
	geo_p = get_datum(vertices)
	geo_l = connect_line(unique_edge_point_list, geo_p)
	geo_f = cover_face(geo_l, face_index_list)
	volume = fill_cell(geo_f)
	gmsh.model.occ.synchronize()

	# pnts = gmsh.model.getBoundary([(3,surf2)], True, True, True)
	# gmsh.model.mesh.setSize(pnts, 0.2)

	# gmsh.model.mesh.field.setAsBackgroundMesh()
	# geo_p = np.array(geo_p)
	# geo_p = np.expand_dims(geo_p,1)
	# geo_p = np.concatenate((np.zeros_like(geo_p), geo_p), axis=1)
	geo_p_list = []
	for ele1 in geo_p:
		for ele2 in ele1:
			geo_p_list.append((0, ele2))

	gmsh.model.mesh.setSize(geo_p_list, mesh_size)
	gmsh.model.mesh.generate(3)
	gmsh.write('gmsh_out/' + out_f)

	if display: gmsh.fltk.run()
	gmsh.finalize()


if __name__ ==  "__main__":
	# simple_test()
	before = time()
	abq_create()
	after = time()
	elapsed = after-before
	print('the total time elapsed is ' + str(elapsed))
	