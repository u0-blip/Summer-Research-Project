from copy import copy
from abaqus import *
from abaqusConstants import *
import numpy as np
import os
import __main__

import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior
import pickle
from time import time 

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

def get_datum(p, vertices):
	for item in vertices:
		p.DatumPointByCoordinate(coords=(item[0], item[1], item[2]))

	
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

def connect_line(p, edge_list, vertices):
	d1 = p.datums
	edge_pointOn = []

	edge_creation_record = []
	
	for edge in edge_list:
		index1 = edge[0]+1 if edge[0] < 999 else edge[0]+2
		index2 = edge[1]+1 if edge[0] < 999 else edge[1]+2
		points=((d1[index1], d1[index2]),)

		recorded_point = [vertices[edge[0]], vertices[edge[1]]]

		recorded = in_record(recorded_point, edge_creation_record)
		edge_creation_record.append(recorded_point)

		if recorded == -1:
			edge_feature = p.WirePolyLine(points=points, mergeType=IMPRINT, meshable=ON)
			for edge_obj_in_part in p.edges:
				if edge_obj_in_part.featureName == edge_feature.name:
					edge_pointOn.append(edge_obj_in_part.pointOn)
					break
		else:
			edge_pointOn.append(edge_pointOn[recorded])
	# this return a list of edge point on that can uniquely identify the edge
	return edge_pointOn


def cover_face(p, edge_pointOn, face_index_list):
	edges = p.edges

	# for some reason, Abaqus refuse to create face if the face already exist
	# even tho it would happily do it for edges
	# so what we need to do is to test whether the edges have already be created
	# use the otherone to put it the pointOn
	# Once go past the face, the cell creation shouldn't be a big problem
	face_creation_record_pointOn = []
	face_pointOn = []

	for face in face_index_list:
		face_edge_obj = []
		face_creation_pointOn = []
		for edge_index in face:	
			# print(edge_index)
			face_creation_pointOn.append(edge_pointOn[edge_index])
			for edge_obj_in_part in edges:
				if edge_obj_in_part.pointOn == edge_pointOn[edge_index]:
					face_edge_obj.append(edge_obj_in_part)
					break

		recorded = in_record(face_creation_pointOn, face_creation_record_pointOn)

		face_creation_record_pointOn.append(face_creation_pointOn)
		if recorded == -1:
			face_feature = p.CoverEdges(edgeList = face_edge_obj, tryAnalytical=True)
			for face_obj_in_part in p.faces:
				if face_obj_in_part.featureName == face_feature.name:
					face_pointOn.append(face_obj_in_part.pointOn)
		else:
			face_pointOn.append(face_pointOn[recorded])
	# this return a list of face point on that can uniquely identify the face
	return face_pointOn

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

def fill_cell(p, face_pointOn, cell_index_list):

	cell_index_list = getCells(face_pointOn, cell_index_list)

	for cell in cell_index_list:
		faces = p.faces
		# print('cell is ' + str(cell))
		cell_face_obj = []
		for face_index in cell:	
			# print('face point on: ' + str(face_pointOn[face_index]))
			for face_obj_in_part in faces:
				if face_obj_in_part.pointOn[0] == face_pointOn[face_index][0]:
					cell_face_obj.append(face_obj_in_part)
					break
		# print(cell_face_obj)
		p.AddCells(faceList = cell_face_obj)
	

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


data = []
convexHull = []

with open(r'C:\\peter_abaqus\\Summer-Research-Project\\meep\\Voronoi_10000_seed_15.geo', 'rb') as f:
	data = pickle.load(f)

vertices, unique_edge_point_list, face_index_list = data

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
