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

    # print(sum(abs(x)))
	if sum(abs(x))/avg_dist < 5e-6: #and (dist1 < 0.05 or dist2 < 0.05:
		# print('before : ' + str(sum(abs(x))) + 'after : ' + str(sum(abs(x))/avg_dist))
		return True
	else:
		# print('wrong : ' + str(dist1) + ' ' + str(dist2))
		return False

data = []
convexHull = []

with open(r'C:\\peter_abaqus\\Summer-Research-Project\\meep\\polygon1.csv', 'rb') as f:
	data = pickle.load(f)

vertices, unique_edge_list, face_index_list = data

parts_list = []

for i in range(1):
	# create coordinate reference and supress
	part_name = 'Part-'+str(i)
	p = mdb.models['Model-1'].Part(name=part_name, dimensionality=THREE_D, type=DEFORMABLE_BODY)


	for item in vertices[i]:
		p.DatumPointByCoordinate(coords=(item[0], item[1], item[2]))
		
	d1 = p.datums
	print(d1)
	print(unique_edge_list[i])
	print(face_index_list[i])
	# join datum for edges

	edge_names = []
	edge_obj = []

	for edge in unique_edge_list[i]:
		edge_feature = p.WirePolyLine(points=((d1[edge[0]+1], d1[edge[1]+1]),), mergeType=IMPRINT, meshable=ON)
		edge_names.append(edge_feature.name)

	# print('face edge name list')
	# print(face_edge_name_list)
	print('edge list')
	for e in p.edges:
		print(e) 

	pedges = p.edges
	for face in face_index_list[i]:
		face_edge_obj = []
		for edge_index in face:	
			print(edge_index)
			for edge_obj_in_part in pedges:
				# print('feature name is: ' + edge_obj_in_part.featureName)
				# print('edge name is: ' + edge_names[edge_index])
				if edge_obj_in_part.featureName == edge_names[edge_index]:
					face_edge_obj.append(edge_obj_in_part)
		
		# face_edge_obj_list.append(edge_obj)
		# print('face edge object')
		# print(face_edge_obj)		

		# 	a = p.WirePolyLine(points=((d1[int(face[0])+1], d1[int(face[1])+1]),), mergeType=IMPRINT, meshable=ON)
		# b = p.WirePolyLine(points=((d1[int(face[1])+1], d1[int(face[2])+1]),), mergeType=IMPRINT, meshable=ON)
		# c = p.WirePolyLine(points=((d1[int(face[2])+1], d1[int(face[0])+1]),), mergeType=IMPRINT, meshable=ON)
		# print(a.name)
		print('face edge: ' + str(face_edge_obj))
		p.CoverEdges(edgeList = face_edge_obj, tryAnalytical=True)

	# eg = p.edges

	# # create faces
	# for hull in convexHull[i]:
	# 	seq = []
	# 	wireSet = []
	# 	for edge in eg :
	# 		print(edge.id)
	# 		point = edge.pointOn[0]
	# 		if isCollinear(point, d1[int(hull[0])+1].pointOn,d1[int(hull[1])+1].pointOn):	
	# 			if 1 in wireSet:
	# 				continue
	# 			else:
	# 				seq.append(edge)
	# 				wireSet.append(1)
	# 		if isCollinear(point, d1[int(hull[1])+1].pointOn,d1[int(hull[2])+1].pointOn):
	# 			if 2 in wireSet:
	# 				continue
	# 			else:
	# 				seq.append(edge)
	# 				wireSet.append(2)
	# 		if isCollinear(point, d1[int(hull[2])+1].pointOn,d1[int(hull[0])+1].pointOn):
	# 			if 3 in wireSet:
	# 				continue
	# 			else:
	# 				seq.append(edge)
	# 				wireSet.append(3)
	# 		if len(seq) == 3:
	# 			p.CoverEdges(edgeList = seq, tryAnalytical=True)
	# 			break

	# f = p.faces
	# p.AddCells(faceList = f)

# execfile('C:/peter_abaqus/Summer-Research-Project/abaqus_macro/abq_create_polygon.py', __main__.__dict__)
