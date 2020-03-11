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

for i in range(len(vertices)):
	# create coordinate reference and supress
	part_name = 'Part-'+str(i)
	p = mdb.models['Model-1'].Part(name=part_name, dimensionality=THREE_D, type=DEFORMABLE_BODY)


	for item in vertices[i]:
		p.DatumPointByCoordinate(coords=(item[0], item[1], item[2]))
		
	d1 = p.datums
	# print(d1)
	# print(unique_edge_list[i])
	# print(face_index_list[i])
	# join datum for edges

	edge_names = []
	edge_obj = []

	for edge in unique_edge_list[i]:
		edge_feature = p.WirePolyLine(points=((d1[edge[0]+1], d1[edge[1]+1]),), mergeType=IMPRINT, meshable=ON)
		edge_names.append(edge_feature.name)

	# print('face edge name list')
	# print(face_edge_name_list)
	# need to check whether the pointOn of the edges can actually uniquely identify the edges



	pedges = p.edges
	edge_pointOn = [None] * len(edge_names)


	for edge_obj_in_part in pedges:
		for edge_index in range(len(edge_names)):
			if edge_obj_in_part.featureName == edge_names[edge_index]:
				edge_pointOn[edge_index] = edge_obj_in_part.pointOn

	# print(edge_pointOn)

	for face in face_index_list[i]:
		face_edge_obj = []
		for edge_index in face:	
			for edge_obj_in_part in pedges:
				# print('feature name is: ' + edge_obj_in_part.featureName)
				# print('edge name is: ' + edge_names[edge_index])
				if edge_obj_in_part.pointOn == edge_pointOn[edge_index]:
					face_edge_obj.append(edge_obj_in_part)

		p.CoverEdges(edgeList = face_edge_obj, tryAnalytical=True)

	f = p.faces
	p.AddCells(faceList = f)

	# mdb.saveAs(pathName='C:/peter_abaqus/Summer-Research-Project/abaqus_working_space/voronoi_500')
# execfile('C:/peter_abaqus/Summer-Research-Project/abaqus_macro/abq_create_polygon.py', __main__.__dict__)
