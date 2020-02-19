# -*- coding: mbcs -*-
# Do not delete the following import lines
import sys
import pickle
import numpy as np
import json
import os
import sys
import shutil
import unicodedata

# working_dir = r'//ad.monash.edu/home/User045/dche145/Documents/Abaqus/microwave-break-rocks/'
working_dir = r'C:/peter_abaqus/Summer-Research-Project/abaqus_macro/'

sys.path.append(working_dir)
# os.chdir(working_dir)
import helpers
reload(helpers)
from helpers import *

import global_var 
reload(global_var)
from global_var import *

import mat_prop
reload(mat_prop)
from mat_prop import *


def create_material_section():
    mat_feld = model.Material(name=feldspar_prop['name'])
    # mat_feld_wo = model.Material(name=quartz_prop['name'])
    mat_quartz = model.Material(name=quartz_prop['name'])
    # mat_quartz_wo = model.Material(name=mat_quartz_name_wo)

    assign_prop(mat_feld, feldspar_prop, False)
    assign_prop(mat_quartz, quartz_prop, False)

    #creating the sections
    model.HomogeneousSolidSection(
        name=feldspar_prop['name'], material=feldspar_prop['name'],
        thickness=None)
    model.HomogeneousSolidSection(
        name=quartz_prop['name'], material=quartz_prop['name'],
        thickness=None)

    # model.HomogeneousSolidSection(
    #     name=mat_quartz_name_wo, material=mat_quartz_name_wo,
    #     thickness=None)
    # model.HomogeneousSolidSection(
    #     name=mat_quartz_name_w, material=mat_quartz_name_w,
    #     thickness=None)

def import_geo_info():
    global num_crystal
    global a_size
    global b_size
    global theta_x
    global theta_y
    global theta_z
    global loc
    geo_distro_3D = working_dir + 'working_with_meep\geometry.peter'
    with open(geo_distro_3D, 'rb') as f:
        read_out = json.load(f)

        num_crystal = read_out[0]
        a_size = read_out[1]
        b_size = read_out[2]
        for i in range(len(a_size)):
            a_size[i] = float(a_size[i])
        for i in range(len(b_size)):
            b_size[i] = float(b_size[i])
        loc = np.array(read_out[3])
        theta_x = np.array(read_out[4])
        theta_y = np.array(read_out[5])
        theta_z = np.array(read_out[6])



def import_3D_geo_shape():
    if clean_up_geo_test:
        with open(part_name_file) as f:
            part_name_list_read = json.load(f)
            for i in range(len(part_name_list_read)):
                part_name_list_read[i] = unicodedata.normalize(
                'NFKD', part_name_list_read[i]).encode('ascii','ignore')
        assembly.deleteFeatures(part_name_list_read)
    #define the geometric shape size and location

    global pyrite_part
    global calcite_part
    global pyrite_ins
    global calcite_ins
    calcite_part = part('calcite', dim=a_size, center=[0,0,0])
    pyrite_part = part('pyrite', dim=b_size, center=[0,0,0])

    for i in range(num_crystal):
        pyrite_ins.append(instance('pyrite-'+str(i), script_part=pyrite_part))
        part_name_list.append('pyrite-'+str(i))

    calcite_ins.append(instance('calcite-1', script_part=calcite_part))
    part_name_list.append('calcite-1')
    with open(part_name_file, 'w') as f:
        json.dump(part_name_list, f)

def create_3D_distro():
    #copy .\geometry.peter \\ad.monash.edu\home\User045\dche145\Documents\Abaqus\microwave-break-rocks\geometry.peter
    global theta_x
    global theta_y
    global theta_z
    global loc
    for i in range(num_crystal):
        pyrite_ins[i].rotate([theta_x[i],theta_y[i], theta_z[i]])
        pyrite_ins[i].translate(loc[i])

def merge_and_material():
    all_parts = []
    for i in range(num_crystal):
        all_parts.append(pyrite_ins[i].part)
    all_parts.append(calcite_ins[0].part)

    #cut so to get rid of external regions
    instance('calcite-for-cut', script_part=calcite_part)
    for i in xrange(num_crystal):
        assembly.InstanceFromBooleanCut(name='pyrite-to-get-rid-'+str(i), 
            instanceToBeCut=assembly.instances['pyrite-'+str(i)], 
            cuttingInstances=(assembly.instances['calcite-for-cut'], ), 
            originalInstances=SUPPRESS)
        assembly.features['pyrite-'+str(i)].resume()
        assembly.features['calcite-for-cut'].resume()

    del assembly.features['calcite-for-cut']

    assembly.InstanceFromBooleanMerge(name='merged', instances=all_parts,
        keepIntersections=ON, originalInstances=DELETE, domain=GEOMETRY)

    assembly.InstanceFromBooleanCut(name='merged', 
        instanceToBeCut=assembly.instances['merged-1'], 
        cuttingInstances=(list(assembly.instances['pyrite-to-get-rid-'+ str(i) + '-1'] for i in xrange(num_crystal))), 
        originalInstances=DELETE)

    for i in xrange(num_crystal):
        del model.parts['pyrite-to-get-rid-'+str(i)]

    with open(part_name_file, 'r') as f:
        part_name_list_read = json.load(f)
        part_name_list_read.append('merged-1')
    with open(part_name_file, 'w') as f:
        json.dump(part_name_list_read, f)

    merged_part = model.parts['merged']
    

    a_cells = []
    b_cells = []
    for cell in merged_part.cells:
        if(cell.getSize(printResults=False) > 0.5):
            a_cells.append(cell)
        else:
            b_cells.append(cell)

    merged_part.SectionAssignment(region=a_cells, sectionName='feldspar', offset=0.0,
    offsetType=MIDDLE_SURFACE, offsetField='',
    thicknessAssignment=FROM_SECTION)

    merged_part.SectionAssignment(region=b_cells, sectionName='quartz', offset=0.0,
    offsetType=MIDDLE_SURFACE, offsetField='',
    thicknessAssignment=FROM_SECTION)

    session.viewports['Viewport: 1'].setValues(displayedObject=assembly)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON,
        predefinedFields=ON, connectors=ON)
    assembly.regenerate()

def mesh_it():
    # Assign an element type to the part instance.
    part = model.parts['merged']
    region = (part.cells,)
    elemType = mesh.ElemType(elemCode=C3D10MT, elemLibrary=STANDARD)
    part.setElementType(regions=region, elemTypes=(elemType,))
    part.setMeshControls(regions=part.cells, elemShape=TET, technique=FREE)

    part.seedEdgeBySize(edges=part.edges, size=0.06, deviationFactor=0.1, 
            minSizeFactor=0.1, constraint=FINER)
    part.generateMesh(regions=region )

    # Display the meshed beam.
    disp_mesh=False
    if disp_mesh:
        myViewport.assemblyDisplay.setValues(mesh=ON)
        myViewport.assemblyDisplay.meshOptions.setValues(meshTechnique=ON)
        myViewport.setValues(displayedObject=assembly)

def load():
    # # Find the end face using coordinates.

    # endFaceCenter = (-100,0,12.5)
    # endFace = myInstance.faces.findAt((endFaceCenter,) )

    # # Create a boundary condition that encastres one end
    # # of the beam.

    # endRegion = (endFace,)
    # myModel.EncastreBC(name='Fixed',createStepName='beamLoad',
    #     region=endRegion)

    # # Find the top face using coordinates.

    # topFaceCenter = (0,10,12.5)
    # topFace = myInstance.faces.findAt((topFaceCenter,) )

    # # Create a pressure load on the top face of the beam.

    # topSurface = ((topFace, SIDE1), )
    # myModel.Pressure(name='Pressure', createStepName='beamLoad',
    #     region=topSurface, magnitude=0.5)
    
    model.CoupledTempDisplacementStep(name='heat_up', 
        previous='Initial', timePeriod=2.0, maxNumInc=150, initialInc=0.2, 
        minInc=0.002, maxInc=2.0, deltmx=50.0)
        
    cells = assembly.instances['merged-2'].cells

    a_cells = []
    b_cells = []
    for cell in cells:
        if(cell.getSize(printResults=False) > 0.5):
            a_cells.append(cell)
        else:
            b_cells.append(cell)

    model.BodyHeatFlux(name='dflux', 
        createStepName='heat_up', region=b_cells, magnitude=10.0, 
        distributionType=USER_DEFINED)

def boundary(type, mag=0):
    #setting up all the points that the surface is on
    #setting up 24 points for all the surfaces
    dim = 0.5
    offset=0.1
    points_corner = [
        [dim - offset, dim - offset,  dim], 
        [dim - offset, dim, -dim + offset],
        [dim - offset, -dim, dim - offset],
        [dim - offset, -dim + offset, -dim],
        [-dim + offset, dim - offset,  dim], 
        [-dim + offset, dim, -dim + offset],
        [-dim + offset, -dim, dim - offset],
        [-dim + offset, -dim + offset, -dim],
        [dim, 0, 0],
        [-dim, 0, 0]
    ]
    faces = assembly.instances['merged-2'].faces
    face_set = set()
    for i in range(len(points_corner)):
        face_set.add(faces.findAt(points_corner[i]).index)

    copy_set = set(face_set)
    for i in face_set:
        adj = faces[i].getAdjacentFaces()
        for adj_face in adj:
            if adj_face.getNormal() == faces[i].getNormal():
                copy_set.add(adj_face.index)
                
    face_set=copy_set
    outer_face = []
    for i in face_set:
        outer_face.append(faces[i])

    if type is 'encastre':
        model.EncastreBC(name='fixed', createStepName='Initial', region = outer_face)
    elif type is 'symmetry':
        model.XsymmBC(name='xsym', createStepName='Initial', region=outer_face, localCsys=None)
    elif type is 'pressure':
        model.Pressure(name='pressure', createStepName='Initial', region = outer_face,         magnitude = mag)

def run_job(magnitude=10e8, timePeriod=5, increment=0.3):
#setup different simulation jobs
    name_job = get_name_job(magnitude)
    pre_fix = 'M'+name_job

    mdb.models['square-3d'].loads['Load-1'].setValues(magnitude=magnitude)
    mdb.models['square-3d'].steps['heat_up'].setValues(timePeriod=timePeriod,
        initialInc=increment)
    mdb.Job(name=name_job, model='square-3d', description='', type=ANALYSIS,
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90,
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF,
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='',
        scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1,
        numGPUs=0)
    job = mdb.jobs[name_job]
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()
    # while os.path.isfile(name_job+'.023') or os.path.isfile(name_job+'.lck') == True:
    #     sleep(0.1)
    shutil.copyfile(name_job+'.odb', working_dir+'data_base'+'/'+name_job+'.odb')
    return name_job

def get_output_data(name_job, step, frame, num_intervals, meta_data = None):
    file_path = name_job +'.odb'
    odb = session.openOdb(name=file_path)
    # odb = session.odbs[file_path]
    session.viewports['Viewport: 1'].setValues(displayedObject=odb)
    #getting the output data from the model
    session.Path(name='Path-diagonal', type=POINT_LIST, expression=((0.0299999993294477,0.0299999993294477,0.0),
    (-0.0299999993294477,-0.0299999993294477,0.0)))
    pth_dia = session.paths['Path-diagonal']

    xy_data = [[] for i in range(frame)]
    xy_data_name = ''
    xy_data_name_str = ''


    for i in xrange(frame):
        session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=i)
        xy_data_name = name_job + 'stress' + str(i)

        if i is not frame-1:
            xy_data_name_str += xy_data_name +','
        else:
            xy_data_name_str += xy_data_name

        session.XYDataFromPath(name=xy_data_name, path=pth_dia, includeIntersections=False,
            projectOntoMesh=False, pathStyle=UNIFORM_SPACING, numIntervals=num_intervals,
            projectionTolerance=0, shape=UNDEFORMED, labelType=TRUE_DISTANCE,
            removeDuplicateXYPairs=True, includeAllElements=False)
        xy_data[i] = session.xyDataObjects[xy_data_name].data

    meta_data['frame'] = frame
    meta_data['num_intervals'] = num_intervals

    global new_session

    if new_session:
        new_session=False
        mode = 'wb'
    else:
        mode = 'ab'

    with open(working_dir + 'saved_data', mode) as f:
        pickle.dump(meta_data, f)
        pickle.dump(xy_data, f)



if __name__== "__main__": 
    try:
        # import_geo_info()
        create_material_section()
        # import_3D_geo_shape()
        # create_3D_distro()
        # merge_and_material()
        # mesh_it()
        # boundary('encastre')
        # load()
    except Exception as inst:
        print type(inst)     # the exception instance
        print inst.args
        print inst

    # for i in xrange(num_change_flux):
    #     name_job = run_job(magnitude[i], timePeriod, increment)
    #     meta_data = {
    #         'name': name_job,
    #         'magnitude': magnitude[i],
    #     }
    #     get_output_data(name_job, step, frame, num_intervals, meta_data)

    # run on TeamViewer
    # shutil.copyfile('C:/Users/dche145/abaqusMacros.py', r'C:/peter_abaqus/Summer-Research-Project/abaqus_macro/macro.py')
    # shutil.copyfile('C:/temp/dflux.inp', r'C:/peter_abaqus/Summer-Research-Project/abaqus_working_space/abaqus_out/dflux.inp')
    # shutil.copyfile('C:/temp/umat_test.inp', r'C:/peter_abaqus/Summer-Research-Project/abaqus_working_space/abaqus_out/umat_test.inp')
    # execfile('C:/peter_abaqus/Summer-Research-Project/abaqus_macro/main.py', __main__.__dict__)
    # os.chdir(r"C:\peter_abaqus\Summer-Research-Project\abaqus_working_space\abaqus_out")

    # Run on Citrix
    # shutil.copyfile('C:/Users/dche145/abaqusMacros.py', r'//ad.monash.edu/home/User045/dche145/Documents/Abaqus/microwave-break-rocks/macro.py')
    # execfile('//ad.monash.edu/home/User045/dche145/Documents/Abaqus/microwave-break-rocks/main.py', __main__.__dict__)
