# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
import __main__

def create_model():
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
    mdb.Model(name='test_macro', modelType=STANDARD_EXPLICIT)
    a = mdb.models['test_macro'].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    del mdb.models['test_macro']
    a = mdb.models['prism-3d'].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    mdb.save()


def merged_mesh():
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
    a = mdb.models['test'].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    a1 = mdb.models['test'].rootAssembly
    a1.DatumCsysByDefault(CARTESIAN)
    p = mdb.models['test'].parts['calcite']
    a1.Instance(name='calcite-1', part=p, dependent=ON)
    p = mdb.models['test'].parts['pyrite']
    a1.Instance(name='pyrite-1', part=p, dependent=ON)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON)
    session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
        meshTechnique=ON)
    p = mdb.models['test'].parts['calcite']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    session.viewports['Viewport: 1'].partDisplay.setValues(mesh=ON)
    session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
        meshTechnique=ON)
    session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
        referenceRepresentation=OFF)
    p = mdb.models['test'].parts['calcite']
    p.seedPart(size=0.1, deviationFactor=0.1, minSizeFactor=0.1)
    p = mdb.models['test'].parts['calcite']
    p.generateMesh()
    p = mdb.models['test'].parts['pyrite']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models['test'].parts['pyrite']
    p.seedPart(size=0.01, deviationFactor=0.1, minSizeFactor=0.1)
    p = mdb.models['test'].parts['pyrite']
    p.generateMesh()
    a1 = mdb.models['test'].rootAssembly
    a1.regenerate()
    a = mdb.models['test'].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
        meshTechnique=OFF)
    a1 = mdb.models['test'].rootAssembly
    p = mdb.models['test'].parts['calcite']
    a1.Instance(name='calcite-2', part=p, dependent=ON)
    p = mdb.models['test'].parts['pyrite']
    a1.Instance(name='pyrite-2', part=p, dependent=ON)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON)
    session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
        meshTechnique=ON)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.57887, 
        farPlane=4.73821, width=2.53273, height=1.13756, 
        viewOffsetX=0.00488481, viewOffsetY=-0.0263411)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.55118, 
        farPlane=4.69291, width=2.50554, height=1.12535, cameraPosition=(
        2.26406, 2.45059, -0.910593), cameraUpVector=(-0.948101, 0.269485, 
        -0.168765), cameraTarget=(0.0116204, -0.0492617, 0.525441), 
        viewOffsetX=0.00483236, viewOffsetY=-0.0260583)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.54124, 
        farPlane=4.65894, width=2.49578, height=1.12096, cameraPosition=(
        1.28268, 2.51986, -1.72881), cameraUpVector=(-0.933572, 0.228355, 
        0.27622), cameraTarget=(0.021047, -0.0512618, 0.547707), 
        viewOffsetX=0.00481353, viewOffsetY=-0.0259567)
    p = mdb.models['test'].parts['pyrite']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models['test'].parts['pyrite']
    n = p.nodes
    nodes = n.getSequenceFromMask(mask=(
        '[#ffffffff:37 #fbffffff #ffffffff:3 #7ffff ]', ), )
    p.Set(nodes=nodes, name='b_nodes')
    p = mdb.models['test'].parts['calcite']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models['test'].parts['calcite']
    e = p.elements
    elements = e.getSequenceFromMask(mask=('[#ffffffff:31 #ff ]', ), )
    p.Set(elements=elements, name='a_elements')
    a1 = mdb.models['test'].rootAssembly
    a1.regenerate()
    a = mdb.models['test'].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
        meshTechnique=OFF)
    a1 = mdb.models['test'].rootAssembly
    a1.InstanceFromBooleanMerge(name='merged_mesh', instances=(
        a1.instances['calcite-1'], a1.instances['pyrite-1'], 
        a1.instances['calcite-2'], a1.instances['pyrite-2'], ), mergeNodes=ALL, 
        nodeMergingTolerance=1e-06, domain=MESH, originalInstances=SUPPRESS)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
        predefinedFields=ON, connectors=ON)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='heat_up')
    session.viewports['Viewport: 1'].partDisplay.setValues(mesh=OFF)
    session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
        meshTechnique=OFF)
    session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
        referenceRepresentation=ON)
    p = mdb.models['test'].parts['calcite']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    session.viewports['Viewport: 1'].partDisplay.setValues(mesh=ON)
    session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
        meshTechnique=ON)
    session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
        referenceRepresentation=OFF)
    p = mdb.models['test'].parts['pyrite']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models['test'].parts['pyrite']
    e = p.elements
    elements = e.getSequenceFromMask(mask=('[#ffffffff:31 #ff ]', ), )
    p.Set(elements=elements, name='b_elements')
    a = mdb.models['test'].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON, loads=OFF, 
        bcs=OFF, predefinedFields=OFF, connectors=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
        meshTechnique=ON)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
        meshTechnique=OFF)
    a = mdb.models['test'].rootAssembly
    a.resumeFeatures(('calcite-2', 'pyrite-2', ))
    a = mdb.models['test'].rootAssembly
    a.features['merged_mesh-1'].suppress()
    a1 = mdb.models['test'].rootAssembly
    a1.InstanceFromBooleanMerge(name='merged_mesh_2', instances=(
        a1.instances['calcite-2'], a1.instances['pyrite-2'], ), mergeNodes=ALL, 
        nodeMergingTolerance=1e-06, domain=MESH, originalInstances=SUPPRESS)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
        predefinedFields=ON, connectors=ON)
    a = mdb.models['test'].rootAssembly
    region = a.instances['merged_mesh_2-1'].sets['b_elements']
    mdb.models['test'].loads['dflux'].setValues(region=region, 
        distributionType=UNIFORM)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.4843, 
        farPlane=4.71588, width=3.33304, height=1.49317, 
        viewOffsetX=-0.0956355, viewOffsetY=-0.0501579)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Initial')
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='heat_up')
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Initial')
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.54638, 
        farPlane=4.56717, width=3.41634, height=1.53048, cameraPosition=(
        -0.270003, 3.14644, -1.13683), cameraUpVector=(-0.864754, -0.229316, 
        0.446783), cameraTarget=(0.0474247, -0.0853209, 0.548369), 
        viewOffsetX=-0.0980256, viewOffsetY=-0.0514114)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.43924, 
        farPlane=4.41859, width=3.27259, height=1.46609, cameraPosition=(
        -1.085, 0.344947, -2.73467), cameraUpVector=(-0.759979, 0.156158, 
        0.630909), cameraTarget=(0.083499, -0.0667263, 0.707725), 
        viewOffsetX=-0.0939011, viewOffsetY=-0.0492482)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.27419, 
        farPlane=4.55609, width=3.05116, height=1.36689, cameraPosition=(
        -2.02659, 1.35531, -1.89207), cameraUpVector=(-0.514532, -0.132839, 
        0.847119), cameraTarget=(0.153087, -0.140889, 0.636826), 
        viewOffsetX=-0.0875474, viewOffsetY=-0.0459159)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.61944, 
        farPlane=4.34662, width=3.51436, height=1.5744, cameraPosition=(
        -0.204271, 3.44284, 0.988799), cameraUpVector=(-0.227562, -0.472197, 
        0.851613), cameraTarget=(-0.0378442, -0.173476, 0.460126), 
        viewOffsetX=-0.100838, viewOffsetY=-0.0528864)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.58525, 
        farPlane=4.39242, width=3.4685, height=1.55385, cameraPosition=(
        0.0581192, 3.43086, 1.13198), cameraUpVector=(-0.108425, -0.492857, 
        0.863329), cameraTarget=(-0.0438788, -0.167254, 0.477659), 
        viewOffsetX=-0.0995219, viewOffsetY=-0.0521961)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.65821, 
        farPlane=4.32805, width=3.56639, height=1.59771, cameraPosition=(
        -0.00912424, 3.4886, 0.681901), cameraUpVector=(-0.0129791, -0.376331, 
        0.926394), cameraTarget=(-0.0345002, -0.165973, 0.513356), 
        viewOffsetX=-0.102331, viewOffsetY=-0.0536692)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.52201, 
        farPlane=4.47537, width=3.38366, height=1.51584, cameraPosition=(
        0.454735, 3.38526, 1.25867), cameraUpVector=(-0.109675, -0.515603, 
        0.849779), cameraTarget=(-0.058858, -0.15257, 0.480906), 
        viewOffsetX=-0.0970877, viewOffsetY=-0.0509192)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.24563, 
        farPlane=4.58131, width=3.01285, height=1.34973, cameraPosition=(
        -1.91138, 2.07218, 2.42628), cameraUpVector=(0.179326, -0.841217, 
        0.510095), cameraTarget=(0.0749023, -0.204862, 0.363594), 
        viewOffsetX=-0.0864481, viewOffsetY=-0.0453391)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.6063, 
        farPlane=4.32251, width=3.49676, height=1.56651, cameraPosition=(
        0.225736, 3.43966, 0.861348), cameraUpVector=(-0.0977197, -0.408176, 
        0.907658), cameraTarget=(-0.105323, -0.189429, 0.537278), 
        viewOffsetX=-0.100333, viewOffsetY=-0.0526211)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.51723, 
        farPlane=4.39645, width=3.37726, height=1.51298, cameraPosition=(
        -0.230811, 3.31766, 1.44893), cameraUpVector=(-0.0503033, -0.568309, 
        0.821276), cameraTarget=(-0.0799608, -0.212517, 0.500254), 
        viewOffsetX=-0.0969041, viewOffsetY=-0.0508228)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.452, 
        farPlane=4.45154, width=3.28974, height=1.47377, cameraPosition=(
        -0.557111, 3.20121, 1.66948), cameraUpVector=(0.0230493, -0.622544, 
        0.782246), cameraTarget=(-0.0567514, -0.2247, 0.487251), 
        viewOffsetX=-0.0943928, viewOffsetY=-0.0495057)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.65487, 
        farPlane=4.263, width=3.56192, height=1.59571, cameraPosition=(
        -0.141994, 3.4568, 0.579652), cameraUpVector=(0.0188305, -0.336033, 
        0.941662), cameraTarget=(-0.0801485, -0.201199, 0.568126), 
        viewOffsetX=-0.102203, viewOffsetY=-0.0536016)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.51802, 
        farPlane=4.39149, width=3.37833, height=1.51346, cameraPosition=(
        -0.4013, 3.36913, 1.15964), cameraUpVector=(0.0652962, -0.485522, 
        0.871782), cameraTarget=(-0.0628546, -0.219673, 0.534458), 
        viewOffsetX=-0.0969348, viewOffsetY=-0.0508386)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.55065, 
        farPlane=4.37357, width=3.42211, height=1.53307, cameraPosition=(
        0.0664774, 3.40972, -0.106093), cameraUpVector=(-0.0429254, -0.143289, 
        0.98875), cameraTarget=(-0.0958482, -0.176246, 0.600581), 
        viewOffsetX=-0.0981909, viewOffsetY=-0.0514974)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.40165, 
        farPlane=4.43078, width=3.2222, height=1.44351, cameraPosition=(
        -3.20959, 0.624374, 1.5011), cameraUpVector=(0.58893, 0.0340512, 
        0.807467), cameraTarget=(0.229737, -0.166185, 0.53625), 
        viewOffsetX=-0.092455, viewOffsetY=-0.0484891)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.50649, 
        farPlane=4.33403, width=3.36286, height=1.50653, cameraPosition=(
        0.188412, 3.37824, -0.0229715), cameraUpVector=(0.0665409, -0.165804, 
        0.983911), cameraTarget=(-0.14013, -0.20566, 0.634767), 
        viewOffsetX=-0.0964911, viewOffsetY=-0.0506059)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.50674, 
        farPlane=4.33388, width=3.3632, height=1.50668, cameraPosition=(
        0.142402, 3.38467, 0.00445638), cameraUpVector=(0.0676837, -0.172801, 
        0.982629), cameraTarget=(-0.13739, -0.208612, 0.632908), 
        viewOffsetX=-0.0965006, viewOffsetY=-0.0506109)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.62325, 
        farPlane=4.21737, width=2.16298, height=0.968992, 
        viewOffsetX=-0.210645, viewOffsetY=-0.28089)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.2301, 
        farPlane=4.21607, width=1.83881, height=0.823767, cameraPosition=(
        -2.63231, 1.64747, 1.40654), cameraUpVector=(0.448333, -0.326912, 
        0.831941), cameraTarget=(0.21213, -0.470427, 0.507315), 
        viewOffsetX=-0.179075, viewOffsetY=-0.238792)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.51716, 
        farPlane=3.98303, width=2.0755, height=0.929805, cameraPosition=(
        -3.20687, -0.515996, 0.775144), cameraUpVector=(0.37776, 0.0235271, 
        0.925605), cameraTarget=(0.432138, -0.182004, 0.599035), 
        viewOffsetX=-0.202126, viewOffsetY=-0.26953)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.45121, 
        farPlane=4.04898, width=2.75393, height=1.23373, viewOffsetX=-0.431536, 
        viewOffsetY=-0.230766)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.49186, 
        farPlane=4.23516, width=2.79961, height=1.2542, cameraPosition=(
        -3.31385, 0.559492, 0.799845), cameraUpVector=(0.374002, -0.0713018, 
        0.924683), cameraTarget=(0.238271, -0.296974, 0.615946), 
        viewOffsetX=-0.438694, viewOffsetY=-0.234594)
    a = mdb.models['test'].rootAssembly
    n1 = a.instances['merged_mesh_2-1'].nodes
    nodes1 = n1.getSequenceFromMask(mask=(
        '[#f5f75fdf #30fcf7f5 #3b3f0f0f #9e7e1e3d #3dcc661f #c3c3cc3f #beebfbcf', 
        ' #5776febe #1dd55555 #33 #600028cc #1986 #cc500 #ee330000', 
        ' #baaaaaaa #31d555db #23000 #0 #1f0 #f8000000 #0', 
        ' #7c0000 #0 #3e00 #0 #c201f #0 #3f0000', 
        ' #0 #fc0000 #0 #3f00000 #0 #fc00000 #0', 
        ' #3f000000 #31800 #aaaab8cc #555ddbba #cc775555 #30000000 #a3', 
        ' #7c000000 #0 #3e0000 #0 #1f00 #80000000 #f', 
        ' #7c00000 #0 #3e000 #cc5 #0 #f80 #0', 
        ' #3f00 #0 #fc00 #0 #3f000 #0 #fc000', 
        ' #0 #3f0000 #30cc0000 #30000003 #baaaaaee #555555dd #cc775', 
        ' #a330000 #662800 #30cc0000 #c0000003 #aaaabb8c #d7f6eeaa #3dfd7d77', 
        ' #c3c33c3f #dccb3bcf #f0cf0fc #cf0fcf3f #fef3f0f0 #bfafaefa #f ]', ), 
        )
    region = a.Set(nodes=nodes1, name='boundary_nodes')
    mdb.models['test'].boundaryConditions['fixed'].setValues(region=region)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=OFF, bcs=OFF, 
        predefinedFields=OFF, connectors=OFF)
    mdb.Job(name='test_merge_mesh', model='test', description='', type=ANALYSIS, 
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
        scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1, 
        numGPUs=0)


def section_ass():
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
    p = mdb.models['test'].parts['pyrite']
    c = p.cells
    cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
    region = regionToolset.Region(cells=cells)
    p = mdb.models['test'].parts['pyrite']
    p.SectionAssignment(region=region, sectionName='quartz', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)
    p = mdb.models['test'].parts['calcite']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models['test'].parts['calcite']
    c = p.cells
    cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
    region = regionToolset.Region(cells=cells)
    p = mdb.models['test'].parts['calcite']
    p.SectionAssignment(region=region, sectionName='feldspar', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)
    p = mdb.models['test'].parts['merged_mesh_2']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    a = mdb.models['test'].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON)
    session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
        meshTechnique=ON)
    p = mdb.models['test'].parts['merged_mesh_2']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.21216, 
        farPlane=4.71605, width=4.6081, height=2.06438, cameraPosition=(1.9748, 
        1.9998, 2.52541), cameraTarget=(-0.0252014, -0.000203911, 0.525405))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.26955, 
        farPlane=4.59088, cameraPosition=(1.59559, 0.56746, -2.48305), 
        cameraUpVector=(-0.342307, 0.866664, 0.362933), cameraTarget=(
        -0.0252016, -0.000203915, 0.525405))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.38784, 
        farPlane=4.47259, width=3.48983, height=1.56341, cameraPosition=(
        1.73229, 0.541054, -2.41439), cameraTarget=(0.111502, -0.02661, 
        0.594071))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.38426, 
        farPlane=4.29085, cameraPosition=(-0.726189, -0.260823, -2.74902), 
        cameraUpVector=(0.132588, 0.961339, 0.241346), cameraTarget=(0.13579, 
        -0.0186881, 0.597377))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.47057, 
        farPlane=4.25725, cameraPosition=(0.268753, 0.388236, -2.83234), 
        cameraUpVector=(0.143258, 0.879392, 0.454033), cameraTarget=(0.0980646, 
        -0.0432986, 0.600536))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=2.37305, 
        farPlane=4.30131, cameraPosition=(-0.602194, 0.39751, -2.75991), 
        cameraUpVector=(0.164879, 0.890013, 0.425079), cameraTarget=(0.124005, 
        -0.0435748, 0.598379))
    p = mdb.models['test'].parts['merged_mesh_2']
    e = p.elements
    elements = e.getSequenceFromMask(mask=('[#ffffffff:62 #ffff ]', ), )
    region = regionToolset.Region(elements=elements)
    p = mdb.models['test'].parts['merged_mesh_2']
    p.SectionAssignment(region=region, sectionName='feldspar', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)


