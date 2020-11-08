# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
import __main__

def rotate():
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
    import Visualization
    import xyPlot
    import displayGroupOdbToolset as dgo
    import connectorBehavior
    session.viewports['Viewport: 1'].view.setValues(nearPlane=11.5562, 
        farPlane=17.5367, width=0.944521, height=0.461161, viewOffsetX=1.27177, 
        viewOffsetY=-2.35087)
    a = mdb.models['square-3d'].rootAssembly
    a.rotate(instanceList=('Part-2-1', 'Part-3-1', 'merge-1'), axisPoint=(4.875, 
        -4.875, 0.125), axisDirection=(10.0, 0.0, 0.0), angle=90.0)
    a = mdb.models['square-3d'].rootAssembly
    a.rotate(instanceList=('Part-2-1', 'Part-3-1', 'merge-1'), axisPoint=(4.9375, 
        -4.75, 0.125), axisDirection=(0.0, -0.125, 0.0), angle=90.0)


def delete():
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
    import Visualization
    import xyPlot
    import displayGroupOdbToolset as dgo
    import connectorBehavior
    a = mdb.models['square-3d-macro-start-origin'].rootAssembly
    a.deleteFeatures(('merged-1', 'pyrite-0', 'pyrite-1', 'pyrite-2', 'pyrite-3', 
        'pyrite-4', 'pyrite-5', 'pyrite-6', 'pyrite-7', 'pyrite-8', 'pyrite-9', 
        'calcite-1', ))


