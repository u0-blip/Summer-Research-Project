# -*- coding: mbcs -*-
# Do not delete the following import lines


from abaqus import *
from abaqusConstants import *
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
import Visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior
from decimal import Decimal
import sys
import pickle
import numpy as np
import json
import os
import sys
import shutil
import unicodedata

# working_dir = r'//ad.monash.edu/home/User045/dche145/Documents/Abaqus/microwave-break-rocks/'
working_dir = r'C:/peter_abaqus/Summer-Research-Project/abaqus_macro'

sys.path.append(working_dir)
# os.chdir(working_dir)
import helpers
import mat_prop
reload(mat_prop)
from mat_prop import *

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

if __name__== "__main__":
    create_material_section()


# TeamViewer
# execfile('C:/peter_abaqus/Summer-Research-Project/abaqus_macro/create_material.py', __main__.__dict__)
# shutil.copyfile('C:/Users/dche145/abaqusMacros.py', r'//ad.monash.edu/home/User045/dche145/Documents/Abaqus/microwave-break-rocks/macro.py')
# os.chdir(r"C:\peter_abaqus\Summer-Research-Project")

# Citrix receiver
# execfile('//ad.monash.edu/home/User045/dche145/Documents/Abaqus/microwave-break-rocks/create_material.py', __main__.__dict__)
# shutil.copyfile('C:/Users/dche145/abaqusMacros.py', r'//ad.monash.edu/home/User045/dche145/Documents/Abaqus/microwave-break-rocks/macro.py')

