
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
# import helpers
# reload(helpers)
# from helpers import *

# import global_var 
# reload(global_var)
# from global_var import *

import create_material
reload(create_material)
from create_material import *

def delete():
    for key in model.materials.keys():
        del model.materials[key]
    for key in model.sections.keys():
        del model.sections[key]
for i in range(len(mdb.models['mesh_500'].parts['PART-1'].sectionAssignments)):
    try:
        del(mdb.models['mesh_500'].parts['PART-1'].sectionAssignments[i])
    except:
        continue

def create():
    create_material_section()

num_seeds = 410
np.random.seed(15)
num_section = 2
assignment = np.random.randint(0, num_section, [num_seeds])
p = mdb.models['mesh_500'].parts['PART-1']

def re_assign():
    for key in p.sets.keys():
        if 'PSOLID' not in key:
            continue
        else:
            num = unicode(key[7:])
            if not num.isnumeric():
                print('error with key: '+key)
                continue
            else:
                num = int(num)-1
                region = p.sets[key]
                if assignment[num] == 0:
                    secion_name = 'feldspar'
                elif assignment[num] == 1:
                    secion_name = 'quartz'
                else:
                    print('invalid assignemnt')

                p.SectionAssignment(region=region, sectionName=secion_name, 
                    offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', 
                    thicknessAssignment=FROM_SECTION)

if __name__ == '__main__':
    delete()
    create()
    re_assign()
    
# run on TeamViewer
# shutil.copyfile('C:/Users/dche145/abaqusMacros.py', r'C:/peter_abaqus/Summer-Research-Project/abaqus_macro/macro.py')
# shutil.copyfile('C:/temp/v500.inp', r'C:/peter_abaqus/Summer-Research-Project/abaqus_working_space/v500/v500.inp')
# shutil.copyfile('C:/temp/umat_test.inp', r'C:/peter_abaqus/Summer-Research-Project/abaqus_working_space/abaqus_out/umat_test.inp')
# execfile('C:/peter_abaqus/Summer-Research-Project/abaqus_macro/setup_after_import_mesh.py', __main__.__dict__)
# os.chdir(r"C:\peter_abaqus\Summer-Research-Project\abaqus_working_space\abaqus_out")
