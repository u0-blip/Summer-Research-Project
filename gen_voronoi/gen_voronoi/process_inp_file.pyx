from copy import deepcopy
from shutil import copyfile
import os
import argparse
from time import time
import numpy as np
from gmsh_create import voronoi_create
import sys
import glob
sys.path.append('..')
import pickle

from configs import config



# parser = argparse.ArgumentParser(description='Process input files')

# parser.add_argument('-g', '--ingeo', default=input_geo_name,
#                     help='the file name to process')

# parser.add_argument('-i', '--outAbqInp', default=input_f_name,
#                     help='the file name to process')

# args = vars(parser.parse_args())


pj_name = config.get('process_inp', 'project_name')
input_f_name = pj_name + '.inp'
input_geo_name = pj_name + '.geo'
data_dir = config.get('process_inp', 'data')

working_dir = data_dir  + pj_name + 'abaqus/'
abaqus_dir = r'C:/peter_abaqus/Summer-Research-Project/meep/gen_geo/gmsh_out/'

mat_data_dir = r'C:/peter_abaqus/Summer-Research-Project/meep/gen_geo/material.data'

processed_f_name = pj_name + '_out.inp'
data_dir = config.get('process_inp', 'data')

def abaqusing():
    new_name = working_dir
    while os.path.exists(new_name):
        i = -1
        while not new_name[i].isnumeric():
            i -= 1
        j = i
        while new_name[j].isnumeric():
            j -= 1
        old_name = new_name
        new_name = new_name[:j+1] + str(int(new_name[j+1:i+1])+1) + new_name[i+1:]

    os.makedirs(new_name)
    
    for CleanUp in glob.glob(old_name + '*.*'):
        print CleanUp
        if not (CleanUp.endswith('dflux.for') or CleanUp.endswith('_out.inp') or CleanUp.endswith('.simdir')):    
            os.remove(CleanUp)
        if CleanUp.endswith('dflux.for') or CleanUp.endswith('_out.inp') :
            os.rename(CleanUp, new_name+CleanUp.split('\\')[-1])
            
    os.chdir(new_name)

    command = [r'call "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018.5.274\windows\bin\ifortvars.bat" intel64',
    r'call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x64',]
    command.append('abaqus job=' + processed_f_name + ' user=' + 'dflux' + ' interactive')

    with open('run.bat', 'w+') as run:
        run.writelines('\n'.join(command))

    os.system('run.bat')
    

def processing():

    if not os.path.exists(working_dir):
        os.makedirs(working_dir)


    last = time()

    verbals = config.getboolean('General', 'verbals')

    f = open(data_dir + input_f_name, 'r')
    fout = open(working_dir + processed_f_name, 'w+')

    mat_file = open(mat_data_dir, 'r')
    mat_data = mat_file.readlines()
    content = f.readlines()

    first_field = 'unknown'
    sec_field= 'unknown'

    section_id = '*Solid Section'
    elset_id = '*Elset'
    material_id = '*Material'
    element_id = '*ELEMENT'
    element_type = 'none'

    assembly_id = '*Assembly'
    part_id = '*Part'
    step_id = 'Step'
    nodes = '*NODE'

    out_lines = []
    node = []
    volumes = []
    start_section = 0
    record_first_line_section = False

    insert_sec_placeholder = False
    insert_mat_placeholder = False

    cdef int i,j, k

    
    for i, line in enumerate(content):
        if '*' in line and '**' not in line:
            if section_id in line:
                sec_field= section_id
                if record_first_line_section == False:
                    record_first_line_section = True
                    start_section = i
            elif elset_id in line:
                sec_field= elset_id
            elif element_id in line:
                first_field = element_id
                if 'S3R' in line:
                    element_type = 'S3R'
                elif 'C3D4' in line:
                    element_type = 'C3D4'
                else:
                    element_type = 'others'

            elif assembly_id in line:
                first_field= assembly_id
            elif nodes in line:
                first_field = nodes
            elif part_id in line:
                first_field = part_id
            elif step_id in line:
                first_field = step_id
            elif material_id in line:
                first_field= material_id
            elif first_field == material_id:
                pass
            else:
                sec_field= 'unknown'

        if line[0].isnumeric():
                line = ' ' + line

        if first_field == element_id and element_type == 'C3D4':
            if 'C3D4' not in line:
                # out_lines.append(line)
                nums = [int(ele) for ele in line.split(',')]
                volumes[-1].append(nums)
            else:
                # index = line.index('C3D4')
                # out_line = line[:index] + 'C3D4T'+ line[index+4:]
                # out_lines.append(out_line)
                volumes.append([])
        elif first_field == element_id and element_type != 'C3D4':
            pass
        elif first_field == nodes and line[1].isnumeric():
            param = line.split(',')
            param = [int(param[0]), float(param[1]), float(param[2]), float(param[3])]
            node.append(param[1:])
            if len(node) != param[0]:
                print('missing node')
        elif first_field != element_id:
            out_lines.append(line)
        
        
        
        # if not sec_field==section_id and not first_field==material_id and not(first_field==element_id and element_type=='S3R' and element_type=='other'):
        #     if 'C3D4' not in line:
        #         out_lines.append(line)
        #     else:
        #         index = line.index('C3D4')
        #         out_line = line[:index] + 'C3D4T\n'+ line[index+4:]
        #         out_lines.append(out_line)
        # elif sec_field == section_id and insert_sec_placeholder == False:
        #     insert_sec_placeholder = True
        #     out_lines.append(section_id+'placeholder\n')
        # elif first_field == material_id and insert_mat_placeholder == False:
        #     insert_mat_placeholder = True
        #     out_lines.append(material_id + 'placeholder\n')
        
    if verbals: 
        now = time()
        print('Input file done, time: ' + str(now - last))
        last = now

    dup_sets = []
    for i in range(len(node)):
        for j in range(i):
            if node[i] == node[j]:
                # print(str(i) + ' is the same as ' + str(j))
                dup_sets.append([i,j])
                
    unique_sets = []

    for i in range(len(dup_sets)):
        found = False
        for j, sets in enumerate(unique_sets):
            if dup_sets[i][0] in sets or dup_sets[i][1] in sets:
                sets.add(dup_sets[i][0])
                sets.add(dup_sets[i][1])
                found = True
                break
        if found:
            found = False
        else:
            unique_sets.append({dup_sets[i][0], dup_sets[i][1]})
    
    
    if verbals: 
        now = time()
        print('Finish duplicate sets, time: ' + str(now - last))
        last = now


    replaced_with = []
    for sets in unique_sets:
        replaced_with.append(sets.pop())

    cdef int num_unique_nodes = sum([len(item) for item in unique_sets])
    
    list_nodes = np.empty(num_unique_nodes, dtype=np.int)
    index_for_sets = np.empty(num_unique_nodes, dtype=np.int)
    cdef int counter = 0

    for i, s in enumerate(unique_sets):
        for n in s:
            list_nodes[counter] = n
            index_for_sets[counter] = i
            counter += 1

    sorted_arg = np.argsort(list_nodes)
    list_nodes = np.sort(list_nodes)
    
    cdef int index

    if verbals: 
        now = time()
        print('Nodes sorted, time: ' + str(now - last))
        last = now

    for v in volumes:
        for ele in v:
            for i, e in enumerate(ele[1:]):
                e_1base = e - 1
                j = i + 1
                #if e_1base not in hashed_nodes:
                    #continue
                index = np.searchsorted(list_nodes, e_1base, side='left')
                if index >= len(list_nodes) or index == 0:
                    continue
                if e_1base == list_nodes[index]:
                    ele[j] = replaced_with[index_for_sets[sorted_arg[index]]] + 1
        
    if verbals: 
        now = time()
        print('Switched out old index, time: ' + str(now - last))
        last = now


    for i, n in enumerate(node):
        # if i not in hashed_nodes:
        out_lines.append(' ' + str(i + 1) + ', ' + str(n[0]) + ', ' + str(n[1]) + ', ' + str(n[2]) + '\n')

    
    if verbals: 
        now = time()
        print('put the rest of index back in, time: ' + str(now - last))
        last = now


    for i, v in enumerate(volumes):
        out_lines.append('*ELEMENT, type=C3D4T, ELSET=Volume' + str(i+1) + '\n')
        for j, e in enumerate(v):
            line = ''
            for num in v:
                line += ' '
                for val in num:
                    line += str(val) + ', '
                line += '\n'
        out_lines.append(line)


    part = '*Part, name=PART-1\n'
    end_part = '*end part\n'
    out_lines.insert(2, part)

    section_string = ['*Solid Section, elset=Volume', '***', ', material=', 
    '***']
    all_section_string = []

    with open(config.get('process_inp', 'data') + config.get('process_inp', 'project_name') + '.partass', 'rb') as f:
        part_ass = pickle.load(f)

    num_seeds = len(part_ass)

    for i in range(num_seeds):
        i_section_string = deepcopy(section_string)
        i_section_string[1] = str(i+1)
        i_section_string[-1] = 'feldspar' if part_ass[i] == 0 else 'quartz'
        s = ''.join(i_section_string)
        all_section_string.append(s)

    out_lines.append('\n'.join(all_section_string))
    out_lines.append('\n')

    out_lines.append(end_part)

    assem = '*Assembly, name=Assembly\n\n*Instance, name=PART-1-1, part=PART-1\n*End Instance\n*End Assembly\n'
    out_lines.insert(len(out_lines), assem)

    mat_data = ''.join(mat_data)
    
    out_lines.insert(len(out_lines), mat_data)

    num_steps = 2
    max_steps = 1.
    sim_step = '\n*Step, name=Step-1, nlgeom=NO\n*Coupled Temperature-displacement, creep=none, deltmx=30.\n1., '+ str(num_steps) + '., 0.01, '+str(max_steps) + '\n'
    # out_lines.insert(step_index, sim_step)
    out_lines.insert(len(out_lines), sim_step)
    # step_index+=1


    start_place = len(out_lines)
    out_lines.insert(start_place, '*Dflux\n')
    start_place += 1
    for i in range(len(part_ass)):
        if part_ass[i] == 1:
            out_lines.insert(start_place, ' PART-1-1.Volume'+ str(i+1) + ', BFNU, 10e6.\n')
            start_place += 1

    init_cond = '** Name: initial_temp   Type: Temperature\n*Initial Conditions, type=TEMPERATURE\n_PickedSet1927, 20.'
    bound_cond = '*Boundary\n_G1951, ENCASTRE'
    
    sim_step = '*Restart, write, frequency=0\n*Output, field, variable=PRESELECT\n*Output, history, variable=PRESELECT\n*End Step'
    out_lines.append(sim_step)

    if verbals: 
        now = time()
        print('Added content, time: ' + str(now - last))
        last = now


    fout.writelines(out_lines)

    if verbals: 
        now = time()
        print('write output file, time: ' + str(now - last))
        last = now

    f.close() 
    fout.close() 

    if verbals: 
        now = time()
        print('close file, time: ' + str(now - last))
        last = now

def run():
    before = time()
    voronoi_create(mesh_size=None, display=False, ingeo=input_geo_name,out_f=input_f_name)
    after = time()
    elapsed = after-before
    print('Time for Create Mesh is ' + str(elapsed))

    before = time()
    processing(input_f_name)
    after = time()
    elapsed = after-before
    print('Time for processing input file is ' + str(elapsed))

    # before = time()
    # abaqusing()
    # after = time()
    # elapsed = after-before
    # print('Time for Abaqusing is ' + str(elapsed))


if __name__ == '__main__':
    run()
