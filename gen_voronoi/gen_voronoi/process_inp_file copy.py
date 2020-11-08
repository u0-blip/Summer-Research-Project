import numpy as np
from copy import deepcopy
from shutil import copyfile
import os
import argparse

input_f_name = r'v_500.inp'

parser = argparse.ArgumentParser(description='Process input files')

parser.add_argument('-f', '--file', default=input_f_name,
                    help='the file name to process')
args = vars(parser.parse_args())

input_f_name = args['file']

working_dir = r'C:/peter_abaqus/Summer-Research-Project/abaqus_working_space/v500/'
# abaqus_dir = r'C:/temp/'
abaqus_dir = r'C:/peter_abaqus/Summer-Research-Project/meep/gen_geo/gmsh_out/'

mat_data_dir = r'C:/peter_abaqus/Summer-Research-Project/meep/gen_geo/material.data'
processed_f_name = input_f_name[:-4] + '_processed7' + input_f_name[-4:]


def post_processing():
    os.chdir(working_dir)
    command = 'abaqus job=' + processed_f_name[:-4]
    print(command)
    os.system(command)

    
def processing():
    copyfile(abaqus_dir + input_f_name, working_dir + input_f_name)
    f = open(working_dir + input_f_name, 'r')
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
    nodes = '*NODE'

    assembly_id = '*Assembly'
    part_id = '*Part'
    step_id = 'Step'

    out_lines = []

    start_section = 0
    record_first_line_section = False

    insert_sec_placeholder = False
    insert_mat_placeholder = False

    node = []

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
            elif nodes in line:
                first_field = nodes
            elif assembly_id in line:
                first_field = assembly_id
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
                out_lines.append(line)
            else:
                index = line.index('C3D4')
                out_line = line[:index] + 'C3D4T'+ line[index+4:]
                out_lines.append(out_line)
        elif first_field == element_id and element_type != 'C3D4':
            pass
        elif first_field == nodes and line[1].isnumeric():
            param = line.split(',')
            node.append({int(param[0]): np.array(param[1:])})
            print(param)
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
        


    # num_seeds = 410
    # np.random.seed(15)
    # num_section = 2
    # assignment = np.random.randint(0, num_section, [num_seeds])

    # section_string = ['*Solid Section, elset=PSOLID_', '***', ', material=', 
    # '***']
    # all_section_string = []

    # for i in range(num_seeds):
    #     i_section_string = deepcopy(section_string)
    #     i_section_string[1] = str(i+1)
    #     i_section_string[-1] = 'feldspar' if assignment[i] == 0 else 'quartz'
    #     s = ''.join(i_section_string)
    #     all_section_string.append(s)

    # start_section = out_lines.index(section_id+'placeholder\n')
    # out_lines.pop(start_section)
    # for i in range(num_seeds):
    #     j = num_seeds - i -1
    #     out_lines.insert(start_section, all_section_string[j]+'\n')

    # start_mat = out_lines.index(material_id+'placeholder\n')
    # out_lines.pop(start_mat)
    # for i in range(len(mat_data)):
    #     j = len(mat_data) - i -1
    #     if j == len(mat_data)-1:
    #         out_lines.insert(start_mat, mat_data[j]+'\n')
    #     else:
    #         out_lines.insert(start_mat, mat_data[j])

    # shell_index = out_lines.index('*Elset, elset=PSHELL_0, generate\n')
    # for i in range(2):
    #     out_lines.pop(shell_index)

    # for i, l in enumerate(out_lines):
    #     if 'PSHELL_' in l and '**' in l and 'Section' in l:
    #         shell_index = i
    #         break
    # for j in range(3):
    #     out_lines.pop(shell_index)


    # step_index = out_lines.index('*Step, name=Step-1, nlgeom=NO\n')
    # step_index += 1
    # l = out_lines[step_index]
    # while '**' not in l:
    #     out_lines.pop(step_index)
    #     l = out_lines[step_index]

    assem = '*Assembly, name=Assembly\n\n*Instance, name=PART-1-1, part=PART-1\n*End Instance\n*End Assembly\n'
    out_lines.insert(len(out_lines), assem)

    for i in range(len(mat_data)):
        j = len(mat_data) - i -1
        if j == len(mat_data)-1:
            out_lines.insert(len(out_lines), mat_data[j]+'\n')
        else:
            out_lines.insert(len(out_lines), mat_data[j])

    num_steps = 2
    max_steps = 1.
    sim_step = '*Step, name=Step-1, nlgeom=NO\n*Coupled Temperature-displacement, creep=none, deltmx=30.\n1., '+ str(num_steps) + '., 0.01, '+str(max_steps) + '\n*Restart, write, frequency=0\n*Output, field, variable=PRESELECT\n*Output, history, variable=PRESELECT\n*End Step'
    # out_lines.insert(step_index, sim_step)
    out_lines.insert(len(out_lines), sim_step)
    # step_index+=1

    # out_lines.insert(step_index, '*Dflux\n')
    # step_index += 1
    # for i in range(len(assignment)):
    #     if assignment[i] == 1:
    #         out_lines.insert(step_index, ' PART-1-1.PSOLID_'+ str(i+1) + ', BF, 10e6.\n')
    #         step_index += 1

    fout.writelines(out_lines)


    f.close() 
    fout.close() 

if __name__ == '__main__':
    processing()
    # post_processing()
