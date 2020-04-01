import numpy as np
from copy import deepcopy

f = open(r'C:/peter_abaqus/Summer-Research-Project/abaqus_working_space/v500/test.inp', 'r')
fout = open(r'C:/peter_abaqus/Summer-Research-Project/abaqus_working_space/v500/test_out.inp', 'w+')
mat_file = open(r'C:/peter_abaqus/Summer-Research-Project/meep/gen_geo/material.data', 'r')

mat_data = mat_file.readlines()
content = f.readlines()

first_field = 'unknown'
sec_field= 'unknown'

section_id = '*Solid Section'
elset_id = '*Elset'
material_id = '*Material'
element_id = '*Element'
element_type = 'none'

assembly_id = '*Assembly'
part_id = '*Part'
step_id = 'Step'

out_lines = []

start_section = 0
record_first_line_section = False

insert_sec_placeholder = False
insert_mat_placeholder = False

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

        elif assembly_id in line:
            first_field= assembly_id
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
    if not sec_field==section_id and not first_field==material_id and not(first_field==element_id and element_type=='S3R'):
        if 'C3D4' not in line:
            out_lines.append(line)
        else:
            index = line.index('C3D4')
            out_line = line[:index] + 'C3D4T\n'
            out_lines.append(out_line)
    elif sec_field == section_id and insert_sec_placeholder == False:
        insert_sec_placeholder = True
        out_lines.append(section_id+'placeholder\n')
    elif first_field == material_id and insert_mat_placeholder == False:
        insert_mat_placeholder = True
        out_lines.append(material_id + 'placeholder\n')


num_seeds = 410
np.random.seed(15)
num_section = 2
assignment = np.random.randint(0, num_section, [num_seeds])

section_string = ['*Solid Section, elset=PSOLID_', '***', ', material=', 
'***']
all_section_string = []

for i in range(num_seeds):
    i_section_string = deepcopy(section_string)
    i_section_string[1] = str(i+1)
    i_section_string[-1] = 'feldspar' if assignment[i] == 0 else 'quartz'
    s = ''.join(i_section_string)
    all_section_string.append(s)

start_section = out_lines.index(section_id+'placeholder\n')
out_lines.pop(start_section)
for i in range(num_seeds):
    j = num_seeds - i -1
    out_lines.insert(start_section, all_section_string[j]+'\n')

start_mat = out_lines.index(material_id+'placeholder\n')
out_lines.pop(start_mat)
for i in range(len(mat_data)):
    j = len(mat_data) - i -1
    if j == len(mat_data)-1:
        out_lines.insert(start_mat, mat_data[j]+'\n')
    else:
        out_lines.insert(start_mat, mat_data[j])

shell_index = out_lines.index('*Elset, elset=PSHELL_0, generate\n')
for i in range(2):
    out_lines.pop(shell_index)



step_index = out_lines.index('*Step, name=Step-1, nlgeom=NO\n')
out_lines.insert(step_index+4, '*Dflux\n')
step_index += 5
for i in range(len(assignment)):
    if assignment[i] == 1:
        out_lines.insert(step_index, ' PART-1-1.PSOLID_'+ str(i+1) + ', BF, 100000.\n')
        step_index += 1

fout.writelines(out_lines)


f.close()
fout.close()