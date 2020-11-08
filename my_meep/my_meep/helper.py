import os
from my_meep.config.configs import get_array
from my_meep.config.config_variables import *

class Translate:
    def __init__(self, leftMin, leftMax, rightMin, rightMax):
        # Figure out how 'wide' each range is
        self.leftMin = leftMin
        self.leftMax = leftMax
        self.rightMin = rightMin
        self.rightMax = rightMax
        self.leftSpan = leftMax - leftMin
        self.rightSpan = rightMax - rightMin

    def __call__(self, value):
        # Convert the left range into a 0-1 range (float)
        valueScaled = float(value - self.leftMin) / float(self.leftSpan)

        # Convert the 0-1 range into a value in the right range.
        return self.rightMin + (valueScaled * self.rightSpan)

def output_file_name(config): 
    
    name_var = [('Geometry', 'shape'), ('Geometry', 'particle_area_or_volume'),  ('Source', 'fcen'), ('Geometry', 'fill_factor'), ('Geometry', 'rotation')]
    name_discr = ['', 'r', 'fcen', 'ff', 'rt']
    name = ''
    counter = 0
    for discr, var in zip(name_discr, name_var):
        if counter == 0:
            name += discr + config.get(*var)
        else:
            name += '_' + discr + '_' + config.get(*var)
        counter += 1

    return '/mnt/c/peter_abaqus/Summer-Research-Project/output/export/3/' + name

def get_offset(data):
    global config
    if config.get('Geometry', 'shape_types') == 'checker':
        offset_index = np.unravel_index(np.argmax(data), data.shape)
        offset = [off/data.shape[i]*cell_size[i] -
                  cell_size[i]/2 for i, off in enumerate(offset_index)]
    else:
        offset = get_array('Visualization', 'viz_offset')
        offset_index = [int((off+cell_size[i]/2)/cell_size[i]*data.shape[i])
                        for i, off in enumerate(offset)]
    return offset, offset_index