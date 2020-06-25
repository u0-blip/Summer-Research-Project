config = None
import os

class translate:
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

def plot_f_name(_config): 
    global config
    config = _config
    name_var = [('geo', 'shape'), ('geo', 'particle_size'), ('geo', 'distance'), ('geo', 'x_loc'), ('source', 'fcen'), ('geo', 'fill_factor'), ('geo', 'rotation')]
    name_discr = ['', 'r', 'gap', 'xloc', 'fcen', 'ff', 'rt']
    name = ''
    counter = 0
    for discr, var in zip(name_discr, name_var):
        if counter == 0:
            name += discr + config.get(*var)
        else:
            name += '_' + discr + '_' + config.get(*var)
        counter += 1

    return os.getcwd() + '/output/export/3/' + name

def get_offset(data):
    global config
    if config.get('geo', 'shape_types') == 'checker':
        offset_index = np.unravel_index(np.argmax(data), data.shape)
        offset = [off/data.shape[i]*cell_size[i] -
                  cell_size[i]/2 for i, off in enumerate(offset_index)]
    else:
        offset = get_array('visualization', 'viz_offset')
        offset_index = [int((off+cell_size[i]/2)/cell_size[i]*data.shape[i])
                        for i, off in enumerate(offset)]
    return offset, offset_index