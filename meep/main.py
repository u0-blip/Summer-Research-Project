import sys
from meep_funcs import *
from gen_geo.bounded_voronoi import *
from gen_geo.simple_geo_and_arg import *
import random
from configs import config
import math
import cmath


def index2coord(index, size_arr, size_geo):
    index = (index/size_arr - 0.5)*size_geo
    return index


# def D3(size_cell):
#     perf_sim = True

#     # b_voronoi(n_towers = 500, seed=15)

#     coords = get_coord(dist)
#     size_solid = np.array([1, 1, 1])*float(args['size'])
#     radius = 0.1
#     prism_coords = get_prism_coord(dist, radius)


#     # geo = create_simple_geo(coords, shape="cube", size_solid=size_solid)
#     # geo = create_simple_geo(prism_coords, shape="prism", size_solid=size_solid, prism_height=0.2, prism_axis=mp.Vector3(0, 0, 1))
    
#     sim = create_sim(mode='eps', geo=my_eps, size_cell = size_cell, my_voronoi_geo = my_voronoi_geo, res = 175, wavelength = 6.6, dim=3)
#     # sim = create_sim(mode='geo', geo=geo, my_voronoi_geo=None, size_cell=size_cell, res=75)

#     sim.init_sim()
#     eps_data = sim.get_array(center=mp.Vector3(), size=size_cell, component=mp.Dielectric)

#     if not perf_sim:
#         size = eps_data.shape
#         slice_eps = eps_data[round(size[0]/2), :, :]
#         plt.pcolormesh(slice_eps)
#         plt.show()
#     else:
#         get_sim_output(str(args['file']), sim, length_t = 20, out_every=0.6, get_3_field=False)

#     write_windows(eps_data, file_name+'.eps')

#     # out_num_geo('checker_geo.bin', my_checker_geo, range_geo=[1.0,1.0,1.0], range_index=[100,100,100])
    

def gen_checker(size_cell, dist, dim = 2):
    checker_coord = []
    xaxis = np.linspace(-size_cell[0], size_cell[0], int(dist))
    yaxis = np.linspace(-size_cell[1], size_cell[1], int(dist))
    zaxis = np.linspace(-size_cell[2], size_cell[2], int(dist))

    for i in range(int(dist)):
        if xaxis[i] > 0.3:
            continue
        for j in range(int(dist)):
            if dim == 2:
                checker_coord.append([xaxis[i], yaxis[j]])
            elif dim == 3:
                for k in range(int(dist)):
                    checker_coord.append([xaxis[i], yaxis[j], zaxis[k]])
    return checker_coord


def create_sim(mode, geo, my_voronoi_geo, size_cell, res = 50, wavelength=1, dim=3):
    if config.getboolean('boundary', 'pml'):
        pml_layers = [mp.PML(0.5)]
    else:
        pml_layers = []


    # my_checker_geo = checker_geo()

    # print(type(center))
    # print(size)

    source = source_wrapper()
    # gen_polygon_data()
    # print(pass_vor(geo, my_voronoi_geo)((0.5, 0.5, 0.5)))
    if mode == 'geo':
        sim = mp.Simulation(resolution=res,
                    cell_size=size_cell,
                    boundary_layers=pml_layers,
                    sources = source,
                    geometry=geo,
                    default_material=mp.Medium(epsilon=7.1))
    elif mode == 'eps':
        sim = mp.Simulation(resolution=res,
            cell_size=size_cell,
            boundary_layers=pml_layers,
            sources = source,
            material_function=pass_vor(geo, my_voronoi_geo))
    else:
        raise Exception('One of the option must be specified')
    # vis(sim)
    return sim

def gaussian_beam(sigma, k, x0):
    def _gaussian_beam(x):
        return cmath.exp(1j*2*math.pi*k.dot(x-x0)-(x-x0).dot(x-x0)/(2*sigma**2))
    return _gaussian_beam

def np2mp(vec):
    return mp.Vector3(vec[0], vec[1], vec[2])

def source_wrapper():
    s = get_array('source', 'size')
    center = get_array('source', 'center')

    print(center)
    if sim_dim==2:
        s[0] = 0

    size = np2mp(s)
    center = np2mp(center)

    fcen = config.getfloat('source', 'fcen') # center frequency of CW source (wavelength is 1 μm)
    mode = config.get('source', 'mode')
    if mode == 'normal':
        return [
            mp.Source(mp.ContinuousSource(fcen, fwidth=0.2*fcen),
                    component= mp.Ez,
                    center=center,
                    size=size)
                    ]

    elif mode == 'gaussian':
        tilt_angle = math.radians(config.getfloat('source', 'tilt_angle')) # angle of tilted beam
        k = mp.Vector3(x=2).rotate(mp.Vector3(z=1),tilt_angle).scale(fcen)
        sigma = config.getfloat('source','sigma') # beam width
        # src_pt = mp.Vector3(y=4) # if you change center, you have to change phase and weird shit like that
        return [mp.Source(src=mp.ContinuousSource(fcen, fwidth=0.2*fcen),
                        component=mp.Ez,
                        center=center,
                        size=size,
                        amp_func=gaussian_beam(sigma,k,center))]
    else:
        return None


def sim_wrapper(size_cell, size_solid, per_sim = False):

    # size_solid[1] = dist
    # size_solid = np.array([1, 1, mp.inf])*(size_cell[0]/(dist-1))

    radius = config.getfloat('geo', 'particle_radius')
    shape = config.get('geo', 'shape')

    if shape=='cube':
        s = size_solid
    elif shape == 'sphere':
        s = radius
    elif shape == 'hexagon':
        s = size_solid
        coords = get_polygon_coord(dist, radius)
    elif shape == 'triangle':
        s = size_solid
        coords = get_polygon_coord(dist, radius)
    else:
        print('not a available shape')
        s = size_solid
        coords = []

    if config.get('sim','type') == 'checker':
        coords = gen_checker(size_cell, config.get('geo','spacing'), dim)

    # geo = create_simple_geo(coords, shape=shape, size_solid=s)
    if config.getboolean('boundary', 'metallic'):
        geo = [mp.Block(size_cell, material=mp.metal)]
        s = size_cell-0.5
        if sim_dim == 2:
            s[2] = 0
        geo.append(mp.Block(s, material=mp.air))
    else:
        geo = []

    geo = create_simple_geo(geo, coords, shape=shape, size_solid=s, prism_height=0.2, prism_axis=mp.Vector3(0, 0, 1))
    

    # sim = create_sim(mode='eps', geo=my_eps, my_voronoi_geo = my_voronoi_geo, res = 50)
    sim = create_sim(mode='geo', geo=geo, my_voronoi_geo=None, size_cell=size_cell, res=config.getfloat('sim', 'resolution'), wavelength = 1, dim=sim_dim)

    sim.init_sim()
    eps_data = sim.get_array(center=mp.Vector3(), size=size_cell, component=mp.Dielectric)


    if per_sim:
        sim_run_wrapper(file_name, sim, length_t = config.getfloat('sim', 'time'), out_every=0.6, get_3_field=False)
        write_windows(eps_data, file_name+'.eps')
        vis_res(sim)

    viz_struct(sim)
    plt.show()
    # out_num_geo('checker_geo.bin', my_checker_geo, range_geo=[1.0,1.0,1.0], range_index=[100,100,100])

def sim_run_wrapper(f_name, sim, length_t=20, out_every=0.6, get_3_field = False):
    if get_3_field:
        result = [[] for i in range(3)]
        @static_vars(counter=0)
        def f(sim):
            result[0].append(sim.get_efield_x()) 
            result[1].append(sim.get_efield_y())  
            result[2].append(sim.get_efield_z())   
            # f.counter += 1
            # if f.counter%5 == 0:
            #     arr = np.array(result)
            #     write_windows(arr, f_name+str(f.counter))
            #     write_windows(arr.shape, f_name+str(f.counter)+'.meta')
    else:
        result = []
        @static_vars(counter=0)
        def f(sim):
            result.append(sim.get_array(component=mp.Ez))
            # f.counter += 1
            # if f.counter%5 == 0:
            #     arr = np.array(result)
            #     write_windows(arr, f_name+str(f.counter))
            #     write_windows(arr.shape, f_name+str(f.counter)+'.meta')

    sim.run(mp.at_every(out_every, f), until=length_t)

    result = np.array(result)

    write_windows(result, f_name)

    print('The output shape of the result matrix is: ' + str(result.shape))
    return result

def vis_res(sim):
    ez_data = sim.get_array(component=mp.Ez)
    plt.figure()
    plt.title('beam freq: ' + config.get('source','fcen'))
    plt.imshow(np.flipud(np.transpose(np.real(ez_data))), interpolation='spline36', cmap='RdBu')
    plt.axis('off')


def viz_struct(sim):
    viz = config.getboolean('sim', 'visualize')
    if viz:
        plt.figure()
        if sim_dim == 2:
            sim.plot2D()
        else:
            size = eps_data.shape
            index = np.unravel_index(eps_data.argmax(), eps_data.shape)
            plot_axis = config.get('sim', '3d_plotting_axis')

            if plot_axis == 'x':
                slice_eps = eps_data[index[0], :, :]
                plt.xlabel('y')
                plt.ylabel('z')
            elif plot_axis == 'y':
                slice_eps = eps_data[:, index[1],  :]
                plt.xlabel('x')
                plt.ylabel('z')
            else:
                slice_eps = eps_data[:, :, index[2]]
                plt.xlabel('x')
                plt.ylabel('y')

            plt.title('print slice ' + str(index[0]))
            
            plt.pcolormesh(slice_eps)
# class config_wrapper(configparser.ConfigParser):
#     def __init__(self, f_name, section):
#         super().__init__()
#         self.read(filenames = f_name)
#         self.config = self[section]

#     def get_array(self, name):
#         val = self.config[name]
#         val = np.fromstring(val, dtype = np.float, sep=',')
#         return val
#     def get_types(self, name):
#         val = self.config[name]
#         return val.split(',')
#     def val(self, name):
#         val = self.config[name]
#         if '.' in val and val[0].isnumeric():
#             return float(val)
#         elif val.isnumeric():
#             return int(val)
#         else:
#             return val
#     def getbool(self, name):
#         return self['simple_geo.sim'].getboolean(name)

def get_array(section,  name, type = np.float):
    val = config.get(section, name)
    val = np.fromstring(val, dtype = type, sep=',')
    return val

if __name__ == "__main__":

    # dimensions = mp.CYLINDRICAL

    file_name = config.get('file', 'file_out')
    dist = config.getfloat('geo', 'distance')
    per_sim = config.getboolean('sim', 'perform_sim')
    
    # random.seed(15)
    # v_seed_points = np.random.rand(500, 3) - 0.5
    # vor = Voronoi(v_seed_points)
    # my_voronoi_geo = voronoi_geo(num_seeds=500, vor=vor)

    sim_dim = config.getint('sim', 'dimension')
    size_solid = get_array('geo', 'particle_size')
    size_cell = get_array('geo', 'cell_size')

    if sim_dim == 2:
        size_solid[2] = mp.inf
        size_cell[2] = 0

    sim_wrapper(size_cell, size_solid,  per_sim=per_sim)

    # D3(size_cell = np.array([1, 1, 1])*1.8)

