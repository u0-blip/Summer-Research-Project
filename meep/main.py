import sys
import os
from gen_geo.bounded_voronoi import b_voronoi, bounding_box
import random
from configs import *
import math
import cmath
import numpy as np
from time import time
import pickle
from meep_funcs import read_windows, write_windows
from animator import my_animate, my_rms_plot


def index2coord(index, size_arr, size_geo):
    index = (index/size_arr - 0.5)*size_geo
    return index

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
                checker_coord.append(mp.Vector3(xaxis[i], yaxis[j]))
            elif dim == 3:
                for k in range(int(dist)):
                    checker_coord.append(mp.Vector3(xaxis[i], yaxis[j], zaxis[k]))
    return checker_coord

def create_sim(geo):

    if config.getboolean('boundary', 'pml'):
        pml_layers = [mp.PML(0.5)]
    else:
        pml_layers = []

    source = source_wrapper()

    if sim_t == 'checker' or sim_t == 'shape':
        sim = mp.Simulation(resolution=res,
                    cell_size=cell_size,
                    boundary_layers=pml_layers,
                    sources = source,
                    geometry=geo,
                    default_material=mp.Medium(epsilon=7.1))
    elif sim_t == 'voronoi':
        sim = mp.Simulation(resolution=res,
            cell_size=cell_size,
            boundary_layers=pml_layers,
            sources = source,
            material_function=pass_vor(my_eps, geo))
    else:
        raise Exception('One of the option must be specified')
    
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

    mode = config.get('source', 'mode')
    fwidth = config.getfloat('source', 'fwidth')
    if not config.getboolean('sim', 'calc_flux'):
        pre_source = mp.ContinuousSource(fcen, fwidth=fwidth*fcen)
    else:
        pre_source = mp.GaussianSource(fcen, fwidth=fwidth*fcen)
    if mode == 'normal':
        return [
                    mp.Source(src=pre_source,
                    component= mp.Ez,
                    center=center,
                    size=size)
                ]

    elif mode == 'gaussian':
        tilt_angle = math.radians(config.getfloat('source', 'tilt_angle')) # angle of tilted beam
        k = mp.Vector3(x=2).rotate(mp.Vector3(z=1),tilt_angle).scale(fcen)
        sigma = config.getfloat('source','sigma') # beam width
        # src_pt = mp.Vector3(y=4) # if you change center, you have to change phase and weird shit like that

        return [
                    mp.Source(src=pre_source,
                    component=mp.Ez,
                    center=center,
                    size=size,
                    amp_func=gaussian_beam(sigma, k, center))
                ]
    else:
        return None

def sim_t_shape(sim_t, size_solid, sim_dim, geo, cell_size):

    # size_solid[1] = dist
    # size_solid = np.array([1, 1, mp.inf])*(size_cell[0]/(dist-1))

    radius = config.getfloat('geo', 'particle_radius')
    shape = config.get('geo', 'shape')

    if shape=='cube':
        s = np2mp(size_solid)
    elif shape == 'sphere':
        s = radius
    elif shape == 'hexagon':
        s = size_solid
    elif shape == 'triangle':
        s = size_solid
    else:
        print('not a available shape')
        s = size_solid
        
    coords = []

    if sim_t == 'checker':
        coords = gen_checker(cell_size, config.get('geo','spacing'), sim_dim)

    geo = create_simple_geo(geo, coords, shape=shape, size_solid=s, prism_height=0.2, prism_axis=mp.Vector3(0, 0, 1))
    return geo

def get_flux_region(sim):
    nfreq = 100
    
    # reflected flux
    refl_fr = mp.FluxRegion(center=np2mp(get_array('source', 'near_flux_loc')) ,size=np2mp(get_array('source', 'flux_size')))
    frs = []
    frs.append(mp.FluxRegion(
        center=mp.Vector3(4, 0, 0) ,
        size=mp.Vector3(0, 9, 0)
        ))
    frs.append(mp.FluxRegion(
        center=mp.Vector3(-4.5, 0, 0)  ,
        size=mp.Vector3(0, 9, 0)
        ))
    frs.append(mp.FluxRegion(
        center=mp.Vector3(0, 4.5, 0)  ,
        size=mp.Vector3(9, 0, 0)
        ))
    frs.append(mp.FluxRegion(
        center=mp.Vector3(0, -4.5, 0) ,
        size=mp.Vector3(9, 0, 0)
        ))

    flux_width = config.getfloat('source', 'flux_width')

    # the following side are added in a clockwise fashion
    
    side = [sim.add_flux(fcen, flux_width*fcen, nfreq, fr) for fr in frs]
    
    # transmitted flux

    return sim, side

def geo_sim(vor = None):
    # dimensions = mp.CYLINDRICAL
    if config.getboolean('visualization', 'structure') or config.getboolean('general', 'perform_mp_sim'):
        if sim_dim == 2:
            size_solid[2] = mp.inf
            cell_size[2] = 0

        if config.getboolean('boundary', 'metallic'):
            geo = [mp.Block(cell_size, material=mp.metal)]
            s = cell_size-0.5
            if sim_dim == 2:
                s[2] = 0
            geo.append(mp.Block(s, material=mp.air))
        else:
            geo = []

        if  sim_t ==  'shape' or sim_t == 'checker':
            geo = sim_t_shape(sim_t, size_solid, sim_dim, geo, cell_size)
        else:
            geo = vor

        sim = create_sim(geo=geo)
        
        if config.getboolean('sim', 'calc_flux'):
            basic_sim = create_sim(geo=[])
            basic_sim, basic_sides = get_flux_region(basic_sim)
            sim, sides = get_flux_region(sim)

        sim.init_sim()
        eps_data = sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Dielectric)
    
    if config.getboolean('visualization', 'structure'):
        viz_struct(sim, sim_dim, eps_data)

    if config.getboolean('general', 'perform_mp_sim'):
        if not config.getboolean('sim', 'calc_flux'):
            sim_run_wrapper(file_name, sim, length_t = config.getfloat('sim', 'time'), write_file = True)

        else:
            sim_run_wrapper(
                None, basic_sim, length_t = config.getfloat('sim', 'time'), write_file = False
                )
            basic_trans_flux_data = basic_sim.get_flux_data(basic_sides[0])
            basic_trans_flux_mag = mp.get_fluxes(basic_sides[0])

            # get rid of the straight transmitted data to get the reflected data
            sim.load_minus_flux_data(sides[0], basic_trans_flux_data)
            sim_run_wrapper(
                file_name, sim, length_t = config.getfloat('sim', 'time'), write_file = False
                )

            trans_flux_mag = [np.array(mp.get_fluxes(side)) for side in sides]
            trans_flux_mag = np.array(trans_flux_mag)

            flux_freqs = np.array(mp.get_flux_freqs(sides[0]))
            wave_len = 1/flux_freqs
            
            normalise_tran = trans_flux_mag[1]/basic_trans_flux_mag
            loss = (basic_trans_flux_mag - np.sum(trans_flux_mag[1:], axis=0))/basic_trans_flux_mag
            reflected = -trans_flux_mag[0]/basic_trans_flux_mag

            if mp.am_master():
                plt.figure()
                plt.plot(wave_len, reflected,'bo-',label='reflectance')
                plt.plot(wave_len, normalise_tran,'ro-',label='transmittance')
                plt.plot(wave_len,loss,'go-',label='loss')
                # plt.axis([5.0, 10.0, 0, 1])
                plt.xlabel("wavelength (μm)")
                plt.legend(loc="upper right")
                plt.show()

        write_windows(eps_data, file_name+'.eps')
        viz_res(sim, sim_dim)

    plt.show()

def sim_run_wrapper(f_name, sim, length_t=20, write_file=True):
    result = []
    def f(sim):
        result.append(sim.get_array(component=mp.Ez))

    out_every = config.getfloat('sim', 'out_every')
    if not config.getboolean('sim', 'calc_flux'):
        sim.run(mp.at_every(out_every, f), until=length_t)
    else:
        pt = np2mp(get_array('source', 'near_flux_loc'))
        sim.run(mp.at_every(out_every, f), until_after_sources=mp.stop_when_fields_decayed(50,mp.Ez, pt,1e-3))

    if write_file:
        result = np.array(result)
        write_windows(result, f_name)
        print('The output shape of the result matrix is: ' + str(result.shape))
    
    return result

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

def viz_res(sim, sim_dim):
    if not (config.getboolean('visualization', 'transiant') or config.getboolean('visualization', 'rms')):
        return
    
    if sim_dim == 2:
        ez_data = read_windows(data_dir + project_name + '.mpout')
        eps = read_windows(data_dir + project_name + '.mpout.eps')

        trans = translate(np.min(eps), np.max(eps), 0, 254)
        vtrans = np.vectorize(trans)
        eps = vtrans(eps).astype(np.uint8)
        import cv2
        eps_edges = cv2.Canny(eps,100,200).astype(np.bool)

        if config.getboolean('visualization', 'transiant'):
            my_animate(ez_data)

        if config.getboolean('visualization', 'rms'):
            out_every = config.getfloat('sim', 'out_every')
            time_sim = config.getfloat('sim', 'time')
            cell_size = get_array('geo', 'cell_size')
            start = int(cell_size[2]*2/out_every)
            end = int(time_sim/out_every) - 1
            print('Time period for RMS: ', [start, end])

            ez_data[-2, eps_edges] = np.max(ez_data)*len(ez_data)/20
            my_rms_plot(ez_data, 0, 'rms', [start, end])

    else:
        ez_data = sim.get_array(component=mp.Ez)
        s = ez_data.shape
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.title('xy')
        plt.pcolor(np.flipud(np.transpose(np.real(ez_data[:, :, round(s[2]/2)]))), interpolation='spline36', cmap='RdBu')
        plt.axis('off')
        plt.subplot(3, 1, 2)
        plt.title('xz')
        plt.pcolor(np.flipud(np.transpose(np.real(ez_data[:, round(s[2]/2), :]))), interpolation='spline36', cmap='RdBu')
        plt.axis('off')
        plt.subplot(3, 1, 3)
        plt.title('yz')
        plt.pcolor(np.flipud(np.transpose(np.real(ez_data[round(s[2]/2), :, :]))), interpolation='spline36', cmap='RdBu')
        plt.axis('off')

def viz_struct(sim, sim_dim, eps_data):
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

def wsl_main():
    if verbals: last = time()
    
    if config.get('sim', 'type') == 'voronoi':
        if config.getboolean('general', 'gen_vor'):
            vor, my_voronoi_geo, geo = b_voronoi(to_out_geo = True)
        else:
            with open(config.get('process_inp', 'posix_data') + config.get('process_inp', 'project_name') + '.vor', 'rb') as f:
                vor, my_voronoi_geo = pickle.load(f)
                
        vor = my_voronoi_geo
    else:
        vor = None

    if verbals: 
        now = time()
        print('created voronoi geo, time: ' + str(now - last))
        last = now

    geo_sim(vor)

from scipy.integrate import romb
def get_ms(arr, rms_interval):
    rms_shape = np.array(arr.shape)
    rms_shape[0] = np.floor(rms_shape[0]/rms_interval)
    rms_arr = np.zeros(rms_shape.astype(np.int))
    for i in range(0, rms_shape[0]):
        slice = arr[i*rms_interval:(i+1)*rms_interval]
        slice = slice**2
        int_slice = romb(slice, dx=1, axis=0)/(rms_interval-1)*10e26
        # rms_arr[i, :, :, :] = np.sqrt(np.mean(slice, axis=0))
        rms_arr[i, :, :, :] = int_slice
    return rms_arr

def process_meep_arr(arr = None):
    if arr == None:
        arr = read_windows(data_dir + config.get('process_inp', 'project_name')+'.mpout')
    
    clean = get_ms(arr, 5)
    print(clean.shape)
    print(np.amax(clean))
    write_windows(clean, data_dir + config.get('process_inp', 'project_name')+'.clean')

def win_main():
    if config.getboolean('general', 'gen_gmsh'):
        if verbals: last = time()
        voronoi_create(display = True, to_out_f = True, to_mesh = True, in_geo = None)

        if verbals: 
            now = time()
            print('Create input file, time: ' + str(now - last))
            last = now

    if config.getboolean('general', 'process_inp'):
        if verbals: last = time()
        processing()
        if verbals: 
            now = time()
            print('Process input file, time: ' + str(now - last))
            last = now

    if config.getboolean('general', 'sim_abq'):
        if verbals: before = time()
        abaqusing()
        if verbals: 
            after = time()
            elapsed = after-before
            print('Time for Abaqusing is ' + str(elapsed))

    if config.getboolean('general', 'clean_array'):
        if verbals: before = time()
        process_meep_arr()
        if verbals: 
            after = time()
            elapsed = after-before
            print('Time for clean array is ' + str(elapsed))


if __name__ == "__main__":
    last = time()
    verbals = config.getboolean('general', 'verbals')

    if os.name == 'nt':
        from gen_geo.gmsh_create import voronoi_create
        from gen_geo.process_inp_file import processing, abaqusing

        win_main()
    elif os.name == 'posix':
        from meep_funcs import *
        from gen_geo.simple_geo_and_arg import *
        wsl_main()