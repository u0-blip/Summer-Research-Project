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
from animator import my_animate, my_rms_plot, plot_3d
import traceback
import logging
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

X = np.arange(-cell_size[0]/2, cell_size[0]/2, 1/res)
Y = np.arange(-cell_size[1]/2, cell_size[1]/2, 1/res)
X, Y = np.meshgrid(X, Y)
plot_f_name = '_'.join(['/mnt/c/peter_abaqus/Summer-Research-Project/output/export/3/', config.get('geo', 'shape'), config.get('geo', 'particle_radius'), config.get('geo', 'distance')]) 

def index2coord(index, size_arr, size_geo):
    index = (index/size_arr - 0.5)*size_geo
    return index


def gen_checker(size_cell, dist, dim=2):
    checker_coord = []
    xaxis = np.linspace(-size_cell[0], size_cell[0], int(dist))
    yaxis = np.linspace(-size_cell[1], size_cell[1], int(dist))
    zaxis = np.linspace(-size_cell[2], size_cell[2], int(dist))

    bounding_box = cell_size/2 + np2mp(get_array('geo', 'solid_center'))
    print(bounding_box)
    for i in range(int(dist)):
        if xaxis[i] > bounding_box[0]:
            continue
        for j in range(int(dist)):
            if dim == 2:
                checker_coord.append(mp.Vector3(xaxis[i], yaxis[j]))
            elif dim == 3:
                for k in range(int(dist)):
                    checker_coord.append(mp.Vector3(
                        xaxis[i], yaxis[j], zaxis[k]))
    return checker_coord


def create_sim(geo):

    if config.getboolean('boundary', 'pml'):
        pml_layers = [mp.PML(0.5)]
    else:
        pml_layers = []

    source = source_wrapper()

    if type_s == 'checker' or type_s == 'shape':
        sim = mp.Simulation(
            resolution=res,
            cell_size=cell_size,
            boundary_layers=pml_layers,
            sources=source,
            geometry=geo,
            default_material=mp.Medium(epsilon=1)
        )
    elif type_s == 'voronoi':
        sim = mp.Simulation(
            resolution=res,
            cell_size=cell_size,
            boundary_layers=pml_layers,
            sources=source,
            material_function=pass_vor(my_eps, geo)
        )
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

    if sim_dim == 2:
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
                      component=mp.Ez,
                      center=center,
                      size=size)
        ]

    elif mode == 'gaussian':
        tilt_angle = math.radians(config.getfloat(
            'source', 'tilt_angle'))  # angle of tilted beam
        k = mp.Vector3(x=2).rotate(mp.Vector3(z=1), tilt_angle).scale(fcen)
        sigma = config.getfloat('source', 'sigma')  # beam width
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


def get_shapes(type_s, particle_size, sim_dim, geo, cell_size):

    # particle_size[1] = dist
    # particle_size = np.array([1, 1, mp.inf])*(size_cell[0]/(dist-1))

    radius = config.getfloat('geo', 'particle_radius')
    shape = config.get('geo', 'shape')

    if shape == 'cube':
        s = np2mp(particle_size)
    elif shape == 'sphere':
        s = radius
    elif shape == 'hexagon':
        s = particle_size
    elif shape == 'triangle':
        s = particle_size
    else:
        print('not a available shape')
        s = particle_size

    coords = []

    if type_s == 'checker':
        coords = gen_checker(cell_size, config.get('geo', 'spacing'), sim_dim)
        geo.append(
            mp.Block(
                cell_size,
                center=np2mp(get_array('geo', 'solid_center')),
                material=mp.Medium(epsilon=7.1)
            )
        )

    geo = create_simple_geo(geo, coords, shape=shape, particle_size=s,
                            prism_height=0.2, prism_axis=mp.Vector3(0, 0, 1))
    return geo


def get_flux_region(sim):
    nfreq = 100

    # reflected flux
    refl_fr = mp.FluxRegion(center=np2mp(get_array(
        'source', 'near_flux_loc')), size=np2mp(get_array('source', 'flux_size')))
    frs = []
    frs.append(mp.FluxRegion(
        center=mp.Vector3(4, 0, 0),
        size=mp.Vector3(0, 9, 0)
    ))
    frs.append(mp.FluxRegion(
        center=mp.Vector3(-4.5, 0, 0),
        size=mp.Vector3(0, 9, 0)
    ))
    frs.append(mp.FluxRegion(
        center=mp.Vector3(0, 4.5, 0),
        size=mp.Vector3(9, 0, 0)
    ))
    frs.append(mp.FluxRegion(
        center=mp.Vector3(0, -4.5, 0),
        size=mp.Vector3(9, 0, 0)
    ))

    flux_width = config.getfloat('source', 'flux_width')

    # the following side are added in a clockwise fashion

    side = [sim.add_flux(fcen, flux_width*fcen, nfreq, fr) for fr in frs]

    # transmitted flux

    return sim, side



def get_fluxes(sim, basic_sim, sides, basic_sides):
    sim_run_wrapper(basic_sim)

    basic_trans_flux_data = basic_sim.get_flux_data(basic_sides[0])
    basic_trans_flux_mag = mp.get_fluxes(basic_sides[0])

    # get rid of the straight transmitted data to get the reflected data
    sim.load_minus_flux_data(sides[0], basic_trans_flux_data)

    ez_data = sim_run_wrapper(sim)

    trans_flux_mag = [np.array(mp.get_fluxes(side)) for side in sides]
    trans_flux_mag = np.array(trans_flux_mag)

    flux_freqs = np.array(mp.get_flux_freqs(sides[0]))
    wave_len = 1/flux_freqs

    normalise_tran = trans_flux_mag[1]/basic_trans_flux_mag
    loss = (basic_trans_flux_mag -
            np.sum(trans_flux_mag[1:], axis=0))/basic_trans_flux_mag
    reflected = -trans_flux_mag[0]/basic_trans_flux_mag

    if mp.am_master():
        fig = plt.figure()
        plt.plot(wave_len, reflected, 'bo-', label='reflectance')
        plt.plot(wave_len, normalise_tran, 'ro-', label='transmittance')
        plt.plot(wave_len, loss, 'go-', label='loss')
        ax = plt.gca()
        # plt.axis([-np.inf, np.inf, 0, 1])
        ax.set_ylim([0, 1])
        plt.legend(loc="center right", fontsize=20)
        plt.xlabel("wavelength (μm)")
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.savefig('_'.join(['/mnt/c/peter_abaqus/Summer-Research-Project/output/export/3/', config.get(
            'geo', 'shape'), config.get('geo', 'particle_radius')]) + '_flux.png', dpi=300, bbox_inches='tight')

    return ez_data

sphere_mean_std_value = []
def gen_geo_and_sim(vor):
    # dimensions = mp.CYLINDRICAL
    if config.getboolean('visualization', 'structure') or config.getboolean('general', 'perform_mp_sim'):
        if sim_dim == 2:
            particle_size[2] = mp.inf
            cell_size[2] = 0

        if config.getboolean('boundary', 'metallic'):
            geo = [mp.Block(cell_size, material=mp.metal)]
            s = cell_size-0.5
            if sim_dim == 2:
                s[2] = 0
            geo.append(mp.Block(s, material=mp.air))
        else:
            geo = []

        if type_s == 'shape' or type_s == 'checker':
            geo = get_shapes(type_s, particle_size, sim_dim, geo, cell_size)
        else:
            geo = vor

        sim = create_sim(geo=geo)

        if config.getboolean('sim', 'calc_flux'):
            basic_sim = create_sim(geo=[])
            basic_sim, basic_sides = get_flux_region(basic_sim)
            sim, sides = get_flux_region(sim)

        sim.init_sim()
        eps_data = sim.get_array(
            center=mp.Vector3(), size=cell_size, component=mp.Dielectric)

    ez_data = None
    ez_trans = None
    if config.getboolean('general', 'perform_mp_sim'):
        pre = time()
        if not config.getboolean('sim', 'calc_flux'):
            ez_data = sim_run_wrapper(sim)
        else:
            ez_data = get_fluxes(sim, basic_sim, sides, basic_sides)
        if verbals:
            after = time()
            print('Time for meep sim: ', after - pre)

        out_every = config.getfloat('sim', 'out_every')
        start = int(cell_size[0]*2/out_every*3)
        end = len(ez_data) - 1
        if start >= end:
            print('Time interval is not sufficient')
            start = end - 20

        ez_data = np.array(ez_data)

        if config.getboolean('visualization', 'transiant'):
            ez_trans = ez_data

        ez_data = np.power(ez_data, 2)
        ez_data = np.trapz(ez_data[start:end], dx=out_every, axis=0)/(config.getfloat('sim','time')/config.getfloat('sim', 'out_every'))

        print('Time period for RMS: ', [start, end])

        print('The RMS matrix shape: ' + str(ez_data.shape))
        write_windows(ez_data, data_dir + project_name + '.eps')
        write_windows(eps_data, data_dir + project_name + '.eps')

    if config.getboolean('visualization', 'transiant') or config.getboolean('visualization', 'rms'):
        viz_res(ez_data, ez_trans)

    if config.getboolean('visualization', 'structure'):
        viz_struct(sim, sim_dim, eps_data)


def sim_run_wrapper(sim):
    ez_data = []

    def f(sim):
        ez_data.append(sim.get_array(component=mp.Ez))

    out_every = config.getfloat('sim', 'out_every')
    if not config.getboolean('sim', 'calc_flux'):
        sim.run(mp.at_every(out_every, f),
                until=config.getfloat('sim', 'time'))
    else:
        pt = np2mp(get_array('source', 'near_flux_loc'))
        sim.run(mp.at_every(out_every, f),
                until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, pt, 1e-3))
    return ez_data


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


def get_offset(data):
    if type_s == 'checker':
        offset_index = np.unravel_index(np.argmax(data), data.shape)
        offset = [off/data.shape[i]*cell_size[i] -
                  cell_size[i]/2 for i, off in enumerate(offset_index)]
        print(offset)
    else:
        offset = get_array('visualization', 'viz_offset')
        offset_index = [int((off+cell_size[i]/2)/cell_size[i]*data.shape[i])
                        for i, off in enumerate(offset)]
    return offset, offset_index


def viz_res(ez_data, ez_trans):
    if not config.getboolean('general', 'perform_mp_sim'):
        try:
            ez_data = read_windows(
                data_dir + project_name + '.mpout').transpose()
            ez_data = np.moveaxis(ez_data, -1, 0)
        except Exception:
            logging.error(traceback.format_exc())
            logging.log(1, e)
    else:
        ez_data = ez_data.transpose()

    eps = read_windows(data_dir + project_name + '.eps').transpose()
    print('particle area is ', np.mean(eps))

    if sim_dim == 2:
        if np.max(eps) - np.min(eps) > 0.1:
            trans = translate(np.min(eps), np.max(eps), 0, 254)
            vtrans = np.vectorize(trans)
            eps = vtrans(eps).astype(np.uint8)
            import cv2
            eps_edges = cv2.Canny(eps, 10, 20).astype(np.bool)
        else:
            eps_edges = []
        eps_rock = eps >= 7.4

        if config.getboolean('visualization', 'transiant') and config.getboolean('general', 'perform_mp_sim'):
            ez_trans = ez_trans.transpose()
            ez_trans = np.moveaxis(ez_trans, -1, 0)
            my_animate(ez_trans, window=1)

        if config.getboolean('visualization', 'rms'):
            out_every = config.getfloat('sim', 'out_every')
            time_sim = config.getfloat('sim', 'time')
            cell_size = get_array('geo', 'cell_size')

            if len(ez_data.shape) == 3:
                start = int(cell_size[0]*2/out_every*3)

                # 3 is to ensure the slower wave in the medium fully propogate
                end = len(ez_data) - 1
                if start >= end:
                    print('Time interval is not sufficient')
                    start = end - 20

                print('Time period for RMS: ', [start, end])
                ez_data[-2, eps_edges] = np.max(ez_data)*len(ez_data)/20
                my_rms_plot(ez_data, 0, 'rms', [start, end])

            elif len(ez_data.shape) == 2:
                view_only_particles = config.getboolean(
                    'visualization', 'view_only_particles')
                if view_only_particles:
                    ez_data[np.logical_not(eps_rock)] = 0
                else:
                    ez_data[eps_edges] = np.max(ez_data)*0.8

                fig = plt.figure(figsize=(8, 6))

                cbar_scale = get_array('visualization', 'cbar_scale')

                if not view_only_particles:
                    ax = plt.axes()
                    graph = plt.pcolor(
                        X, Y, ez_data, vmin=cbar_scale[0], vmax=cbar_scale[1])                
                    cb = fig.colorbar(graph, ax=ax)
                    cb.set_label(label='E^2 (V/m)^2',
                                size='xx-large', weight='bold')
                    cb.ax.tick_params(labelsize=20)
                else:
                    cbar_scale /= 6
                    ax = fig.gca(projection='3d')
                    trans = translate(-cell_size[0]/2, cell_size[0]/2, 0, ez_data.shape[0])
                    ax_lim = [-1, 1, -2, 2]
                    ax_index_lim = [int(trans(ele)) for ele in ax_lim]

                    mean = np.trapz(ez_data, axis=0)
                    mean = np.trapz(mean, axis=0)
                    area = np.sum(eps_rock)
                    mean = mean /area
                    std = np.std(ez_data[eps_rock])

                    # plt.figure()
                    # plt.plot(ez_data[eps_rock])
                    # plt.show()

                    print('mean', mean, 'std', std, area)
                    sphere_mean_std_value.append([mean, std])

                    # sphere_point_value.append(point_v)
                    # ax_index_lim = [tuple(range(ax_index_lim[0], ax_index_lim[1])), tuple(range(ax_index_lim[2], ax_index_lim[3]))]
                    # plt.axis(ax_lim)
                    # print(ez_data[ax_index_lim].shape)

                    graph = ax.plot_surface(X[ax_index_lim[0]:ax_index_lim[1], ax_index_lim[2]:ax_index_lim[3]], Y[ax_index_lim[0]:ax_index_lim[1], ax_index_lim[2]:ax_index_lim[3]], ez_data[ax_index_lim[0]:ax_index_lim[1], ax_index_lim[2]:ax_index_lim[3]], cmap=cm.coolwarm, linewidth=0, antialiased=False)


                ax.tick_params(axis='both', which='major', labelsize=20)

                plt.savefig(plot_f_name + '_rms.png', dpi=300, bbox_inches='tight')

                # plt.show()

    else:
        offset, offset_index = get_offset(ez_data)
        plot_3d(ez_data, offset, offset_index)
        plt.savefig(plot_f_name + '3D_rms.png', dpi=300, bbox_inches='tight')


def viz_struct(sim, sim_dim, eps_data):
    if sim_dim == 2:
        plt.figure()
        sim.plot2D()
        plt.savefig(plot_f_name + 'struct.png', dpi=300, bbox_inches='tight')
    else:
        offset, offset_index = get_offset(eps_data)
        plot_3d(eps_data, offset, offset_index)


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


def process_meep_arr(arr=None):
    if arr == None:
        arr = read_windows(
            data_dir + config.get('process_inp', 'project_name')+'.mpout')

    clean = get_ms(arr, 5)
    print(clean.shape)
    print(np.amax(clean))
    write_windows(clean, data_dir +
                  config.get('process_inp', 'project_name')+'.clean')


def win_main():
    if config.getboolean('general', 'gen_gmsh'):
        if verbals:
            last = time()
        voronoi_create(display=True, to_out_f=True, to_mesh=True, in_geo=None)

        if verbals:
            now = time()
            print('Create input file, time: ' + str(now - last))
            last = now

    if config.getboolean('general', 'process_inp'):
        if verbals:
            last = time()
        processing()
        if verbals:
            now = time()
            print('Process input file, time: ' + str(now - last))
            last = now

    if config.getboolean('general', 'sim_abq'):
        if verbals:
            before = time()
        abaqusing()
        if verbals:
            after = time()
            elapsed = after-before
            print('Time for Abaqusing is ' + str(elapsed))

    if config.getboolean('general', 'clean_array'):
        if verbals:
            before = time()
        process_meep_arr()
        if verbals:
            after = time()
            elapsed = after-before
            print('Time for clean array is ' + str(elapsed))


def wsl_main():
    global dist
    if verbals:
        last = time()

    if config.get('sim', 'type') == 'voronoi':
        if config.getboolean('general', 'gen_vor'):
            _, complete_vor, geo = b_voronoi(to_out_geo=True)
        else:
            with open(config.get('process_inp', 'posix_data') + config.get('process_inp', 'project_name') + '.vor', 'rb') as f:
                _, complete_vor = pickle.load(f)

        vor = complete_vor

        if verbals:
            now = time()
            print('Created voronoi geo, time: ' + str(now - last))
            last = now
    else:
        vor = None

    distance = get_array('geo', 'distance')
    dist_vec = np.linspace(distance[0], distance[1], distance[2])
    for i in range(len(dist_vec)):
        dist = dist_vec[i]
        config.set('geo', 'distance', str(dist_vec[i]))
        gen_geo_and_sim(vor)
    plt.figure()
    plt.plot(sphere_mean_std_value)
    plt.savefig(plot_f_name + '_diff_dist.png', dpi=300, bbox_inches='tight')
    write_windows(sphere_mean_std_value, plot_f_name+'.std')


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
