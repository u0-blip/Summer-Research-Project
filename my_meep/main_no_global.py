import sys
import os
import multiprocessing as mp

sys.path.append('/mnt/c/peter_abaqus/Summer-Research-Project/')

from gen_geo.bounded_voronoi import b_voronoi, bounding_box
import random
from my_meep.configs import *
import math
import cmath
import numpy as np
from time import time
import pickle
from my_meep.meep_funcs import read_windows, write_windows
from my_meep.animator import my_animate, my_rms_plot, plot_3d
import traceback
import logging
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import contextlib
import io

sphere_mean_std_value = []
total_area = []


last = time()
verbals = config.getboolean('general', 'verbals')
if os.name == 'nt':
    from gen_geo.gmsh_create import voronoi_create
    from gen_geo.process_inp_file import processing, abaqusing
elif os.name == 'posix':
    import my_meep.meep_funcs as mf
    import gen_geo.simple_geo_and_arg as simp
    import meep as mp

class DummyFile(object):
    def write(self, x): pass
    def flush(self): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

@contextlib.contextmanager
def silence_stdout():
    old_target = sys.stdout
    try:
        with open(os.devnull, "w") as new_target:
            sys.stdout = new_target
            yield new_target
    finally:
        sys.stdout = old_target

def plot_f_name(): 
    name_var = [('geo', 'shape'), ('geo', 'particle_size'), ('geo', 'distance'), ('geo', 'x_loc'), ('source', 'fcen'), ('geo', 'fill_factor')]
    name_discr = ['', 'r', 'gap', 'xloc', 'fcen', 'ff']
    name = ''
    for discr, var in zip(name_discr, name_var):
        name += '_' + discr + '_' + config.get(*var)
    return '/mnt/c/peter_abaqus/Summer-Research-Project/output/export/3/' + name

def index2coord(index, size_arr, size_geo):
    index = (index/size_arr - 0.5)*size_geo
    return index


def create_sim(geo):
    if config.getboolean('boundary', 'pml'):
        pml_layers = [mp.PML(1)]
    else:
        pml_layers = []

    source = source_wrapper()

    r = config.getfloat('geo', 'particle_size')
    res = config.getfloat('sim', 'resolution')
    if type_s == 'checker' and config.getboolean('sim', 'change_res'):
        _res = 3.4/r+40
    else:
        _res = res

    config.set('sim', 'resolution', str(_res))
    print('res is ', _res)
    S = 0.5
    # the out_every need to be changed so that the it outputs the same steps
    dx = 1/_res


    if type_s == 'checker' or type_s == 'shape':
        sim = mp.Simulation(
            resolution=_res,
            cell_size=cell_size,
            boundary_layers=pml_layers,
            sources=source,
            geometry=geo,
            Courant=S,
            default_material=mp.Medium(epsilon=1)
        )
    elif type_s == 'voronoi':
        sim = mp.Simulation(
            resolution=_res,
            cell_size=cell_size,
            boundary_layers=pml_layers,
            sources=source,
            Courant=S,
            material_function=pass_vor(my_eps, geo)
        )
    else:
        raise Exception('One of the option must be specified')

    return sim


def gaussian_beam(sigma, k, x0):
    def _gaussian_beam(x):
        return cmath.exp(1j*2*math.pi*k.dot(x-x0)-(x-x0).dot(x-x0)/(2*sigma**2))
    return _gaussian_beam


def source_wrapper():
    s = get_array('source', 'size')
    center = get_array('source', 'center')
    dim = config.getfloat('sim', 'dimension')
    if sim_dim == 2:
        s[0] = 0

    size = mf.np2mp(s)
    
    center = mf.np2mp(center)
    if dim == 3:
        size.z = size.y

    mode = config.get('source', 'mode')
    fwidth = config.getfloat('source', 'fwidth')
    if not config.getboolean('sim', 'calc_flux'):
        pre_source = mp.ContinuousSource(
            fcen, fwidth=fwidth*fcen, is_integrated=True)
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


def get_flux_region(sim):
    nfreq = 100

    # reflected flux
    refl_fr = mp.FluxRegion(center=mf.np2mp(get_array(
        'source', 'near_flux_loc')), size=mf.np2mp(get_array('source', 'flux_size')))
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
            'geo', 'shape'), config.get('geo', 'particle_size')]) + '_flux.png', dpi=300, bbox_inches='tight')

    return ez_data

def cut(cell_size, center, eps, geo):
    for i in range(len(center)):
        geo.append(
            mp.Block(
                cell_size,
                center=mf.np2mp(center[i]),
                material=mp.Medium(epsilon=eps)
            )
        )

def gen_geo_and_sim(vor):
    # dimensions = mp.CYLINDRICAL
    if config.getboolean('visualization', 'structure') or config.getboolean('general', 'perform_mp_sim') or True:
        # use air to cut the boundary
        center = np.array([
            [9, 0, 0],
            [0, 9, 0],
            [-9, 0, 0],
            [0, -9, 0],
            [0, 0, 9],
            [0, 0, -9]
        ])
        center_matrix = center/9*8
        ff = config.getfloat('geo', 'fill_factor')
        effictive_median = (ff*np.sqrt(7.69)+(1-ff)*np.sqrt(7.1))**2
        center[0, 0] = 8


        if sim_dim == 2:
            cell_size[2] = 0

        if config.getboolean('boundary', 'metallic'):
            geo = [mp.Block(cell_size, material=mp.metal)]
            s = cell_size-0.5
            if sim_dim == 2:
                s[2] = 0
            geo.append(mp.Block(s, material=mp.air))
        else:
            geo = []

        if config.get('sim', 'type') == 'checker':
            cut(cell_size, [mp.Vector3(-2, 0, 0)], 7.1, geo)
            # cut(cell_size, center_matrix, effictive_median, geo)

        if type_s == 'shape' or type_s == 'checker':
            geo = mf.create_simple_geo(geo, prism_height=0.2, prism_axis=mp.Vector3(0, 0, 1))
        else:
            geo = vor

        cut(cell_size, center, 1, geo)

        eps_sim = create_sim(geo=geo)

        if config.getboolean('sim', 'calc_flux'):
            basic_sim = create_sim(geo=[])
            basic_sim, basic_sides = get_flux_region(basic_sim)
            sim, sides = get_flux_region(sim)
        
        eps_sim.eps_averaging = False
        with silence_stdout():
            eps_sim.init_sim()

        global eps_data
        global total_area

        eps_data = eps_sim.get_array(
            center=mp.Vector3(), 
            size=cell_size, 
            component=mp.Dielectric
            )
        total_area.append(get_area(eps_data))
        
    ez_data = None
    ez_trans = None
    if config.getboolean('general', 'perform_mp_sim'):
        pre = time()
        sim = create_sim(geo=geo)
        with silence_stdout():
            sim.init_sim()
        if not config.getboolean('sim', 'calc_flux'):
            ez_data = sim_run_wrapper(sim)
            print('mean ez is: ', np.mean(ez_data))
        else:
            ez_data = get_fluxes(sim, basic_sim, sides, basic_sides)
        if verbals:
            after = time()
            print('Time for meep sim: ', after - pre)

        if config.getboolean('visualization', 'transiant'):
            ez_trans = ez_data

        print('The RMS matrix shape: ' + str(ez_data.shape))


        if config.getfloat('sim', 'dimension') == 3:
            write_windows(ez_data, plot_f_name() + '_3D.ez')
        # write_windows(eps_data, data_dir + project_name + '.eps')
        get_mean_std(ez_data, ez_trans, eps_data)

    if config.getboolean('visualization', 'transiant') or config.getboolean('visualization', 'rms'):
        viz_res(ez_data, ez_trans, eps_data)

    if config.getboolean('visualization', 'structure'):
        viz_struct(eps_sim, sim_dim, eps_data)
    del eps_sim


def sim_run_wrapper(sim):
    """ returns the integrated RMS over time in average of 1 time unit """

    out_every = config.getfloat('sim', 'out_every')

    start_factor = 10
    start = int(cell_size[0]*start_factor/out_every)
    print('start record time is: ', start)
    time = config.getfloat('sim', 'time')
    ttt = int(time/out_every)
    print('end record time is: ', ttt)
    if start > ttt:
        print('insufficient start: ')
        start = round(ttt/2)

    save_every = int(config.getfloat('sim', 'save_every'))
    arr_dim = eps_data.shape
    out_every = config.getfloat('sim', 'out_every')
    dim = int(config.getfloat('sim', 'dimension'))
    counter = 0
    overall_count = 0


    ez_data_inted = np.zeros(arr_dim)
    ez_data = np.empty(
        np.array((save_every, *arr_dim)))

    if dim == 3:
        def ass (ez_data, ass_ed, counter): ez_data[counter, :, :, :] = ass_ed
    elif dim == 2:
        def ass (ez_data, ass_ed, counter): ez_data[counter, :, :] = ass_ed

    def f(ref, sim):
        if ref.overall_count >= ref.start:
            ref.ass(ref.ez_data, sim.get_array(component=mp.Ez), ref.counter)
            ref.counter += 1
        if ref.counter == save_every:
            ref.counter = 0
            ref.ez_data = np.power(ref.ez_data, 2)
            ref.ez_data_inted += np.trapz(ref.ez_data, dx=out_every, axis=0)
        if ref.overall_count == ref.ttt-1 and counter != 0:
            ref.ez_data = np.power(ref.ez_data, 2)
            ref.ez_data_inted += np.trapz(ref.ez_data[:counter-1], dx=out_every, axis=0)
        ref.overall_count+= 1
        
    class ref_to_vars:
        def __init__(self):
            self.ez_data_inted = ez_data_inted
            self.ez_data = ez_data
            self.counter = counter
            self.overall_count = overall_count
            self.start = start
            self.ass = ass
            self.ttt = ttt
        ez_data_inted, ez_data, counter, overall_count, start, ass
    ref = ref_to_vars()
    
    def f1(sim):
        f(ref, sim)

    if not config.getboolean('sim', 'calc_flux'):
        sim.run(mp.at_every(out_every, f1),
                until=config.getfloat('sim', 'time'))
    else:
        pt = mf.np2mp(get_array('source', 'near_flux_loc'))
        sim.run(mp.at_every(out_every, f1),
                until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, pt, 1e-3))
    
    return ez_data_inted/((ttt-start)*out_every)

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
    else:
        offset = get_array('visualization', 'viz_offset')
        offset_index = [int((off+cell_size[i]/2)/cell_size[i]*data.shape[i])
                        for i, off in enumerate(offset)]
    return offset, offset_index



def get_area(eps_data):
    eps_rock = eps_data > 7.69
    area = np.sum(eps_rock)
    return area/np.prod(eps_data.shape)

def get_mean_std(ez_data, ez_trans, eps):
    global sphere_mean_std_value

    s = eps.shape
    cell_size = get_array('geo', 'cell_size')

    res_block = [-3, 3, -3, 3]
    t = translate(-cell_size[0]/2, cell_size[1]/2, 0, s[0])
    res_block_index = [int(t(ele)) for ele in res_block]
    
    res_block_mat = np.zeros_like(eps)
    res_block_mat[res_block_index[0]:res_block_index[1], res_block_index[2]:res_block_index[3]] = 1

    eps_rock = np.logical_and((eps > 7.69), res_block_mat)


    ez_data[np.logical_not(eps_rock)] = 0

    res = config.getfloat('sim', 'resolution')

    mean = np.trapz(ez_data, axis=0, dx=1/res)
    mean = np.trapz(mean, axis=0, dx=1/res)
    dim = config.getfloat('sim', 'dimension')

    if dim == 3:
        mean = np.trapz(mean, axis=0, dx=1/res)

    area = np.sum(eps_rock)
    std = np.std(ez_data[eps_rock])

    print('mean', mean, 'std', std, 'area', area/ez_data.shape[0]**2)

    sphere_mean_std_value.append([mean, std])


def viz_res(ez_data, ez_trans, eps):
    res = config.getfloat('sim', 'resolution')
    X = np.arange(-cell_size[0]/2, cell_size[0]/2, 1/res)
    Y = np.arange(-cell_size[1]/2, cell_size[1]/2, 1/res)
    X, Y = np.meshgrid(X, Y)
    if not config.getboolean('general', 'perform_mp_sim'):
        try:
            ez_data = read_windows(
                data_dir + project_name + '.mpout').transpose()
            ez_data = np.moveaxis(ez_data, -1, 0)
            eps = read_windows(data_dir + project_name + '.eps').transpose()
        except Exception:
            logging.error(traceback.format_exc())
            logging.log(1, e)
    else:
        ez_data = ez_data.transpose()

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
        eps_rock = eps >= 7.69

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
                    trans = translate(-cell_size[0]/2,
                                      cell_size[0]/2, 0, ez_data.shape[0])
                    ax_lim = [-1, 1, -2, 2]
                    ax_index_lim = [int(trans(ele)) for ele in ax_lim]

                    graph = ax.plot_surface(X[ax_index_lim[0]:ax_index_lim[1], ax_index_lim[2]:ax_index_lim[3]], Y[ax_index_lim[0]:ax_index_lim[1], ax_index_lim[2]
                                            :ax_index_lim[3]], ez_data[ax_index_lim[0]:ax_index_lim[1], ax_index_lim[2]:ax_index_lim[3]], cmap=cm.coolwarm, linewidth=0, antialiased=False)

                ax.tick_params(axis='both', which='major', labelsize=20)

                plt.savefig(plot_f_name() + '_rms.png',
                            dpi=300, bbox_inches='tight')

                # plt.show()

    else:
        offset, offset_index = get_offset(ez_data)
        plot_3d(ez_data, offset, offset_index)
        # plt.show()
        plt.savefig(plot_f_name() + '3D_rms.png', dpi=300, bbox_inches='tight')


def viz_struct(sim, sim_dim, eps_data):
    if sim_dim == 2:
        plt.figure()
        sim.plot2D()
        plt.savefig(plot_f_name() + 'struct.png', dpi=300, bbox_inches='tight')
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

    shape_str = [ele.strip() for ele in config.get('geo', 'shape_types').split(',')]

    params = []
    param_names = ['shape', 'fill_factor', 'particle_size', 'x_loc', 'distance', 'std']
    param_cat = ['geo', 'geo', 'geo', 'geo', 'geo', 'geo', 'geo']

    for cat, name in zip(param_cat, param_names):
        temp = list(get_array(cat, name))
        temp[-1] = int(temp[-1])
        params.append(temp)

    total = np.prod([ele[-1] for ele in params])
    counter = 0
    from itertools import product

    for param_val in product(*[np.linspace(*ele) for ele in params]):
        for i, val in enumerate(param_val):
            if param_names[i] == 'shape':
                config.set(param_cat[i], param_names[i], shape_str[int(val)])
            else:
                config.set(param_cat[i], param_names[i], str(val))

        gen_geo_and_sim(vor)
        counter += 1
        print(counter, ' out of ', total, ' is done.')

    iter_vars_ele = []
    iter_vars_index = []
    for i, ele in enumerate(params):
        if ele[-1] > 1:
            iter_vars_ele.append(ele)
            iter_vars_index.append(i)

    iter_vars_ele = np.array(iter_vars_ele)
    iter_vars_size = iter_vars_ele[:, -1].astype(np.int)

    global sphere_mean_std_value
    if config.getboolean('general', 'perform_mp_sim'):
        sphere_mean_std_value = np.array(sphere_mean_std_value)

        sphere_mean_std_value = sphere_mean_std_value.reshape(*iter_vars_size, 2)
        
    to_write = {
        'param names': [param_names[i] for i in iter_vars_index], 
        'iter vals': iter_vars_ele, 
        'std mean' : sphere_mean_std_value,
        'area' : total_area
        }

    var_descrip_str = ''
    if len(iter_vars_ele) != 0:
        p = iter_vars_ele[0]
        i = iter_vars_index[0]
        for i, p in zip(iter_vars_index, iter_vars_ele):
            var_descrip_str += param_names[i] + '_' + str('_'.join([str(p) for p in params[i]]))

    with open(plot_f_name() + '_' + var_descrip_str + '.std', 'wb') as f:
        pickle.dump(to_write, f)


if __name__ == "__main__":
    if os.name == 'nt':
        win_main()
    elif os.name == 'posix':
        wsl_main()
