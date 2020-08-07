import cmath
import numpy as np
import pandas as pd
from time import time
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from my_meep.config.configs import *
from my_meep.source import source_wrapper
from my_meep import gen_geo_helper
import meep as mp

def create_sim(geo):
    if config.getboolean('boundary', 'pml'):
        pml_layers = [mp.PML(1)]
    else:
        pml_layers = []

    source = source_wrapper()

    r = config.getfloat('geo', 'particle_size')
    res = config.getfloat('sim', 'resolution')
    if config.getboolean('sim', 'change_res'):
        _res = 3.4/r+40
    else:
        _res = res

    config.set('sim', 'resolution', str(_res))

    S = 0.5
    # the out_every need to be changed so that the it outputs the same steps
    1/_res


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
            material_function=gen_geo_helper.pass_vor(gen_geo_helper.my_eps, geo)
        )
    else:
        raise Exception('One of the option must be specified')

    return sim

def start_sim(geo, eps_data):
    """ run the simulation and decide whether to use flux calculation """
    ez_data = None

    pre = time()

    sim = create_sim(geo=geo)
    sim.init_sim()

    ez_data = sim_run_wrapper(sim, eps_data)

    if verbals:
        after = time()
        print('Time for meep sim: ', after - pre)
    return ez_data

def sim_run_wrapper(sim, eps_data):
    """ returns the integrated RMS over time in average of 1 time unit """

    out_every = config.getfloat('sim', 'out_every')
    time = config.getfloat('sim', 'time')
    save_every = int(config.getfloat('sim', 'save_every'))
    arr_dim = eps_data.shape
    out_every = config.getfloat('sim', 'out_every')
    dim = int(config.getfloat('sim', 'dimension'))

    start_factor = 10
    start = int(cell_size[0]*start_factor/out_every)
    # print('start record time is: ', start)
    counter = 0
    overall_count = 0
    ttt = int(time/out_every)
    # print('end record time is: ', ttt)
    if start > ttt:
        print('insufficient start: ')
        start = round(ttt/2)

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
    ref = ref_to_vars()
    
    def f1(sim):
        f(ref, sim)
                
    sim.run(mp.at_every(out_every, f1), until=config.getfloat('sim', 'time'))
    
    return ez_data_inted/((ttt-start)*out_every)
