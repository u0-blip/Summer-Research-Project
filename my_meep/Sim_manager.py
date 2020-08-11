import cmath
import numpy as np
import pandas as pd
from time import time
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from my_meep.config.configs import get_array
from my_meep.config.config_variables import *
from my_meep.source import source_wrapper
from my_meep import gen_geo_helper
import meep as mp
from copy import deepcopy

class Sim_manager:
    def get_eps(self):
        if self.eps_data is not None:
            return self.eps_data
        
        eps_sim_manager = Sim_manager(self.geo, self.config)
        eps_sim = eps_sim_manager.create_sim() 
        eps_sim.eps_averaging = False
        eps_sim.init_sim()
        self.eps_data = eps_sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Dielectric)
        return self.eps_data

    def get_config(self):
        self.particle_size = self.config.getfloat('geo', 'particle_size')
        self.res = self.config.getfloat('sim', 'resolution')
        

    def create_array_assign_func(self):
        if dim == 3:
            def assign (ez_data, ass_ed, counter): ez_data[counter, :, :, :] = ass_ed
        elif dim == 2:
            def assign (ez_data, ass_ed, counter): ez_data[counter, :, :] = ass_ed
        self.assign = assign

    def __init__(self, geo, config):
        self.sim = None
        self.eps_data = None
        self.config = config
        self.geo = geo
        self.get_config()
        self.create_array_assign_func()

    def create_sim(self):
        if self.sim is not None:
            return self.sim

        if pml:
            pml_layers = [mp.PML(1)]
        else:
            pml_layers = []

        source = source_wrapper(self.config)

        if change_res:
            _res = 3.4/self.particle_size+40
        else:
            _res = self.res

        self.config.set('sim', 'resolution', str(_res))

        S = 0.5

        sim_args = {
            'resolution':_res,
            'cell_size':cell_size,
            'boundary_layers':pml_layers,
            'sources':source,
            'Courant':S
        }

        if shape_type == 'checker' or shape_type == 'shape':
            self.sim = mp.Simulation(
                **sim_args,
                geometry=self.geo,
                default_material=mp.Medium(epsilon=1)
            )
        elif shape_type == 'voronoi':
            self.sim = mp.Simulation(
                **sim_args,
                material_function=gen_geo_helper.pass_vor(gen_geo_helper.my_eps, self.geo)
            )
        else:
            raise Exception('One of the option must be specified')
        return self.sim

    def __call__(self):
        """ run the simulation and decide whether to use flux calculation """
        pre = time()


        self.create_sim()
        self.get_eps()
        self.arr_dim = self.eps_data.shape

        self.sim.init_sim()
        self.calc_sim_time()
        self.ez_data_inted = np.zeros(self.arr_dim)
        self.ez_data = np.empty(np.array((save_every, *self.arr_dim)))
        
        def func_wrapper(sim):
            self.run_every(sim)
                    
        self.sim.run(mp.at_every(out_every, func_wrapper), until=sim_time)
        self.ez_data = self.ez_data_inted/((self.end_time-self.start)*out_every)

        if verbals:
            after = time()
            print('Time for meep sim: ', after - pre)
            
        return self.ez_data

    def calc_sim_time(self):
        self.start_factor = 10
        self.start = int(cell_size[0]*self.start_factor/out_every)
        # print('start record time is: ', start)
        self.counter = 0
        self.overall_count = 0
        self.end_time = int(sim_time/out_every)
        

        if self.start > self.end_time:
            print('insufficient start: ')
            self.start = round(self.end_time/2)


    def run_every(self, sim):
        if self.overall_count >= self.start:
            self.assign(self.ez_data, sim.get_array(component=mp.Ez), self.counter)
            self.counter += 1
        if self.counter == save_every:
            self.counter = 0
            self.ez_data = np.power(self.ez_data, 2)
            self.ez_data_inted += np.trapz(self.ez_data, dx=out_every, axis=0)
        if self.overall_count == self.end_time-1 and self.counter != 0:
            self.ez_data = np.power(self.ez_data, 2)
            self.ez_data_inted += np.trapz(self.ez_data[:self.counter-1], dx=out_every, axis=0)
        self.overall_count+= 1