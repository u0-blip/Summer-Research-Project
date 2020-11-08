import cmath
import numpy as np
import pandas as pd
from time import time
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os
import redis

from my_meep.config.configs import get_array
from my_meep.config.config_variables import *
from my_meep.source import source_wrapper
from my_meep import gen_geo_helper
import meep as mp
from copy import deepcopy

r = redis.Redis(port=6379, host='localhost', db=0)

class Sim_manager:
    def get_eps(self):
        if self.eps_data is not None:
            return self.eps_data
        
        eps_sim_manager = Sim_manager(self.Geometry, self.config)
        eps_sim = eps_sim_manager.create_sim() 
        eps_sim.eps_averaging = False
        eps_sim.init_sim()
        self.eps_data = eps_sim.get_array(center=mp.Vector3(), size=self.cell_size, component=mp.Dielectric)
        return self.eps_data

    def get_config(self):
        self.particle_size = self.config.getfloat('Geometry', 'particle_area_or_volume')
        self.res = self.config.getfloat('Simulation', 'resolution')
        

    def create_array_assign_func(self):
        dim = int(self.config.getfloat('Simulation', 'dimension'))
        if dim == 3:
            def assign (ez_data, ass_ed, counter): ez_data[counter, :, :, :] = ass_ed
        elif dim == 2:
            def assign (ez_data, ass_ed, counter): ez_data[counter, :, :] = ass_ed
        self.assign = assign

    def __init__(self, Geometry, config):
        self.Simulation = None
        self.eps_data = None
        self.config = config
        self.Geometry = Geometry
        self.get_config()
        self.cell_size = conf.get_array('Geometry', 'cell_size', self.config)
        self.create_array_assign_func()
        self.save_every = int(self.config.getfloat('Simulation', 'save_every'))
        self.out_every = self.config.getfloat('Simulation', 'out_every')
        self.sim_time = self.config.getfloat('Simulation', 'time')
        self.produce_video = self.config.getboolean('Visualization', 'video')
        self.produce_video_cleanup = False
        self.video_produced = False

        self.video = []
        self.video_count = 0
        self.res = self.config.getfloat('Simulation', 'resolution')
        self.X = np.arange(-self.cell_size[0]/2, self.cell_size[0]/2, 1/self.res)
        self.Y = np.arange(-self.cell_size[1]/2, self.cell_size[1]/2, 1/self.res)
        self.X, self.Y = np.meshgrid(self.X, self.Y)
        if self.produce_video:
            if os.path.isfile('animate.mp4'):  
                os.remove('animate.mp4')
            self.writer = animation.FFMpegWriter(fps=5, metadata=dict(artist='matplotlib'), bitrate=1800)

    def create_sim(self):
        if self.Simulation is not None:
            return self.Simulation
        pml = self.config.getboolean('boundary', 'pml')
        if pml:
            pml_layers = [mp.PML(1)]
        else:
            pml_layers = []

        Source = source_wrapper(self.config)

        change_res = self.config.getboolean('Simulation', 'change_res')
        if change_res:
            self.res = 3.4/self.particle_size+40
            self.config.set('Simulation', 'resolution', str(self.res))

        S = 0.5
        cell_size = conf.get_array('Geometry', 'cell_size', self.config)
        sim_args = {
            'resolution':self.res,
            'cell_size':cell_size,
            'boundary_layers':pml_layers,
            'sources':Source,
            'Courant':S
        }
        self.eps_value = get_array('Geometry', 'eps', self.config)
        self.eps_background = self.eps_value[0]

        sim_type = self.config.get('Simulation', 'sim_types')

        if sim_type in ['checker', 'shape', 'effective medium']:
            self.Simulation = mp.Simulation(
                **sim_args,
                geometry=self.Geometry,
                default_material=mp.Medium(epsilon=1)
            )
        elif sim_type == 'voronoi':
            def vor_eps(coord):
                my_vor = self.Geometry
                inbox = my_vor.inbox(coord)

                if inbox:
                    return my_vor.parts_eps[gen_geo_helper.closest_node([coord[0],coord[1],coord[2]], my_vor.points)]
                else:
                    return mp.Medium(epsilon=self.eps_background)

            self.Simulation = mp.Simulation(
                **sim_args,
                material_function=vor_eps
            )
        else:
            raise Exception('One of the option must be specified')
        return self.Simulation

    def __call__(self):
        """ run the simulation and decide whether to use flux calculation """
        pre = time()


        self.create_sim()
        self.get_eps()
        self.arr_dim = self.eps_data.shape

        self.Simulation.init_sim()
        self.calc_sim_time()
        self.ez_data_inted = np.zeros(self.arr_dim)
        self.ez_data = np.empty(np.array((self.save_every, *self.arr_dim)))

        def func_wrapper(Simulation):
            self.run_every(Simulation)
                    
        self.Simulation.run(mp.at_every(self.out_every, func_wrapper), until=self.sim_time)
        self.ez_data = self.ez_data_inted/((self.end_time-self.start)*self.out_every)
        r.set('animate_file', os.getcwd() + '/' + 'animate.mp4')

        if self.config.getboolean('General', 'verbals'):
            after = time()
            print('Time for meep Simulation: ', after - pre)
            
        return self.ez_data

    def calc_sim_time(self):
        self.start_factor = self.config.getfloat('Simulation', 'start_factor')
        self.start = int(self.cell_size[0]*self.start_factor/self.out_every)
        # start is when the transient phase has past
        # print('start record time is: ', start)
        self.counter = 0
        self.overall_count = 0
        self.end_time = int(self.sim_time/self.out_every)
        
        if self.start > self.end_time:
            print('insufficient start: ')
            self.start = round(self.end_time/2)


    def run_every(self, Simulation):
        # every timestep
        if self.produce_video and self.overall_count < self.start:
            self.assign(self.ez_data, Simulation.get_array(component=mp.Ez), self.counter)
            self.video.append((plt.pcolor(self.X, self.Y, self.ez_data[self.counter, :].transpose()),))
            self.counter += 1

        if self.overall_count == self.start and self.produce_video:
            for ele in range(10): self.video.append((plt.text(-2, 0.5, 'Transient state finished', fontsize=12), ))

        # every timestep after start
        if self.overall_count >= self.start:
            if self.produce_video and not self.produce_video_cleanup:
                self.ez_data *= 0
                self.counter = 0
                self.produce_video_cleanup = True
            self.assign(self.ez_data, Simulation.get_array(component=mp.Ez), self.counter)
            if self.produce_video and not self.video_produced:
                self.video.append((plt.pcolor(self.X, self.Y, self.ez_data[self.counter, :].transpose()),))
            self.counter += 1

        # every integration timestep
        if self.counter == self.save_every:
            self.counter = 0
            self.ez_data = np.power(self.ez_data, 2)
            self.ez_data_inted += np.trapz(self.ez_data, dx=self.out_every, axis=0)
            if self.produce_video and not self.video_produced:
                animate = animation.ArtistAnimation(plt.gcf(), self.video, interval=1/self.config.getfloat('Visualization', 'frame_speed'), repeat_delay=3000, blit=True)
                animate.save('animate.mp4', writer=self.writer)
                self.video = []
                self.video_produced = True

        # end of the simulation
        if self.overall_count == self.end_time-1 and self.counter != 0:
            pass

        self.overall_count += 1