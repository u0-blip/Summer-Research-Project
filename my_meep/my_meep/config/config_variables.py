import my_meep.config.configs as conf

file_name = conf.primitive_config.get('process_inp', 'posix_data') + conf.primitive_config.get('process_inp', 'project_name') + '.mpout'

# cell_size = conf.get_array('Geometry', 'cell_size', self.config)
# vor_size = conf.get_array('Geometry', 'solid_size', self.config)
# vor_center = conf.get_array('Geometry', 'solid_center', self.config)


# per_sim = self.config.getboolean('General', 'perform_mp_sim')    
# sim_dim = self.config.getint('Simulation', 'dimension')
# sim_type = self.config.get('Simulation', 'sim_types')
# project_name = self.config.get('process_inp', 'project_name')
# verbals = self.config.getboolean('General', 'verbals')
# in_sim_frequency = self.config.getfloat('Source', 'fcen')*0.34753
# pml = self.config.getboolean('boundary', 'pml')
# out_every = self.config.getfloat('Simulation', 'out_every')
# sim_time = self.config.getfloat('Simulation', 'time')
# save_every = int(self.config.getfloat('Simulation', 'save_every'))
# out_every = self.config.getfloat('Simulation', 'out_every')
# dim = int(self.config.getfloat('Simulation', 'dimension'))
# change_res = self.config.getboolean('Simulation', 'change_res')
# particle_size_t = self.config.get('Geometry','particle_size_t')