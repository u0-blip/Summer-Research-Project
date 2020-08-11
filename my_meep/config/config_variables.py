import my_meep.config.configs as conf

file_name = conf.config.get('process_inp', 'posix_data') + conf.config.get('process_inp', 'project_name') + '.mpout'

cell_size = conf.get_array('geo', 'cell_size', conf.config)
vor_size = conf.get_array('vor', 'size', conf.config)
vor_center = conf.get_array('vor', 'center', conf.config)


per_sim = conf.config.getboolean('general', 'perform_mp_sim')    
sim_dim = conf.config.getint('sim', 'dimension')
shape_type = conf.config.get('sim', 'type')
project_name = conf.config.get('process_inp', 'project_name')
verbals = conf.config.getboolean('general', 'verbals')
wcen = conf.config.getfloat('source', 'fcen')*0.34753
pml = conf.config.getboolean('boundary', 'pml')
out_every = conf.config.getfloat('sim', 'out_every')
sim_time = conf.config.getfloat('sim', 'time')
save_every = int(conf.config.getfloat('sim', 'save_every'))
out_every = conf.config.getfloat('sim', 'out_every')
dim = int(conf.config.getfloat('sim', 'dimension'))
change_res = conf.config.getboolean('sim', 'change_res')
particle_size_t = conf.config.get('geo','particle_size_t')