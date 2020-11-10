
from copy import deepcopy
import meep as mp
from functools import partial


from my_meep.visulization import Plot_res
import my_meep.gen_geo_helper as gen_geo_helper
import gen_voronoi.simple_geo_and_arg as simp
from my_meep.gen_geo import Gen_geo
from my_meep.flux import Flux_manager
from my_meep.Result_manager import Result_manager

from my_meep.config.configs import get_array, Config_manager, primitive_config
from my_meep.config.config_variables import *

from my_meep.Result_manager import write_res
from my_meep.Sim_manager import Sim_manager
from my_meep.get_vor import get_vor
import redis
import json
import numpy as np

r = redis.Redis(port=6379, host='0.0.0.0', db=0)

def sim_main(vor, current_user_id, config):
    # dimensions = mp.CYLINDRICAL
    gen_geo = Gen_geo(vor, config)
    Geometry = gen_geo()

    sim_manager = Sim_manager(Geometry, config)

    eps_data = sim_manager.get_eps()

    actual_fill_factor = Result_manager.get_area(eps_data, config)
        

    if config.getboolean('Simulation', 'calc_flux'):
        basic_sim_manager = Sim_manager(Geometry=[], config=config)
        basic_sim = basic_sim_manager.create_sim()

        Simulation = sim_manager.create_sim()

        flux_manager = Flux_manager(config, Simulation)
        basic_flux_manager = Flux_manager(config, basic_sim)

        basic_sim, basic_sides = basic_flux_manager.get_flux_region()
        Simulation, sides = flux_manager.get_flux_region()

        ez_data = flux_manager.get_fluxes(basic_sim, sides, basic_sides)
    else:
        ez_data = sim_manager()

    print('The RMS matrix shape: ' + str(ez_data.shape))
    # r.set('user_' + str(current_user_id) + '_plot_rms_whole', json.dumps(ez_data.tolist()))
    r.set('user_' + str(current_user_id) + '_plot_rms_block_max', json.dumps(np.max(ez_data)))
    r.set('user_' + str(current_user_id) + '_plot_rms_block_max_log', json.dumps(np.log(np.max(ez_data))))
    result_manager = Result_manager(ez_data, eps_data, config, current_user_id)
    mean, std = result_manager.result_statistics()

    plot_res = Plot_res(result_manager, sim_manager.Simulation, current_user_id)

    plot_res()

    return mean, std, actual_fill_factor
    
def wsl_main(web_config=None, current_user_id=0):

    if web_config:
        config = deepcopy(web_config)
        config.add_section('web')
        config.set('web', 'web', '1')
    else:
        config = primitive_config
        config.add_section('web')
        config.set('web', 'web', '0')

    verbals = config.getboolean('General', 'verbals')
    if not verbals:
        mp.quiet(True)

    cell_size = get_array('Geometry', 'cell_size', config)
    sim_dim = config.getint('Simulation', 'dimension')
    if sim_dim == 2:
        cell_size[2] = 0
        config.set('Geometry', 'cell_size', ','.join([str(ele) for ele in cell_size]))

    # convert from GHz to meep unit
    config.set('Source', 'fcen', str(config.getfloat('Source', 'fcen')*0.34753))

    vor = get_vor(config)
    config_manager = Config_manager(config)
    configs = config_manager.break_down_config()
    
    func = partial(sim_main, vor, current_user_id)

    # with mproc.Pool(processes=config.getint('General', 'sim_cores')) as pool:
    #     mean_std_area_value = pool.map(func, configs)
    #     counter += 1
    #     print(counter, ' out of ', total, ' is done.')

    mean_std_area_value = []
    
    if web_config: 
        for i, c in enumerate(configs):
            res = func(c)
            mean_std_area_value.append(res)
            yield i, config_manager.total, res
    else:
        for i, c in enumerate(configs):
            res = func(c)
            mean_std_area_value.append(res)
            yield None


    data, var_descrip_str, param_iterated = config_manager.sort_res(mean_std_area_value)
    write_res(config, data, var_descrip_str, current_user_id, param_iterated)
