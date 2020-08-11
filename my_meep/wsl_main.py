
from copy import deepcopy
import meep as mp
from functools import partial


from my_meep.visulization import Plot_res
import my_meep.gen_geo_helper as gen_geo_helper
import gen_voronoi.simple_geo_and_arg as simp
from my_meep.gen_geo import Gen_geo
from my_meep.flux import Flux_manager
from my_meep.Result_manager import Result_manager

from my_meep.config.configs import get_array, Config_manager, config
from my_meep.config.config_variables import *

from my_meep.Result_manager import write_res
from my_meep.Sim_manager import Sim_manager
from my_meep.get_vor import get_vor


def sim_main(vor, config):
    # dimensions = mp.CYLINDRICAL
    gen_geo = Gen_geo(vor, config)
    geo = gen_geo()

    sim_manager = Sim_manager(geo, config)

    eps_data = sim_manager.get_eps()

    actual_fill_factor = Result_manager.get_area(eps_data, config)
        

    if config.getboolean('sim', 'calc_flux'):
        basic_sim_manager = Sim_manager(geo=[], config=config)
        basic_sim = basic_sim_manager.create_sim()

        sim = sim_manager.create_sim()

        flux_manager = Flux_manager(config, sim)
        basic_flux_manager = Flux_manager(config, basic_sim)

        basic_sim, basic_sides = basic_flux_manager.get_flux_region()
        sim, sides = flux_manager.get_flux_region()

        ez_data = flux_manager.get_fluxes(basic_sim, sides, basic_sides)
    else:
        ez_data = sim_manager()

    print('The RMS matrix shape: ' + str(ez_data.shape))

    result_manager = Result_manager(ez_data, eps_data, config)
    mean, std = result_manager.result_statistics()

    plot_res = Plot_res(result_manager, sim_manager.sim, eps_data)

    plot_res()

    return mean, std, actual_fill_factor
    
def wsl_main(web_config=None):
    mp.quiet(True)
    global config

    if web_config:
        config = deepcopy(web_config)
        config.add_section('web')
        config.set('web', 'web', '1')
    else:
        config.add_section('web')
        config.set('web', 'web', '0')

    vor = get_vor(config)
    config_manager = Config_manager(config)
    configs = config_manager.break_down_config()
    
    func = partial(sim_main, vor)

    # with mproc.Pool(processes=config.getint('general', 'sim_cores')) as pool:
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

    data, var_descrip_str = config_manager.sort_res(mean_std_area_value)
    write_res(config, data, var_descrip_str)
