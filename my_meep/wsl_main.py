
from copy import deepcopy
import meep as mp
from functools import partial


from my_meep.config.config_manager import Config_manager
from my_meep.visulization import Plot_res
import my_meep.gen_geo_helper as gen_geo_helper
import gen_voronoi.simple_geo_and_arg as simp
from my_meep.gen_geo import Gen_geo
from my_meep.sim import create_sim, start_sim
from my_meep.flux import get_flux_region, get_fluxes
from process_res import get_area, result_statistics
from my_meep.config.configs import *
from my_meep.process_res import write_res
from get_vor import get_vor

def get_eps(geo):
    eps_sim = create_sim(geo=geo)
    eps_sim.eps_averaging = False
    eps_sim.init_sim()
    eps_data = eps_sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Dielectric)
    return eps_data

def sim_main(vor, config1):
    # dimensions = mp.CYLINDRICAL
    global config
    config = config1

    gen_geo = Gen_geo(vor)
    geo = gen_geo()

    eps_data = get_eps(geo)
    actual_fill_factor = get_area(eps_data)
        
    if config.getboolean('sim', 'calc_flux'):
        basic_sim = create_sim(geo=[])
        sim = create_sim(geo=geo)
        basic_sim, basic_sides = get_flux_region(basic_sim)
        sim, sides = get_flux_region(sim)
        ez_data = get_fluxes(sim, basic_sim, sides, basic_sides)
    else:
        ez_data = start_sim(geo, eps_data)

    print('The RMS matrix shape: ' + str(ez_data.shape))

    mean, std = result_statistics(ez_data, eps_data)

    plot_res = Plot_res(ez_data, eps_data, create_sim(geo=geo), eps_data)

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
    for i, c in enumerate(configs):
        res = func(c)
        mean_std_area_value.append(res)
        # if web_config: yield i, config_manager.total, res

    data, var_descrip_str = config_manager.sort_res(mean_std_area_value)
    write_res(web_config, data, var_descrip_str)
