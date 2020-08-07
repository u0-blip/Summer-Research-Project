import os
import cmath
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import my_meep.gen_geo_helper as gen_geo_helper
import meep as mp
from my_meep.config.configs import *
from my_meep.sim import sim_run_wrapper

def get_flux_region(sim):
    nfreq = 100

    # reflected flux
    refl_fr = mp.FluxRegion(center=gen_geo_helper.np2mp(get_array(
        'source', 'near_flux_loc', config)), size=gen_geo_helper.np2mp(get_array('source', 'flux_size', config)))
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

    side = [sim.add_flux(wcen, flux_width*wcen, nfreq, fr) for fr in frs]

    # transmitted flux

    return sim, side

def sim_run(sim):
    out_every = config.getfloat('sim', 'out_every')
    pt = gen_geo_helper.np2mp(get_array('source', 'near_flux_loc', config))
    sim.run(mp.at_every(out_every, f1),
            until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, pt, 1e-3))
    return 0

def get_fluxes(sim, basic_sim, sides, basic_sides):
    sim_run(basic_sim)

    basic_trans_flux_data = basic_sim.get_flux_data(basic_sides[0])
    basic_trans_flux_mag = mp.get_fluxes(basic_sides[0])

    # get rid of the straight transmitted data to get the reflected data
    sim.load_minus_flux_data(sides[0], basic_trans_flux_data)

    ez_data = sim_run(sim)

    trans_flux_mag = [np.array(mp.get_fluxes(side)) for side in sides]
    trans_flux_mag = np.array(trans_flux_mag)

    flux_freqs = np.array(mp.get_flux_freqs(sides[0]))
    wave_len = 1/flux_freqs

    normalise_tran = trans_flux_mag[1]/basic_trans_flux_mag
    loss = (basic_trans_flux_mag -
            np.sum(trans_flux_mag[1:], axis=0))/basic_trans_flux_mag
    reflected = -trans_flux_mag[0]/basic_trans_flux_mag

    if mp.am_master():
        plt.figure()
        plt.plot(wave_len, reflected, 'bo-', label='reflectance')
        plt.plot(wave_len, normalise_tran, 'ro-', label='transmittance')
        plt.plot(wave_len, loss, 'go-', label='loss')
        ax = plt.gca()
        # plt.axis([-np.inf, np.inf, 0, 1])
        ax.set_ylim([0, 1])
        plt.legend(loc="center right", fontsize=20)
        plt.xlabel("wavelength (μm)")
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.savefig('_'.join([os.getcwd() + '/output/export/3/', config.get(
            'geo', 'shape'), config.get('geo', 'particle_size')]) + '_flux.png', dpi=300, bbox_inches='tight')

    return ez_data
