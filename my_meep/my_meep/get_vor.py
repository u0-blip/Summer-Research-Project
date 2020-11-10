import cmath
import numpy as np
import pandas as pd
from time import time
import pickle
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gen_voronoi.bounded_voronoi import Gen_vor
import redis

def get_vor(config):
    """ get the voronoi geometry file """
    last = time()
    r = redis.Redis(port=6379, host='meep_celery', db=0)
    if config.get('Simulation', 'sim_types') == 'voronoi':
        if config.getboolean('General', 'gen_vor') or r.get('vor_vor') is None:
            gen_vor = Gen_vor(config)
            _, complete_vor, _ = gen_vor.b_voronoi(to_out_geo=True)
        else:
            _, complete_vor = pickle.loads(r.get('vor_vor'))

        vor = complete_vor
        verbals = config.getboolean('General', 'verbals')
        if verbals:
            now = time()
            print('Created voronoi Geometry, time: ' + str(now - last))
    else:
        vor = None
    return vor
