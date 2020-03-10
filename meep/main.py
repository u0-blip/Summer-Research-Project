import sys
from meep_funcs import *
from gen_geo.bounded_voronoi import *
from gen_geo.simple_geo import *

np.random.seed(15)

dist = float(sys.argv[1])
coords = get_coord(dist)
size_solid = [0.2, 0.2, 0.2]
radius = 0.1
prism_coords = get_prism_coord(dist, radius)

# geo = create_simple_geo(coords, shape="sphere", size_solid=radius)
# geo = create_simple_geo(prism_coords, shape="prism", size_solid=size_solid, prism_height=0.2, prism_axis=mp.Vector3(0, 0, 1))

vor = b_voronoi(120)

# v_seed_points = np.random.rand(120, 3) - 0.5
# vor = Voronoi(v_seed_points)
# my_voronoi_geo = voronoi_geo(num_seeds=100, vor=vor)

# sim = create_sim(mode='eps', geo=my_eps, my_voronoi_geo = my_voronoi_geo, res = 50)
# # sim = create_sim(mode='geo', geo=geo, my_voronoi_geo=my_voronoi_geo, res=75)

# # vis(sim)
# get_sim_output(str(sys.argv[2]), sim, length_t = 20, out_every=0.6, get_3_field=False)

# out_num_geo('checker_geo.bin', my_checker_geo, range_geo=[1.0,1.0,1.0], range_index=[100,100,100])
