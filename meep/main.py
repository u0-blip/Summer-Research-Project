from meep_funcs import *
from gen_geo.bounded_voronoi import *
import sys

np.random.seed(15)
dist = float(sys.argv[1])
geo = create_simple_geo(dist)
# vor = b_voronoi(120)
v_seed_points = np.random.rand(120, 3) - 0.5
vor = Voronoi(v_seed_points)

sim = create_sim(mode='eps', arg=my_eps, vor = vor, res = 30)
get_sim_output(str(sys.argv[2]), sim, length_t = 400, out_every=0.6, get_3_field=False)

# out_num_geo('checker_geo.bin', my_checker_geo, range_geo=[1.0,1.0,1.0], range_index=[100,100,100])
