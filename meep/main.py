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

def index2coord(index, size_arr, size_geo):
    index = (index/size_arr - 0.5)*size_geo
    return index


def create_sim():
    size_cell = [2, 2, 2]
    size_solid = [0.5, 0.5, 0.5]
    size_crystal_base = [0.1, 0.1, 0.1]
    num_crystal = 200
    pml_layers = [mp.PML(0.3)]

    dist = float(sys.argv[1])

    solid_region1 = mp.Block(size_solid, 
                        center = mp.Vector3(dist/2. + 0.25, 0., 0.),
                        material=mp.Medium(epsilon=7.69, D_conductivity=2*math.pi*0.42*2.787/3.4))

    solid_region2 = mp.Block(size_solid, 
                        center = mp.Vector3(-dist/2 - 0.25,  0., 0.),
                        material=mp.Medium(epsilon=7.69, D_conductivity=2*math.pi*0.42*2.787/3.4))

    geometry = [solid_region1, solid_region2]

    source_pad = 0.25
    source = [mp.Source(mp.ContinuousSource(wavelength=2*(11**0.5), width=20),
                    component= mp.Ez,
                    center=mp.Vector3(0.55, 0, 0),
                    size=mp.Vector3(0, 0.1, 0.1))]

# geo = create_simple_geo(coords, shape="sphere", size_solid=radius)
# geo = create_simple_geo(prism_coords, shape="prism", size_solid=size_solid, prism_height=0.2, prism_axis=mp.Vector3(0, 0, 1))

vor = b_voronoi(120)

# v_seed_points = np.random.rand(120, 3) - 0.5
# vor = Voronoi(v_seed_points)
# my_voronoi_geo = voronoi_geo(num_seeds=100, vor=vor)

# sim = create_sim(mode='eps', geo=my_eps, my_voronoi_geo = my_voronoi_geo, res = 50)
# # sim = create_sim(mode='geo', geo=geo, my_voronoi_geo=my_voronoi_geo, res=75)

if __name__ == "__main__":
    b_voronoi(n_towers = 500, seed=15)

    get_sim_output(str(sys.argv[2]), create_sim(), length_t = 40, out_every=0.6, get_3_field=False)
    out_num_geo('checker_geo.bin', my_checker_geo, range_geo=[1.0,1.0,1.0], range_index=[100,100,100])
