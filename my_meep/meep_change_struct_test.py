import meep as mp

def change_mat(sim):
    t = sim.meep_time()
    fn = t * 0.2
    geom = [mp.Sphere(radius=0.025, center=mp.Vector3(fn))]
    mp.set_materials_from_geometry(
        sim.structure,
        geom,
        sim.eps_averaging,
        sim.subpixel_tol,
        sim.subpixel_maxeval,
        sim.ensure_periodicity,
        False,
        sim.default_material,
        None,
        sim.extra_materials
    )

sim = mp.Simulation(
        cell_size=mp.Vector3(10, 10),
        resolution=16
    )

sim.run(mp.at_beginning(mp.output_epsilon), mp.at_time(10, change_mat),
mp.at_end(mp.output_epsilon), until=60)

