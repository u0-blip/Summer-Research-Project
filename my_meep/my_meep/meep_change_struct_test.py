import meep as mp

def change_mat(Simulation):
    t = Simulation.meep_time()
    fn = t * 0.2
    geom = [mp.Sphere(radius=0.025, center=mp.Vector3(fn))]
    mp.set_materials_from_geometry(
        Simulation.structure,
        geom,
        Simulation.eps_averaging,
        Simulation.subpixel_tol,
        Simulation.subpixel_maxeval,
        Simulation.ensure_periodicity,
        False,
        Simulation.default_material,
        None,
        Simulation.extra_materials
    )

Simulation = mp.Simulation(
        cell_size=mp.Vector3(10, 10),
        resolution=16
    )

Simulation.run(mp.at_beginning(mp.output_epsilon), mp.at_time(10, change_mat),
mp.at_end(mp.output_epsilon), until=60)

