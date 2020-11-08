import math
import meep as mp
import numpy as np
import matplotlib.pyplot as plt

def metal_cavity(w):
    resolution = 50
    sxy = 2
    dpml = 1
    sxy = sxy+2*dpml
    cell = mp.Vector3(sxy,sxy)

    pml_layers = [mp.PML(dpml)]
    a = 1
    t = 0.1
    geometry = [mp.Block(mp.Vector3(a+2*t,a+2*t,mp.inf), material=mp.metal),
                mp.Block(mp.Vector3(a,a,mp.inf), material=mp.air)]

    geometry.append(mp.Block(center=mp.Vector3(a/2),
                             size=mp.Vector3(2*t,w,mp.inf),
                             material=mp.air))

    fcen = math.sqrt(0.5)/a
    df = 0.2
    sources = [mp.Source(src=mp.GaussianSource(fcen,fwidth=df),
                         component=mp.Ez,
                         center=mp.Vector3())]

    symmetries = [mp.Mirror(mp.Y)]

    Simulation = mp.Simulation(cell_size=cell,
                        geometry=geometry,
                        boundary_layers=pml_layers,
                        sources=sources,
                        symmetries=symmetries,
                        resolution=resolution)
    Simulation.plot2D()
    plt.show()
    h = mp.Harminv(mp.Ez, mp.Vector3(), fcen, df)
    print(h)
    Simulation.run(mp.after_sources(h), until_after_sources=500)


    m = h.modes[0]
    f = m.freq
    Q = m.Q
    Vmode = 0.25*a*a
    ldos_1 = Q / Vmode / (2 * math.pi * f * math.pi * 0.5)

    Simulation.reset_meep()

    T = 2*Q*(1/f)
    Simulation.run(mp.dft_ldos(f,0,1), until_after_sources=T)
    ldos_2 = Simulation.ldos_data[0]

    return ldos_1, ldos_2

def vis_res(Simulation):
    ez_data = Simulation.get_array(component=mp.Ez)
    plt.figure()
    plt.title('beam freq: ' + config.get('Source','fcen'))
    plt.imshow(np.flipud(np.transpose(np.real(ez_data))), interpolation='spline36', cmap='RdBu')
    plt.axis('off')

a, b = metal_cavity(1)