## generate a titled Gaussian beam profile by defining the amplitude function of the Source

import meep as mp
import math
import cmath
import numpy as np
import matplotlib.pyplot as plt



def gaussian_beam(sigma, k, x0):
    def _gaussian_beam(x):
        return cmath.exp(1j*2*math.pi*k.dot(x-x0)-(x-x0).dot(x-x0)/(2*sigma**2))
    return _gaussian_beam

def get_gaussian_source(sigma, k, src_pt):
    return [mp.Source(src=mp.ContinuousSource(fcen, fwidth=0.2*fcen),
                     component=mp.Ez,
                     center=src_pt,
                     size=mp.Vector3(y=20),
                     amp_func=gaussian_beam(sigma,k,src_pt))]

if __name__ == "__main__":
    resolution = 40 # pixels/μm

    cell_size = mp.Vector3(20,20,0)

    pml_layers = [mp.PML(thickness=1.0)]

    fcen = 5.0 # center frequency of CW Source (wavelength is 1 μm)

    tilt_angle = math.radians(0) # angle of tilted beam
    k = mp.Vector3(x=1).rotate(mp.Vector3(z=1),tilt_angle).scale(fcen)

    sigma = 1.0 # beam width
    src_pt = mp.Vector3(x=4)

    Simulation = mp.Simulation(cell_size=cell_size,
                        sources=get_gaussian_source(sigma, k, src_pt),
                        k_point=k,
                        boundary_layers=pml_layers,
                        resolution=resolution)

    non_pml_vol = mp.Volume(center=mp.Vector3(), size=mp.Vector3(20,20,0))
    Simulation.run(until=20)

    plt.figure()
    Simulation.plot2D()

    ez_data = Simulation.get_array(vol=non_pml_vol, component=mp.Ez)
    print(ez_data.shape)
    plt.figure()
    plt.imshow(np.flipud(np.transpose(np.real(ez_data))), interpolation='spline36', cmap='RdBu')
    plt.axis('off')
    plt.show()
