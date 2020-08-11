from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import matplotlib.colors as colors

from my_meep.config.configs import get_array
from my_meep.config.config_variables import *

import numpy as np
import matplotlib.pyplot as plt



class IndexTracker(object):
    def __init__(self, ax, config):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        res = config.getfloat('sim', 'resolution')
        X = np.arange(-cell_size[0]/2, cell_size[0]/2, 1/res)
        Y = np.arange(-cell_size[1]/2, cell_size[1]/2, 1/res)
        X, Y = np.meshgrid(X, Y)

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

def plot_3d(eps_data, offset, offset_index, config):

    res = config.getfloat('sim', 'resolution')
    X = np.arange(-cell_size[0]/2, cell_size[0]/2, 1/res)
    Y = np.arange(-cell_size[1]/2, cell_size[1]/2, 1/res)
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure(constrained_layout=True, figsize=(17, 5))
    gs = fig.add_gridspec(1, 3)

    f1 = fig.add_subplot(gs[0, 0])
    slice_eps = eps_data[offset_index[0], :, :]
    f1.pcolormesh(slice_eps)
    plt.xlabel('y')
    plt.ylabel('z')

    f2 = fig.add_subplot(gs[0, 1])
    slice_eps = eps_data[:, offset_index[1],  :]
    f2.pcolormesh(slice_eps)
    plt.xlabel('x')
    plt.ylabel('z')

    f3 = fig.add_subplot(gs[0, 2])
    slice_eps = eps_data[:, :, offset_index[2]]
    p_collect = f3.pcolormesh(slice_eps)
    plt.xlabel('x')
    plt.ylabel('y')
    ax = fig.gca()
    fig.colorbar(p_collect, ax=ax)

    fig = plt.figure(constrained_layout=True)

    ax = fig.gca(projection='3d')

    cset = [[] for i in range(3)]

    Z = eps_data[50, :, :]

    if np.min(Z) == np.max(Z):
        max_level = np.min(Z) + 1
    else:
        max_level = np.max(Z)

    cset[0] = ax.contourf(X, Y, eps_data[:,:,offset_index[2]], zdir='z', offset=offset[0],
                        levels=np.linspace(np.min(Z),max_level,30),cmap='jet', alpha=0.5)

    # now, for the x-constant face, assign the contour to the x-plot-variable:
    cset[1] = ax.contourf(eps_data[:,offset_index[1],:], Y, X, zdir='x', offset=offset[1],
                        levels=np.linspace(np.min(Z),max_level,30),cmap='jet', alpha=0.5)

    # likewise, for the y-constant face, assign the contour to the y-plot-variable:
    cset[2] = ax.contourf(Y, eps_data[offset_index[0],:,:], X, zdir='y', offset=offset[2],
                        levels=np.linspace(np.min(Z),max_level,30),cmap='jet', alpha=0.5)

    fig.colorbar(cset[2], ax=ax)


def get_axis(data, fig):
    ax = plt.subplot(1, 1, 1)
    p_collect = ax.pcolor(data[0], vmin=np.min(data)*1.01, vmax=np.max(data)*0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Field strength')
    fig.colorbar(p_collect, ax=ax)
    
    return ax, p_collect

def animate(i, args):
    data, ax, p_collect = args
    data = data[i]
    p_collect.set_array(data.ravel())
    ax.set_title('time: '+str(i))
    return p_collect

def get_line_data(data):
    x0, y0 = 5, 4.5 # These are in _pixel_ coordinates!!
    x1, y1 = 60, 75
    num = 1000
    x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)

    # Extract the values along the line, using cubic interpolation
    zi = scipy.ndimage.map_coordinates(z, np.vstack((x,y)))

    #-- Plot...
    fig, axes = plt.subplots(nrows=2)
    axes[0].imshow(z)
    axes[0].plot([x0, x1], [y0, y1], 'ro-')
    axes[0].axis('image')

    axes[1].plot(zi)

    plt.show()

def my_animate(data, window=1):
    if window == 1:
        pass
    else:
        data = moving_average(np.abs(data), window)

    fig = plt.figure(figsize=(16, 8),facecolor='white')
    gs = gridspec.GridSpec(5, 2)


    ax, p_collect = get_axis(data, fig)


    frames = data.shape[0]
    anim = animation.FuncAnimation(fig,animate,frames=frames,interval=config.getfloat('visualization', 'frame_speed'),blit=False,repeat=False, fargs=[[data, ax, p_collect]])
    plt.show()
    print( 'Finished!!')

def moving_average(a, n):
    ret = np.cumsum(a, axis=0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def my_rms_plot(data, axis, process_func, interval):
    """ interval is between 0-1, if interval is none, it is stationary """
    if interval[1] <= 1 and isinstance(interval[1], float):
        interval = [int(i*data.shape[axis]) for i in interval]
    data = np.take(data, range(*interval), axis=axis)
    
    if process_func == 'rms' or process_func == 'ms':
        data = np.mean(np.power(data, 2), axis=axis)
        if process_func == 'rms':
            data = np.sqrt(data)
    elif process_func == 'exp_smoothing':
        pass
    else:
        data = process_func(data, axis)
    if process_func == 'ms':
        data = np.mean(np.power(data, 2), axis=axis)

    if len(data.shape) == 2:
        fig = plt.figure()
        ax = plt.axes()
        if config.getboolean('visualization', 'log_res'):
            c = colors.LogNorm(vmin=data.min(), vmax=data.max())
        else:
            c = None
        p_collect = plt.pcolor(X, Y, data ,norm=c)
        fig.colorbar(p_collect, ax=ax)
        plt.show()
    else:
        my_animate(data)
    

if __name__ == "__main__":
    data = np.random.randint(0, 10, size=(100, 10, 10))
    # my_animate(data)
    my_rms_plot(data, 0, 'rms', [0, 90])
    
    fig, ax = plt.subplots(1, 1)

    X = np.random.rand(20, 20, 40)

    tracker = IndexTracker(ax, X)


    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()
