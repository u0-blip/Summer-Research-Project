import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import matplotlib.colors as colors
from configs import config

def get_axis(data, fig):
    ax = plt.subplot(1, 1, 1)
    p_collect = ax.pcolor(data[0])
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

def my_animate(data):
    fig = plt.figure(figsize=(16, 8),facecolor='white')
    gs = gridspec.GridSpec(5, 2)

    ax, p_collect = get_axis(data, fig)

    frames = data.shape[0]
    anim = animation.FuncAnimation(fig,animate,frames=frames,interval=config.getfloat('visualization', 'frame_speed'),blit=False,repeat=False, fargs=[[data, ax, p_collect]])
    plt.show()
    print( 'Finished!!')

def moving_average(a, axis, n=3) :
    a = np.swapaxes(a, axis, 0)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret = np.swapaxes(ret, 0, axis)
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
    elif process_func == 'moving_average':
        moving_average(data, axis, n = interval[1])
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
        p_collect = plt.pcolor(data ,norm=c)
        fig.colorbar(p_collect, ax=ax)
        plt.show()
    else:
        my_animate(data)
    

if __name__ == "__main__":
    data = np.random.randint(0, 10, size=(100, 10, 10))
    # my_animate(data)
    my_rms_plot(data, 0, 'rms', [0, 90])
