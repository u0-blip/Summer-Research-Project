import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

y, x = np.meshgrid(np.linspace(-10, 10,100), np.linspace(-10, 10,100))

z = np.sin(x)*np.sin(x)+np.sin(y)*np.sin(y)

v = np.linspace(-10, 10,100)
t = np.sin(v)*np.sin(v)
tt = np.cos(v)*np.cos(v)
###########

fig = plt.figure(figsize=(16, 8),facecolor='white')
gs = gridspec.GridSpec(5, 2)
ax1 = plt.subplot(gs[0,0])

line, = ax1.plot([],[],'b-.',linewidth=2)
ax1.set_xlim(-10,10)
ax1.set_ylim(0,1)
ax1.set_xlabel('time')
ax1.set_ylabel('amplitude')
ax1.set_title('Oscillationsssss')
time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)

#############################
ax2 = plt.subplot(gs[1:3,0])
quad1 = ax2.pcolormesh(x,y,z,shading='gouraud')
ax2.set_xlabel('time')
ax2.set_ylabel('amplitude')
cb2 = fig.colorbar(quad1,ax=ax2)

#########################
ax3 = plt.subplot(gs[3:,0])
quad2 = ax3.pcolormesh(x, y, z,shading='gouraud')
ax3.set_xlabel('time')
ax3.set_ylabel('amplitude')
cb3 = fig.colorbar(quad2,ax=ax3)

############################
ax4 = plt.subplot(gs[:,1])
line2, = ax4.plot(v,tt,'b',linewidth=2)
ax4.set_xlim(-10,10)
ax4.set_ylim(0,1)

def init():
    line.set_data([],[])
    line2.set_data([],[])
    quad1.set_array([])
    return line,line2,quad1

def animate(iter):
    t = np.sin(2*v-iter/(2*np.pi))*np.sin(2*v-iter/(2*np.pi))
    tt = np.cos(2*v-iter/(2*np.pi))*np.cos(2*v-iter/(2*np.pi))
    z = np.sin(x-iter/(2*np.pi))*np.sin(x-iter/(2*np.pi))+np.sin(y)*np.sin(y)
    line.set_data(v,t)
    quad1.set_array(z.ravel())
    line2.set_data(v,tt)
    return line,line2,quad1

gs.tight_layout(fig)

anim = animation.FuncAnimation(fig,animate,frames=100,interval=50,blit=False,repeat=False)
plt.show()

print( 'Finished!!')
