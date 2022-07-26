# Reference: https://towardsdatascience.com/animations-with-matplotlib-d96375c5442c

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

plt.style.use('seaborn-pastel')

fig = plt.figure()
ax = fig.add_subplot(xlim = (0,4), ylim=(-2, 2))
line, = ax.plot([], [], lw=3)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    x = np.linspace(0,4,1000)
    y = np.sin(2*np.pi* (x - 0.01 * i))
    line.set_data(x, y)
    print(i)
    return line,

anim = FuncAnimation(fig, animate, init_func=init, frames = 200, interval = 20, blit=True)
# 

plt.show()
#anim.save('sine_wave.gif', writer= 'ffmpeg')
