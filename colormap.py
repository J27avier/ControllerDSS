import numpy as np
#import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
#from colorspacious import cspace_converter

colormap_list = ['jet', 'inferno', 'gnuplot', 'plasma']

x = np.linspace(0, 1.0, 101) # 101 items from 0 to 1
y = np.sin(np.pi*2*x)
i = 0

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)

color_val = np.linspace(0, 1.0, 11) # 11 items from 0 to 1

def plot(event):
    global i
    ax1.cla()
    color_val = np.linspace(0, 1.0, 21) # 11 items from 0 to 1
    c = color_val[i%len(color_val)]
    rgb = cm.get_cmap(colormap_list[3])(c)
    ax1.scatter(x,y, s = 50, color = rgb)
    ax1.scatter(x,y, s = 1, color = 'k')
    fig1.canvas.draw()
    print(i)
    i+=1

cid = fig1.canvas.mpl_connect('key_press_event', plot)
plt.show()
#ax1.scatter(x,y, color = (1,0,0,1))
#ax1.scatter(x,y, color = (0,0,1,1))

# rgb = cm.get_cmap(cmap)(x)[np.newaxis, :, :3]
# lab = cspace_converter ("sRGB1", "CAM02-UCS")(rgb)
# print(rgb)
# print(lab)

print(cm.get_cmap('jet')(0.00))
print(cm.get_cmap('jet')(0.25))
print(cm.get_cmap('jet')(0.50))
print(cm.get_cmap('jet')(0.75))
print(cm.get_cmap('jet')(1.00))

