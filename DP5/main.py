import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from help import *

k = 500
time = 10
dt = 0.001
walls = create_walls(10)

p1 = particle(0, 0, 5)
p1.v = V(0, 0)
p1.r = 1
p1.q = 10 * 10**(-4)

p2 = particle(2.5, 0, 1)
p2.v = V(0, -30)
p2.r = 0.5
p2.q = -7.5 * 10**(-4)

p3 = particle(-2.5, 0, 1)
p3.v = V(0, 30)
p3.r = 0.5
p3.q = -7.5 * 10**(-4)

parts = [
    p1, 
    p2, 
    p3
    ]



fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot()
plt.grid(True)
plt.xlim(walls.get('xmin'), walls.get('xmax'))
plt.ylim(walls.get('ymin'), walls.get('ymax'))
plt.xticks(np.linspace(int(walls.get('xmin')), int(walls.get('xmax')), int(walls.get('xmax')-walls.get('xmin') + 1)))
plt.yticks(np.linspace(int(walls.get('ymin')), int(walls.get('ymax')), int(walls.get('ymax')-walls.get('ymin') + 1)))
points1, = plt.plot([], [], ls="", marker = 'o', color = 'red', markersize = 10)
points2, = plt.plot([], [], ls="", marker = 'o', color = 'green', markersize = 5)
points3, = plt.plot([], [], ls="", marker = 'o', color = 'blue', markersize = 5)



def update(dt):
    tdt(dt, parts, walls, k)
    points1.set_data(parts[0].x, parts[0].y)
    points2.set_data(parts[1].x, parts[1].y)
    points3.set_data(parts[2].x, parts[2].y)



temps = int(time // dt)
Time = tuple([time / temps] * temps)
anim = FuncAnimation(fig, update, frames = Time, repeat = False, interval = 1)
plt.show()