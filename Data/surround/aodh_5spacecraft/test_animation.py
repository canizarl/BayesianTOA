import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers


x_data = []
y_data = []



fig, ax = plt.subplots()
ax.set_xlim(0,105)
ax.set_ylim(0,12)
line, = ax.plot(0,0,'*')
line2, = ax.plot(0,0,'b*')

point = ax.plot(4,3,'r*')

def animation_frame(i):
    x_data = i*10
    y_data = i

    line.set_xdata(x_data)
    line.set_ydata(y_data)
    line2.set_ydata(x_data)
    line2.set_xdata(y_data)



    return line, line2,



animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(0,10,0.01), interval=10)
# Writer = writers["ffmpeg"]
# writer = Writer(fps=15, metadata={'artist':'Me'}, bitrate=1800)
#
# animation.save('animation.mp4', writer)


plt.show(block=False)


