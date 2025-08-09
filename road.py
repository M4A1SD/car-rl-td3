import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

x = np.array([0,300,600,1000,1200,1600,1800,2000,2500,2700,3000,3100,3500,4000,4300,4600,5000,6000])
y = np.array([0,20,20,80,80,100,100,80,80,60,80,80,120,20,30,20,20,60])

spline = make_interp_spline(x,y)

x_new = np.linspace(x.min(),x.max(),300)
y_new = spline(x_new)

plt.plot(x_new,y_new)
plt.show()

fuel_usage = 0

# calculate slope
slope = np.diff(y_new) / np.diff(x_new)
tanh_slopes = np.tanh(slope)  # [-1,1]
external_acc = tanh_slopes * 3 # [-3,3]

acc = throttle +  external_acc # [-11 , 7]
speed = speed + acc

fuel_usage += throttle * 0.01

if speed < 0:
    reward = -1000






# need to test minimum fuel usage for completing the road


# these are the coordinates of the road 
# x,y
# (0,0) 
# (300,20)
# (600,20)
# (1000,80)
# (1200,80)
# (1600,100)
# (1800,100)
# (2000,80)
# (2500,80)
# (2700,60)
# (3000,80)
# (3100,80)
# (3500,120)
# (4000,20)
# (4300,30)
# (4600,20)
# (5000,20)
# (6000,60)




# a max = 4
# a min = -3.5

