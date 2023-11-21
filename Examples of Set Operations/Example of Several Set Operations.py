"""
This script tests the implementation of the Minkowski sum, the Pontryagin difference, and the scaling of a polytope.
This is inspired by the plots on set operations done in the pytope package.
"""

import LinearMPCOverNetworks.utils_polytope as utils_polytope
import numpy as np
import polytope as pc
import matplotlib.pyplot as plt

## Pontryagin difference 

# Create two polytopes
P_m1 = pc.box2poly([[-3, 3], [-3, 3]])
vertices = np.r_[[[1, 0]], [[0, -1]], [[-1, 0]], [[0, 1]]]
P_m2 = pc.qhull(vertices)

# create extended polytopes from the polytopes
P_diff = utils_polytope.pont_diff(P_m1, P_m2)

## Scaling of polytope
# generate rotation matrix
rot1 = np.array([[np.cos(np.pi/4),np.sin(np.pi/4)],[-np.sin(np.pi/4), np.cos(np.pi/4)]])

P_m2_scaled = utils_polytope.scale(P_m2,rot1)

fig, ax = plt.subplots()
P_m2.plot(ax, alpha = 0.5, color = (1,0,1))
P_m2_scaled.plot(ax, alpha = 0.5, color = (1,1,1))
ax.legend((r'$P$', r'$P$ rotated by 45 degrees'))
plt.grid()
plt.axis([-1.5, 1.5, -1.5, 1.5])
plt.title('Scaling of a polytope')

fig, ax = plt.subplots()
plt.grid()
plt.axis([-3.5, 3.5, -3.5, 3.5])
P_m1.plot(ax, alpha = 1, color = (1,1,1))
P_diff.plot(ax, alpha = 1, color = (1,1,1))
P_m2.plot(ax, alpha = 1, color = (1,1,1))
ax.legend((r'$P$', r'$P \ominus Q$', r'$Q$'))
plt.title('Pontryagin difference of two polytopes')

## Minkowski sum
# Plot two polytopes and their Minkowski sum
P_m3 = pc.box2poly([[-2, 2], [-2, 2]])
P_m4 = pc.box2poly([[-1, 1], [-1, 1]])
P_mink = utils_polytope.mink_sum(P_m3,P_m4)

x = np.ones((1,2))
P_mink_x = utils_polytope.mink_sum(P_m3,x)

fig, ax = plt.subplots()
plt.grid()
plt.axis([-3.5, 3.5, -3.5, 3.5])
P_m3.plot(ax, alpha = 0.5, color = (1,0,1))
P_m4.plot(ax, alpha = 0.5, color = (0,1,1))
P_mink.plot(ax, alpha = 0.5, color = (1,1,1))
# P_mink_x.plot(ax, alpha = 0.5, color = (1,1,1))
ax.legend((r'$P$', r'$Q$', r'$P \oplus Q$'))
plt.title('Minkowski sum of two polytopes')

plt.show()