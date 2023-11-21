"""
In this script, we demonstrate how the function for calculating the output admissible set can be used to determine the terminal set of an MPC.
To do so, we reproduce the sets Xf and Xf \oplus Z in Figure 2 of [1]

[1] D.Q. Mayne, M.M. Seron, S.V. Raković, Robust model predictive control of constrained linear systems with bounded disturbances, Automatica, Volume 41, Issue 2, 2005, Pages 219-224, https://doi.org/10.1016/j.automatica.2004.08.019.
"""

import numpy as np
import matplotlib.pyplot as plt
import control as ct
import polytope as pc
import LinearMPCOverNetworks.utils_polytope as up

# In the following, we use the same dynamics, MPC parameters, and constraint sets as in Section 4.1 of [1]
# initialize plant dynamics
A = np.array([[1, 1], [0, 1]])
nx = A.shape[1]
B = np.array([[0.5], [1]])
nu = B.shape[1]

# initialize cost matrices for MPC
Q = np.eye(nx)
R = 0.01 * np.eye(nu)

# obtain LQR gain and the closed loop system matrix
K, P, _=ct.dlqr(A, B, Q, R)
A_cl=A - B@K

# define disturbance polytope
Hw=np.r_[np.eye(nx),-np.eye(nx)]
hw=0.1*np.ones((2*nx,1))
W = pc.Polytope(Hw,hw)

# calculate RPI
Z, _ = up.calculate_minimal_robust_positively_invariant_set(A_cl,W)

# define input constraint polytope
Hu = np.array([[1],
              [-1]])
hu = np.ones((2,1))
U = pc.Polytope(Hu,hu)

# define state constraint polytope
# NOTE: In the paper the constraints for the state are x_2<=2
#       Here, we use a compact convex polytope, which sets a large bound on x_1
Hx=np.r_[np.eye(2),-np.eye(2)]
hx=np.r_[10, 2, 10, 10]
X = pc.Polytope(Hx,hx)

# Tighten the set X and U given the rpi Z
Xc = up.pont_diff(X,Z)
Hxc, hxc = Xc.A, Xc.b
Uc = up.pont_diff(U,up.scale(Z,-K))
Huc, huc = Uc.A, Uc.b

# define set, where the nominal states should lie in to fulfill the tightened state and input sets
Hxu = np.r_[Hxc,-Huc@K]
hxu = np.r_[hxc,huc]
XU=pc.Polytope(Hxu,hxu)

# calculate maximum admissible output set
Xf = up.calculate_maximum_admissible_output_set(A_cl,XU)
# add mRPI to maximum admissible output set
XfpZ = up.mink_sum(Xf,Z)

# Plot the sets to reproduce Figure 2 without the trajectories
fig , ax = plt.subplots()
ax.axis([-8, 4, -3, 3])
ax.plot([-8, 4],[2, 2],'k')
XfpZ.plot(ax, color = 'grey', edgecolor = 'k', alpha = 0.5, linewidth = 2)
Xf.plot(ax, color = 'grey', edgecolor = 'k', linewidth = 2)
ax.grid()

plt.show()