'''
In this script, we demonstrate how the function 'calculate_minimal_robust_positively_invariant_set()' in utils_polytope can be used to
approximate the minimum robust positively invariant set according to [1].
More specifically, we want to reproduce Figure 1 of [1].

[1] S. V. Rakovic, E. C. Kerrigan, K. I. Kouramas and D. Q. Mayne, "Invariant approximations of the minimal robust positively Invariant set," in IEEE Transactions on Automatic Control, vol. 50, no. 3, pp. 406-410, March 2005, doi: 10.1109/TAC.2005.843854.
'''

import numpy as np
import matplotlib.pyplot as plt
import polytope as pc
import LinearMPCOverNetworks.utils_polytope as up

# initialize double integrator dynamics as in equation (15) of [1]
A = np.array([[1, 1], [0, 1]])
nx = A.shape[1]
B = np.array([[1], [1]])
# initialize controller
K = np.array([[1.17, 1.03]])
# obtain closed loop system matrix
A_cl = A - B@K

# define disturbance polytope
Hw=np.r_[np.eye(2),-np.eye(2)]
hw=1*np.ones((4,1))
W = pc.Polytope(Hw,hw)

# calculate approximation of minimal rpi according to Algorithm 1 of [1] 
Fs, status = up.calculate_minimal_robust_positively_invariant_set(A = A_cl, W = W, eps_var = 1.9*10**(-5))

# reproduce Figure 1 of [1], where we only plot W and the approximation of the mrpi
fig , ax = plt.subplots()
ax.axis([-1.5, 1.5, -3, 3])
ax.grid()
Fs.plot(ax, color = 'grey', alpha = 0.5)
W.plot(ax, color = 'grey')

plt.show()