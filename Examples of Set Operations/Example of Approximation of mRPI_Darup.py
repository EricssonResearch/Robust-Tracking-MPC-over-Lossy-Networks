'''
In this script, we demonstrate how the approximation of the minimum robust positively invariant set according to [1] can be used.
Here, we want to reproduce Figure 3 of [1] and also demonstrate that the obtained values for k_star are the same as in [1].

[1] M. S. Darup and D. Teichrib, "Efficient computation of RPI sets for tube-based robust MPC," 2019 18th European Control Conference (ECC), Naples, Italy, 2019, pp. 325-330, doi: 10.23919/ECC.2019.8796265.

'''

import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True
import polytope as pc
import LinearMPCOverNetworks.utils_polytope as up
import control as ct
import matplotlib.pyplot as plt

# discrete double integrator
A = np.array([[1, 1], [0, 1]])
nx = A.shape[1]
B = np.array([[0.5], [1]])
nu= B.shape[1]

# define disturbance polytope
Hw=np.r_[np.eye(nx),-np.eye(nx)]
w_max = 0.1
w_min = 0.1
hw= np.r_[w_max*np.ones((nx,1)),w_min*np.ones((nx,1))]
W = pc.Polytope(Hw,hw)

# define state constraint polytope
Hx=np.r_[np.eye(nx),-np.eye(nx)]
hx = np.r_[4,
          2,
          8,
          4]
X = pc.Polytope(Hx,hx)

# define input constraint polytope
Hu=np.array([[1], [-1]])
hu=1*np.ones((2*nu,1))
U = pc.Polytope(Hu,hu)

# cost matrices for LQR
Q = np.eye(nx)
R = np.eye(nu)
K, _, _ =ct.dlqr(A, B, Q, R)
Acl = A-B@K

# Now, we calculate the RPI for different values of epsilon to check if k_star has the correct value
print("Setting eps_var to 1e-1, which should return k_star = 5")
rpi_P1, C1, status = up.calculate_RPI(Acl, W, X, U, K, 10**(-1), 50, return_container = True)
print("Setting eps_var to 1e-2, which should return k_star = 6")
rpi_P2, status = up.calculate_RPI(Acl, W, X, U, K, 10**(-2), 50)
print("Setting eps_var to 1e-3, which should return k_star = 10")
rpi_P3, status = up.calculate_RPI(Acl, W, X, U, K, 10**(-3), 50)

# Next, we reproduce Figure 3 of [1], for which eps_var=1e-1 is assumed
# First, we need to determine (1+eps)R_{k_star} according to 
for i in range(5):
    if i == 0:
        R = W.copy()
    else:
        R = up.mink_sum(R,up.scale(W,np.linalg.matrix_power(Acl, i)))
R = up.scale(R,1+0.1)

# Now, we reproduce Figure 3 of [1] 
fig , ax = plt.subplots()
ax.axis([-0.4, 0.4, -0.3, 0.3])
ax.grid()
C1.plot(ax, color = 'grey', linewidth = 0.1)
rpi_P1.plot(ax, color = 'yellow', linewidth = 0.1)
R.plot(ax, color = 'green', linewidth = 0.1)
ax.set_xticks([-0.4, 0, 0.4])
ax.set_yticks([-0.3, 0, 0.3])

plt.show()