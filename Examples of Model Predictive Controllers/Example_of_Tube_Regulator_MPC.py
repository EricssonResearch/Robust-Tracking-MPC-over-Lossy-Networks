'''
In this script, we show how the tube regulator MPC proposed in [1] can be used 
and we reproduce a figure similar to Figure 2 of [1].

[1] D.Q. Mayne, M.M. Seron, S.V. Raković, Robust model predictive control of constrained linear systems with bounded disturbances, Automatica, Volume 41, Issue 2, 2005, Pages 219-224, https://doi.org/10.1016/j.automatica.2004.08.019.
'''

import numpy as np
import matplotlib.pyplot as plt
import polytope as pc 
import LinearMPCOverNetworks.utils_polytope as up
from LinearMPCOverNetworks.TubeRegulatorMPC import TubeRegulatorMPC

# set random generator for reproducibility
np.random.seed(1)
rng_w = np.random.default_rng(1)

# In the following, we use the same dynamics, MPC parameters, and constraint sets as in Section 4.1 of [1]
# initialize plant dynamics
A = np.array([[1, 1], [0, 1]])
nx = A.shape[1]
B = np.array([[0.5], [1]])
nu = B.shape[1]

# initialize cost matrices for MPC
Q = np.eye(nx)
R = 0.01 * np.eye(nu)

# MPC horizon
N = 9

# define disturbance polytope
Hw=np.r_[np.eye(nx),-np.eye(nx)]
hw=0.1*np.ones((2*nx,1))
W = pc.Polytope(Hw,hw)

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

MPC_obj=TubeRegulatorMPC(A,B,Q,R,N)
# add input constraints to the MPC
MPC_obj.set_input_constraints(U)
# add state constraints to the MPC
MPC_obj.set_state_constraints(X)
# setup the optimization problem
MPC_obj.setup_optimization(W)
# obtain controller gain used in the MPC
K = MPC_obj.get_controller_gain()

# set number of time steps for the simulation
T=10

# set initial state of the plant
x0 = np.array([[-5],[-2]])
x_traj = x0
x=x0
# initialize input trajectory
u_traj = np.zeros((nu,T))
# initialize nominal state trajectory
x_traj_nom = np.zeros((nx,T))

for t in range(T):
    # get new state for MPC problem
    x_init=x_traj[:,t]
    x_init.shape=(nx, )
    x_mpc, u_mpc = MPC_obj.solve_optimization_problem(x_init)
    # obtain optimal nominal input by using the column vector in u_mpc
    u_nom_t=u_mpc[:,0]
    u_nom_t.shape=(nu,1)
    # obtain optimal nominal state by using the column vector in x_mpc
    x_nom_t=x_mpc[:,0]

    # obtain control input applied to the plant
    u = u_nom_t - K@(x_init-x_nom_t)

    # draw random disturbance
    w = rng_w.uniform(-0.1,0.1,nx)
    w.shape = (nx,1)
    # update dynamics
    x = A @ x + B @ u + w

    # save trajectories for plotting
    x_traj = np.hstack((x_traj,x))
    u_traj[:,t] = u
    x_traj_nom[:,t] = x_nom_t

# determine rpi
Z = MPC_obj._Z
# determine terminal set
Xf = MPC_obj._Xf
Xf.b.shape = (Xf.b.shape[0],)
# determine Minkowski sum of terminal set and the rpi 
XfpZ = up.mink_sum(Xf,Z)

fig, ax = plt.subplots()

# plot terminal sets
XfpZ.plot(ax, color = 'grey', edgecolor = 'k', alpha = 0.5, linewidth = 2)
Xf.plot(ax, color = 'grey', edgecolor = 'k', linewidth = 2)

# plot tubes centered at the nominal state at each time step 
for i in range(T):
    x_nompZ = up.mink_sum(Z,x_traj_nom[:,i])
    x_nompZ.plot(ax, color = 'white', alpha = 0.5, edgecolor = 'g', linewidth = 2)

# plot nominal and actual state trajectories
ax.plot(x_traj[0,:], x_traj[1,:],'-.',marker='x', label = 'State')
ax.plot(x_traj_nom[0,:], x_traj_nom[1,:],marker='+', label = 'Nominal state')
ax.plot([-10,5],[2,2],'r--')
ax.set_xticks([-8, -6, -4, -2, 0, 2, 4])
ax.set_xlim([-8,4])
ax.set_ylim([-3,3])
ax.grid()
ax.legend()

plt.show()