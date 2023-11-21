'''
This script shows how the tracking MPC controller of [1] can be used.

[1] D. Limon, I. Alvarado, T. Alamo, E.F. Camacho, MPC for tracking piecewise constant references for constrained linear systems, Automatica, Volume 44, Issue 9, 2008,
'''

import numpy as np
import matplotlib.pyplot as plt
import polytope as pc
from LinearMPCOverNetworks.TrackingMPC import TrackingMPC

# set random seed for reproducibility
np.random.seed(1)

# define discrete double integrator dynamics for the plant
A = np.array([[1, 1], [0, 1]])
nx = A.shape[1]
B = np.array([[0], [1]])
nu= B.shape[1]

# MPC horizon
N = 10

# cost matrices 
Q = np.eye(nx)
R = np.eye(nu)

# define state constraint polytope
Hx=np.r_[np.eye(nx),-np.eye(nx)]
hx=8*np.ones((2*nx,1))
X = pc.Polytope(Hx,hx)

# define input constraint polytope
Hu=np.array([[1], [-1]])
hu=1*np.ones((2*nu,1))
U = pc.Polytope(Hu,hu)

# initialize TrackingMPC object
MPC_obj=TrackingMPC(A,B,Q,R,N)
# add input constraints to the MPC
MPC_obj.set_input_constraints(U)
# add state constraints to the MPC
MPC_obj.set_state_constraints(X)
# setup the optimization problem
MPC_obj.setup_optimization()

# set number of time steps for the simulation
T=120

# set initial state of the plant
x0 = np.array([[1],[2]])
x_traj = x0
x=x0
# initialize input trajectory
u_traj = np.zeros((nu,T))

# specify time-varying reference
ref = np.zeros((1,T))
ref[0,0:30]=5*np.ones((1,30))
# NOTE: -9 and 9 are values which are outside of the sate constraint set
ref[0,30:60]=-9*np.ones((1,30))
ref[0,60:90]=9*np.ones((1,30))
ref[0,90:120]=4*np.ones((1,30))

for t in range(T):        
    # get new state for MPC problem
    x_init=x_traj[:,t]
    x_init.shape=(nx, )
    # solve MPC problem
    _, u_mpc, _, _ = MPC_obj.solve_optimization_problem(x_init,np.hstack((ref[0,t],0)))

    # obtain control input by using the column vector in u_mpc
    u=u_mpc[:,0]
    u.shape=(nu,1)
    
    # update plant state
    x = A @ x + B @ u

    # save trajectories for plotting
    x_traj = np.hstack((x_traj,x))
    u_traj[:,t] = u

# plot the trajectory of the input and the two states 
fig, (ax0,ax1,ax2) = plt.subplots(nrows=3)
ax0.plot(range(T),u_traj.T,'--', label='$u$')
# plot boundaries of the constraint set U
ax0.plot([0,T], [-1,-1], 'r--')
ax0.plot([0,T], [1,1], 'r--')
ax0.legend()
ax0.grid()
# plot first state
ax1.plot(range(T+1),x_traj[0,:],'--', label='$x_1$')
# plot boundaries of the constraint set X
ax1.plot([0,T], [-8,-8], 'r--')
ax1.plot([0,T], [8,8], 'r--')
ax1.plot(range(T),ref.T, label='$r$')
ax1.legend()
ax1.grid()
# plot second state
ax2.plot(range(T+1),x_traj[1,:],'--', label='$x_2$')
ax2.legend()
ax2.grid()
plt.show()