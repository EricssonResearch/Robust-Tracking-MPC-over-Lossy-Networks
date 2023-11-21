'''
This script shows an example on how the RegulatorMPC class can be used.
'''

import numpy as np
import matplotlib.pyplot as plt
import polytope as pc
from LinearMPCOverNetworks.RegulatorMPC import RegulatorMPC

# Specify plant dynamics
A = np.array([[1, 1], [0, 1]])
nx = A.shape[1]
B = np.array([[0], [1]])
nu= B.shape[1]

# MPC horizon
N = 10

# cost matrices for MPC
Q = np.eye(nx)
R = np.eye(nu)

# input constraint polytope
Hu = np.array([[1],
              [-1]])
hu = np.ones((2,1))
U = pc.Polytope(Hu,hu)

# setup MPC object
MPC_obj=RegulatorMPC(A,B,Q,R,N)
# add input constraints to MPC object
MPC_obj.set_input_constraints(U)
# generate the optimization problem
MPC_obj.generate_optimization_problem()

# set number of time steps for the simulation
T=20

# set initial state of the plant
x0 = np.array([[1],[3]])
x_traj = x0
x=x0
# initialize input trajectory
u_traj = np.zeros((nu,T))

for t in range(T):
    # get new state for MPC problem 
    x_init=x_traj[:,t]
    x_init.shape=(nx, )
    # solve MPC problem
    _,u_mpc = MPC_obj.solve_optimization_problem(x_init)

    # obtain control input by using the first column in u_mpc
    u=u_mpc[:,0]
    u.shape=(nu,1)

    # update dynamics
    x = A @ x + B @ u

    # save trajectories for plotting
    x_traj = np.hstack((x_traj,x))
    u_traj[:,t] = u

# plot the trajectory of the input and the two states 
fig, (ax0,ax1,ax2) = plt.subplots(nrows=3)
ax0.plot(range(u_traj.shape[1]),u_traj.T)
# plot boundaries of the constraint set U
ax0.plot([0,T], [-1,-1], 'r--')
ax0.plot([0,T], [1,1], 'r--')
ax0.set_xlabel('Time step k')
ax0.set_ylabel('input u')
ax0.grid()
# plot first state
ax1.plot(range(x_traj.shape[1]),x_traj[0,:])
ax1.set_xlabel('Time step k')
ax1.set_ylabel('$x_1$')
ax1.grid()
# plot second state
ax2.plot(range(x_traj.shape[1]),x_traj[1,:])
ax2.set_xlabel('Time step k')
ax2.set_ylabel('$x_2$')
ax2.grid()
plt.show()