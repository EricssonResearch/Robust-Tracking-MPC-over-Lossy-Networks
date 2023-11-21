'''
This script shows how the tube tracking MPC controller of [1] can be used.
Here, we use the same setup as in Example_of_Tracking_MPC.py but this time we add a disturbance w(k) to the plant dynamics

[1] D. Limon, I. Alvarado, T. Alamo, E.F. Camacho, Robust tube-based MPC for tracking of constrained linear systems with additive disturbances, Journal of Process Control, Volume 20, Issue 3, 2010
'''

import numpy as np
import matplotlib.pyplot as plt
import polytope as pc
from LinearMPCOverNetworks.TubeTrackingMPC import TubeTrackingMPC
import LinearMPCOverNetworks.utils_polytope as up

# set random generator for reproducibility
np.random.seed(1)
rng_w = np.random.default_rng(1)

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

# define disturbance polytope
Hw=np.r_[np.eye(nx),-np.eye(nx)]
hw=0.1*np.ones((2*nx,1))
W = pc.Polytope(Hw,hw)

# initialize TubeTrackingMPC object
MPC_obj=TubeTrackingMPC(A,B,Q,R,N)
# add input constraints to the MPC
MPC_obj.set_input_constraints(U)
# add state constraints to the MPC
MPC_obj.set_state_constraints(X)
# setup the optimization problem
MPC_obj.setup_optimization(W)
# get ancillary controller
K = MPC_obj.get_ancillary_controller_gain()

# set number of time steps for the simulation
T=120

# set initial state of the plant
x0 = np.array([[1],[2]])
x_traj = x0
x=x0
# initialize input trajectory
u_traj = np.zeros((nu,T))
# initialize nominal state trajectory
x_traj_nom = np.zeros((nx,T))

# specify time-varying reference
ref = np.zeros((1,T))
ref[0,0:30]=5*np.ones((1,30))
# NOTE: -9 and 9 are values which are outside of the sate constraint set
ref[0,30:60]=-9*np.ones((1,30))
ref[0,60:90]=9*np.ones((1,30))
ref[0,90:120]=4*np.ones((1,30))

for t in range(T):        
    x_init=x_traj[:,t]
    x_init.shape=(nx, )
    # solve MPC problem
    x_nom_traj, u_tube_traj, _, _ = MPC_obj.solve_optimization_problem(x_init,np.hstack((ref[0,t],0)))
    # obtain optimal nominal input by using the column vector in u_tube_traj
    u_tube = u_tube_traj[:,0]
    u_tube.shape=(nu,1)
    # obtain optimal nominal state by using the column vector in x_nom_traj
    x_nom_0 = x_nom_traj[:,0]
    x_nom_0.shape = (nx,1)
    # obtain control input applied to the plant
    u = u_tube - K@(x-x_nom_0)
    
    # draw random disturbance
    w = rng_w.uniform(-0.1,0.1,nx)
    w.shape = (nx,1)
    
    # update plant state
    x=A @ x + B @ u + w
    
    # check if input is inside constraint set U
    if u not in U:
        print(f"Input constraints violated at t = {t} with input u = {u}")

    # save trajectories
    x_traj = np.hstack((x_traj,x))    
    u_traj[:,t] = u
    x_nom_0.shape = (nx,)
    x_traj_nom[:,t] = x_nom_0

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
ax1.plot(range(T),ref.T, label='$r$')
# plot boundaries of the constraint set X
ax1.plot([0,T], [-8,-8], 'r--')
ax1.plot([0,T], [8,8], 'r--')
ax1.legend()
ax1.grid()
# plot second state
ax2.plot(range(T+1), x_traj[1,:], '--', label='$x_2$')
ax2.legend()
ax2.grid()

# get rpi 
Z = MPC_obj._Z

fig1, ax3 = plt.subplots()
# plot tubes centered at the nominal state at each time step 
for i in range(T):
    x_nompZ = up.mink_sum(Z,x_traj_nom[:,i])
    x_nompZ.plot(ax3, color = 'white', alpha = 0.5, edgecolor = 'g', linewidth = 2)

# plot nominal and actual state trajectories
ax3.plot(x_traj[0,:], x_traj[1,:],marker='x')
ax3.plot(x_traj_nom[0,:], x_traj_nom[1,:],'--',marker='+')
ax3.grid()
plt.show()