'''
This script shows how the tracking MPC controller over lossy networks of [1] can be used.
Here, we use the same setup as in Example_of_Tracking_MPC.py but this time we add a lossy network, such that we need to use the Estimator and Smart Actuator class.

[1] M. Pezzutto, M. Farina, R. Carli and L. Schenato, "Remote MPC for Tracking Over Lossy Networks," in IEEE Control Systems Letters, vol. 6, pp. 1040-1045, 2022, doi: 10.1109/LCSYS.2021.3088749.
'''

import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True
from LinearMPCOverNetworks.TrackingMPC import TrackingMPC
import LinearMPCOverNetworks.SmartActuator as SmartActuator
import LinearMPCOverNetworks.Estimator as Estimator
import polytope as pc

# set random seed for reproducibility
np.random.seed(1)
rng_w = np.random.default_rng(679)
rng_gamma = np.random.default_rng(347)
rng_theta = np.random.default_rng(124)

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

# obtain steady state controller
K= MPC_obj.get_steady_state_controller_gain()

# set number of time steps for the simulation
T=120

# set initial state of the plant
x0 = np.array([[1],[2]])
x_traj = x0
x=x0
# initialize input trajectory
u_traj = np.zeros((nu,T))
# initialize estimate trajectory
x_hat_traj = x0

# intitialize estimator
estim=Estimator.Estimator(A,B,K,x0[:],N)

# initialize state estimate
x_hat = estim.get_estimate()

# initialize smart actuator
smart_act = SmartActuator.SmartActuator(K)

# initialize trajectory for Theta_t
Theta_t_traj=np.zeros(T)

# specify time-varying reference
ref = np.zeros((1,T))
ref[0,0:30]=5*np.ones((1,30))
# NOTE: -9 and 9 are values which are outside of the sate constraint set
ref[0,30:60]=-9*np.ones((1,30))
ref[0,60:90]=9*np.ones((1,30))
ref[0,90:120]=4*np.ones((1,30))

# probability of packet loss from controller to plant
p_c2p=0.7
# probability of packet loss from plant to controller
p_p2c=0.7

for t in range(T):
    if t%10 ==0:
        print(f"t = {t}")

    # determine if signals are sent over the network in the current time step
    if (t==0):
        gamma_t=1
        theta_t=1
    else:
        if (rng_theta.uniform()<p_c2p):
            theta_t=0
        else:
            theta_t=1
        if (rng_gamma.uniform()<p_p2c):
            gamma_t=0
        else:
            gamma_t=1

    # obtain qt at time t
    qt= estim.get_qt()
    
    # determine new controller packet to be sent to plant
    controller_packet = MPC_obj.determine_packet(x_hat, np.hstack((ref[0,t],0)), qt)

    # store controller signals in controller packet to list of packets in estimator
    estim.store_sent_control_sequence(controller_packet['U_t'])

    # process controller packet on plant side to determine u_t and the return packet
    u_t, plant_packet = smart_act.process_packet(controller_packet, x, theta_t)
    # save input trajectory
    u_traj[:,t] = u_t
    # save trajectory of Theta_t
    Theta_t_traj[t]=smart_act.get_Theta_t()
    
    # update plant state
    x =  A @ x + B @ u_t

    # save state trajectory
    x_traj = np.hstack((x_traj,x))

    # update estimator state
    estim.update_estimate(plant_packet,gamma_t)
    x_hat = estim.get_estimate()
    
    # save estimate trajectory
    x_hat_traj = np.hstack((x_hat_traj,x_hat))

# according to Proposition 1 of [1] the estimation error should be zero when Theta_t=1
# Below we want to check if that is the case here as well

# determine estimation error and the norm of the error over time
e_traj=x_traj-x_hat_traj
e_traj_norm=np.linalg.norm(e_traj,axis=0)

# create a binary vector which is 1 if the estimation error is zero and otherwise it is 0
e_traj_norm_bin = (e_traj_norm[0:-1]==0)

# get the indices when Theta_t = 1
ind = Theta_t_traj == 1

# check if the error is 0 whenever Theta_t = 1
if sum(ind) - sum(e_traj_norm_bin[ind]) != 0:
    print("The state is not always the same as the estimated state when Theta_t_traj = 1 ")

# plot the trajectory of the input and the two states 
fig, (ax0,ax1,ax2) = plt.subplots(nrows=3)
ax0.plot(range(T),u_traj.T,'--', label='$u$')
# plot boundaries of the constraint set U
ax0.plot([0,T], [-1,-1], 'r--')
ax0.plot([0,T], [1,1], 'r--')
ax0.legend()
ax0.grid()

# plot first state and its estimate
ax1.plot(range(T+1),x_traj[0,:], label='$x_1$')
ax1.plot(range(T+1), x_hat_traj[0,:], '--', label='$\hat{x}_1$')
# plot boundaries of the constraint set X
ax1.plot([0,T], [-8,-8], 'r--')
ax1.plot([0,T], [8,8], 'r--')
ax1.plot(range(T), ref.T, ':', label='$r$')
ax1.legend()
ax1.grid()

# plot second state its estimate
ax2.plot(range(T+1),x_traj[1,:], label='$x_2$')
ax2.plot(range(T+1),x_hat_traj[1,:],'--', label='$\hat{x}_2$')
ax2.legend()
ax2.grid()

plt.show()