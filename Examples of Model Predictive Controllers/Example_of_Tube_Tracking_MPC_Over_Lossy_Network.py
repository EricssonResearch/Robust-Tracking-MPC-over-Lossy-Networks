'''
This script shows how the tube tracking MPC controller of [1] can be used.
Here, we use the same setup as in Example_of_Tracking_MPC.py but this time we add a disturbance w(k) to the plant dynamics and include a lossy network.
These additions require us to use the Consistent Actuator and the Estimator proposed in [1].

[1] D. Umsonst and F. S. Barbosa, "Remote Tube-based MPC for Tracking Over Lossy Networks," 2024 IEEE 63rd Conference on Decision and Control (CDC), Milan, Italy, 2024, pp. 1041-1048, doi: 10.1109/CDC56724.2024.10885830.
'''

import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True
import LinearMPCOverNetworks.TubeTrackingMPC as TubeTrackMPC
import LinearMPCOverNetworks.SmartActuator as SmartActuator
import LinearMPCOverNetworks.Estimator as Estimator
import polytope as pc
import LinearMPCOverNetworks.utils_polytope as up

# set random seed for reproducibility
np.random.seed(1)
rng_w = np.random.default_rng(1)
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

# define disturbance polytope
Hw=np.r_[np.eye(nx),-np.eye(nx)]
hw=0.1*np.ones((2*nx,1))
W = pc.Polytope(Hw,hw)

# set number of time steps for the simulation
T=120

# set initial state of the plant
x0 = np.array([[1],[2]])
x_traj = x0[:]
# initialize nominal state trajectory
x_traj_nom=x0[:]
# initialize input trajectory
u_traj = np.zeros((nu,T))

x_hat_traj = x0[:]
# initialize plant state
x=x0[:]

# initialize TubeTrackingMPC object
MPC_obj=TubeTrackMPC.TubeTrackingMPC(A,B,Q,R,N)

# add input constraints to the MPC
MPC_obj.set_input_constraints(U)

# add state constraints to the MPC
MPC_obj.set_state_constraints(X)

# setup the optimization problem
# NOTE: Here, we need to set the fixed_initial_state flag to True, 
#       because the Remote Tube Tracking MPC of [1] requires that (see equation (8d)) 
MPC_obj.setup_optimization(W, fixed_initial_state = True)

# obtain steady state controller
K_ss = MPC_obj.get_steady_state_controller_gain()

# obtain plant controller
K_ancillary = MPC_obj.get_ancillary_controller_gain()

# define rpi set
Z = MPC_obj._Z

# intitialize estimator
estim = Estimator.Estimator(A,B,K_ss,x0[:],N)

# initialize state estimate
x_hat = estim.get_estimate()

# initialize smart actuator
consistent_act = SmartActuator.ConsistentActuator(A, B, K_ss, K_ancillary, x0[:])

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

    w = rng_w.uniform(-0.1,0.1,nx)
    w.shape = (nx,1)

    # obtain qt at time t
    qt = estim.get_qt()
    # determine new controller packet to be sent to plant
    controller_packet = MPC_obj.determine_packet(x_hat, np.hstack((ref[0,t],0)), qt)

    # store controller signals in controller packet to list of packets in estimator
    estim.store_sent_control_sequence(controller_packet['U_t'])
    
    # process controller packet on plant side to determine u_t and the return packet
    u_t, plant_packet = consistent_act.process_packet(controller_packet, x, theta_t)
    # save nominal trajectory and trajectory of Theta_t
    x_traj_nom = np.hstack((x_traj_nom,consistent_act.get_x_nom()))
    Theta_t_traj[t]=consistent_act.get_Theta_t()

    u_traj[:,t] = u_t

    # update plant state
    x =  A @ x + B @ u_t + w

    # save state trajectory
    x_traj = np.hstack((x_traj,x))

    # update estimator state
    estim.update_estimate(plant_packet,gamma_t)
    x_hat = estim.get_estimate()

    # save estimate trajectory
    x_hat_traj = np.hstack((x_hat_traj,x_hat))

# according to Proposition 1 of [1] the estimation error should be zero when Theta_t=1
# Below we want to visualize if that is true in our simulation

# determine if the error is in Z
e_traj=x_traj-x_hat_traj
is_e_in_Z = np.zeros((T, 1))
for i in range(T):
    is_e_in_Z[i,:] = (e_traj[:,i] in Z) 

# get the indices when Theta_t = 1
ind = Theta_t_traj == 1

# check if the error is in Z whenever Theta_t = 1
if sum(ind) - sum(is_e_in_Z[ind]) != 0:
    print("The state is not always in a tube around the nominal trajectory when Theta_t_traj = 1 ")

# Next, we want to check if the state is always in a tube around the nominal trajectory in the smart actuator
for i in range(T):
    if not (x_traj[:,i]-x_traj_nom[:,i] in Z):
        print(f"At i = {i} the real trajectory is not in a tube around the nominal one") 

# plot results
fig, (ax0,ax1,ax2) = plt.subplots(nrows=3)
ax0.plot(range(T),u_traj.T,'--', label='$u$')
ax0.legend()

ax1.plot(range(T+1),x_traj[0,:], label='$x_1$')
ax1.plot(range(T+1), x_hat_traj[0,:], '--', label='$\hat{x}_1$')
ax1.plot(range(T), ref.T, ':', label='$r$')
ax1.legend()

ax2.plot(range(T+1),x_traj[1,:], label='$x_2$')
ax2.plot(range(T+1),x_hat_traj[1,:],'--', label='$\hat{x}_2$')
ax2.legend()

fig1, ax3 = plt.subplots()
# plot tubes centered at the nominal state at each time step 
for i in range(T):
    x_nompZ = up.mink_sum(Z, x_traj_nom[:,i])
    x_nompZ.plot(ax3, color = 'white', alpha = 0.5, edgecolor = 'g', linewidth = 2)


# plot nominal and actual state trajectories
ax3.plot(x_traj[0,:], x_traj[1,:],marker='x')
ax3.plot(x_traj_nom[0,:], x_traj_nom[1,:],'--',marker='+')
ax3.grid()

plt.show()