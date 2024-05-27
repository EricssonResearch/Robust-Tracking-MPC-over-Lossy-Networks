'''
Script to generate animations of the inverted pendulum controlled with different remote MPCs.
'''

import numpy as np
import LinearMPCOverNetworks.TubeTrackingMPC as TubeTrackMPC
import LinearMPCOverNetworks.TrackingMPC as TrackMPC
import LinearMPCOverNetworks.SmartActuator as SmartActuator
import LinearMPCOverNetworks.Estimator as Estimator
import polytope as pc
import control as ct
from Cartpole.cartpole import Cartpole

# fix random seed for reproducibility
np.random.seed(1)
rng_gamma = np.random.default_rng(3467)
rng_theta = np.random.default_rng(124)

# sampling time for the controller
Th=0.02
# sampling time for the physics simulation (run at 500 Hz)
physics_timestep = 1/500
# define number of time steps for simulation 
total_time = 5 # in seconds
# trajectory length in discrete time steps
T = int(total_time / physics_timestep)
# counter for zero order hold
lim_zoh = Th/physics_timestep

# system parameters of inverted pendulum
M = 1.0
m = 0.1
b = 0
I = 0.001
g = 9.8
l = 0.5
p = I*(M+m)+M*m*l**2

# state matrix
Ac = np.array([[ 0,    1,    0,    0],
                [ 0, -(I+m*l**2)*b/p,  -(m**2*g*l**2)/p,   0],
                [ 0,    0,    0,    1],
                [ 0, -(m*l*b)/p,  m*g*l*(M+m)/p,  0]])

# input matrix
Bc = np.array(
    [[0],
    [(I+m*l**2)/p],
    [0],
    [-m*l/p]])
    
# get dimensions of state and input
nx = Ac.shape[1]
nu= Bc.shape[1]

# output matrix
Cc = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

# obtain discretized system matrices
sys = ct.ss(Ac,Bc,Cc,0)
d_sys = ct.c2d(sys, Th)
A, B = d_sys.A, d_sys.B

# MPC horizon
N = 20

# cost matrices for LQR controller and MPC
Q = np.diag([100, 10, 100, 10])
R = 0.1*np.eye(nu)

# convex polytopes for the disturbance, the state and the input

# disturbance polytope
Hw=np.r_[np.eye(nx),-np.eye(nx)]
# parameters for disturbance set from estimate_W_for_Cartpole.py
w_pos_max = 0.0001
w_pos_min = 0.0001
w_vel_max = 0.0027
w_vel_min = 0.0027
w_ang_max = 0.0003
w_ang_min = 0.0003
w_ang_vel_max = 0.043
w_ang_vel_min = 0.043
hw= np.r_[w_pos_max,
          w_vel_max,
          w_ang_max,
          w_ang_vel_max,
          w_pos_min,
          w_vel_min,
          w_ang_min,
          w_ang_vel_min]
W = pc.Polytope(Hw,hw)

# define state constraint polytope
Hx=np.r_[np.eye(nx),-np.eye(nx)]
hx=np.ones((2*nx,1))
hx[0,0] = 5
hx[nx,0] = 5
hx[1,0] = 5
hx[nx+1,0] = 5
hx[2,0] = 0.3
hx[nx+2,0] = 0.3
hx[3,0] = 2
hx[nx+3,0] = 2
X = pc.Polytope(Hx,hx)

# define input constraint polytope
Hu=np.array([[1], [-1]])
hu=10*np.ones((2*nu,1))
U = pc.Polytope(Hu,hu)

### TUBE MPC ###
# initialize robust tube-based tracking MPC object
TubeMPC=TubeTrackMPC.TubeTrackingMPC(A,B,Q,R,N)
# set the input constraints
TubeMPC.set_input_constraints(U)
# set the state constraints
TubeMPC.set_state_constraints(X)
# setup the MPC optimization problem
TubeMPC.setup_optimization(W, fixed_initial_state = True, rpi_method = 1)

# get minimum robust positively invariant set, which represents the tube around the nominal state
Z_tube = TubeMPC._Z

# obtain steady state controller
K_steady_state_tube = TubeMPC.get_steady_state_controller_gain()
# obtain plant controller
K_plant_tube = TubeMPC.get_ancillary_controller_gain()

### Extended TUBE MPC ###
# initialize robust tube-based tracking MPC object
ExtTubeMPC=TubeTrackMPC.ExtendedTubeTrackingMPC(A,B,Q,R,N)
# set the input constraints
ExtTubeMPC.set_input_constraints(U)
# set the state constraints
ExtTubeMPC.set_state_constraints(X)
# setup the MPC optimization problem
ExtTubeMPC.setup_optimization(W, fixed_initial_state = True, rpi_method = 1)

# get minimum robust positively invariant set, which represents the tube around the nominal state
Z_Exttube = ExtTubeMPC._Z

# obtain steady state controller
K_steady_state_Exttube = ExtTubeMPC.get_steady_state_controller_gain()
# obtain plant controller
K_plant_Exttube = ExtTubeMPC.get_ancillary_controller_gain()

### TRACK MPC ###
# initialize non-robust tracking MPC object
TrackMPC=TrackMPC.TrackingMPC(A,B,Q,R,N)
# set the input constraints
TrackMPC.set_input_constraints(U)
# set the state constraints
TrackMPC.set_state_constraints(X)
# setup the MPC optimization problem
TrackMPC.setup_optimization()
# obtain steady state controller
K_steady_state_track = TrackMPC.get_steady_state_controller_gain()

# define constant reference 
ref = 0.5*np.ones((1,T))

# initialize vector of packet loss probabilities we want to investigate
prob_packet_loss = 0.4

# set initial value of the plant
x0 = np.array([[0],[0],[0],[0]])

# initialize cartpoles
cartpole_tube = Cartpole(timeStep=physics_timestep, initState=x0)
cartpole_Exttube = Cartpole(timeStep=physics_timestep, initState=x0)
cartpole_track = Cartpole(timeStep=physics_timestep, initState=x0)   

### TUBE MPC ###
# initialize MPC variables

# intialize state trajectory
x_traj_tube = x0[:]
# initialize discrete-time state trajectory
x_traj_tube_discrete = x0[:]
# initialize discrete-time nominal state trajectory
x_nom_traj_tube_discrete = x0[:]

# initialize plant state
x_tube=x0[:]

# intitialize estimator
estim_tube = Estimator.Estimator(A, B, K_steady_state_tube, x0[:], N)

# initialize smart actuator
smart_act_tube = SmartActuator.ConsistentActuator(A, B, K_steady_state_tube, K_plant_tube, x0[:])

# initialize state estimate 
x_hat_tube = estim_tube.get_estimate()

### EXT TUBE MPC ###
# initialize MPC variables
x_traj_Exttube = x0[:]
x_traj_Exttube_discrete = x0[:]
x_nom_traj_Exttube_discrete = x0[:]
x_hat_traj_Exttube = x0[:]

# initialize plant state
x_Exttube=x0[:]

# intitialize estimator
estim_Exttube = Estimator.RobustEstimator(A, B, K_steady_state_Exttube, K_plant_Exttube, x0[:], N)

# initialize smart actuator
smart_act_Exttube = SmartActuator.ConsistentActuator(A, B, K_steady_state_Exttube, K_plant_Exttube, x0[:], is_extended_MPC_used = True)

# initialize state estimate
x_hat_Exttube = estim_Exttube.get_estimate()

### TRACK MPC ###
# initialize MPC variables
x_traj_track = x0[:]
x_traj_track_discrete = x0[:]
x_hat_traj_track = x0[:]

# initialize plant state
x_track=x0[:]

# intitialize estimator
estim_track = Estimator.Estimator(A, B, K_steady_state_track, x0[:], N)

# initialize smart actuator
smart_act_track = SmartActuator.SmartActuator(K_steady_state_track)

# initialize state estimate
x_hat_track = estim_track.get_estimate()

# initialize variables that check if the remote tracking MPC and the extended tube MPC are feasible
track_feasible = True
ext_tube_feasible = True

# initialize zero order holder counter variable
j_zoh = 0
# initialize discrete time step variable
i_discrete = 0

for t in range(T):
    if j_zoh == 0:
        # determine if signals are sent over the network in the current time step
        if (t==0):
            # assume that the first transmission of packets is successful
            gamma_t=1
            theta_t=1
        if t != 0:
            i_discrete +=1
            x_traj_tube_discrete =  np.hstack((x_traj_tube_discrete, x_tube))
            x_nom_traj_tube_discrete =  np.hstack((x_nom_traj_tube_discrete, smart_act_tube.get_x_nom()))

            x_traj_Exttube_discrete =  np.hstack((x_traj_Exttube_discrete, x_Exttube))
            x_nom_traj_Exttube_discrete =  np.hstack((x_nom_traj_Exttube_discrete, smart_act_Exttube.get_x_nom()))

            x_traj_track_discrete =  np.hstack((x_traj_track_discrete, x_track))

        ##### Calculate conroller packet and store it in estimator #####

        ### TUBE MPC ###

        # obtain qt at time t
        qt_tube = estim_tube.get_qt()
        # determine new controller packet to be sent to plant
        controller_packet_tube = TubeMPC.determine_packet(x_hat_tube, np.hstack((ref[0,t],0,0,0)), qt_tube)
        # store controller signals in controller packet to list of packets in estimator
        estim_tube.store_sent_control_sequence(controller_packet_tube['U_t'])

        ### EXTENDED TUBE MPC ### 

        if ext_tube_feasible:
            # obtain qt at time t
            qt_Exttube = estim_Exttube.get_qt()
            # determine new controller packet to be sent to plant
            controller_packet_Exttube, x_nom_0_Exttube = ExtTubeMPC.determine_packet(x_hat_Exttube, np.hstack((ref[0,t],0,0,0)), qt_Exttube, gamma_t)

            if controller_packet_Exttube['U_t'] is None:
                ext_tube_feasible = False

            else:
                # store controller signals in controller packet to list of packets in estimator
                estim_Exttube.store_sent_control_sequence(controller_packet_Exttube['U_t'])
                estim_Exttube.store_current_optimal_inital_nominal_plant_states(x_nom_0_Exttube)

        ### TRACKING MPC ###

        if track_feasible:
            # obtain qt at time t
            qt_track = estim_track.get_qt()            
            # determine new controller packet to be sent to plant

            controller_packet_track = TrackMPC.determine_packet(x_hat_track, np.hstack((ref[0,t],0,0,0)), qt_track)

        # check if the problem was infeasible
        if controller_packet_track['U_t'] is None:
            track_feasible = False
        else:                        
            # store controller signals in controller packet to list of packets in estimator
            estim_track.store_sent_control_sequence(controller_packet_track['U_t'])

        ##### process controller packet in smart actuator and apply control input to plant #####

        # determine if controller packet is sent over the network in the current time step if it is not the first time step
        if t>0:
            # determine if a packet transmission is successful based on drawing a uniform random variable
            if (rng_theta.uniform()<prob_packet_loss):
                theta_t=0
            else:
                theta_t=1

        ### TUBE MPC ###
                                    
        # process controller packet on plant side to determine u_t and the plant packet
        u_t_tube, plant_packet_tube = smart_act_tube.process_packet(controller_packet_tube, x_tube, theta_t)

        # check if the plant state is in a tube around the nominal state 
        if not (x_traj_tube_discrete[:,i_discrete]-x_nom_traj_tube_discrete[:,i_discrete] in Z_tube):
            print(f"At k = {t} in for probability loss p={prob_packet_loss} the real trajectory is not in a tube around the nominal one") 

        ### EXTENDED TUBE MPC ###

        if ext_tube_feasible:
            # process controller packet on plant side to determine u_t and the return packet
            u_t_Exttube, plant_packet_Exttube = smart_act_Exttube.process_packet(controller_packet_Exttube, x_Exttube, theta_t)
                        
            # check if the plant state is in a tube around the nominal state 
            if not (x_traj_Exttube_discrete[:,i_discrete]-x_nom_traj_Exttube_discrete[:,i_discrete] in Z_Exttube):
                print(f"At k = {t} in for probability loss p={prob_packet_loss} the real trajectory is not in a tube around the nominal one for the extended MPC") 

        ### TRACKING MPC ### 

        if track_feasible:
            # process controller packet on plant side to determine u_t and the return packet
            u_t_track, plant_packet_track = smart_act_track.process_packet(controller_packet_track, x_track, theta_t)


        ##### send plant packet to controller #####

        # determine if plant packet is sent over the network in the current time step if it is not the first time step
        if (t>0):
            # determine if a packet transmission is successful based on drawing a uniform random variable
            if (rng_gamma.uniform()<prob_packet_loss):
                gamma_t=0
            else:
                gamma_t=1

        ### TUBE MPC ### 

        # update estimator state
        estim_tube.update_estimate(plant_packet_tube, gamma_t)      

        ### EXTENDED MPC ###
        if ext_tube_feasible:
            # update estimator state
            estim_Exttube.update_estimate(plant_packet_Exttube,gamma_t)

        ### TRACKING MPC ###
        if track_feasible:
                            
            # update estimator state
            estim_track.update_estimate(plant_packet_track,gamma_t)

        # increment zero order hold index
        j_zoh += 1
            
    # check if zero order hold index needs to be reset
    if j_zoh >=lim_zoh:
        j_zoh = 0
    else:
        j_zoh += 1

    #### update cartpole that uses tube MPC ####

    # update plant state of plant controller by the 
    cartpole_tube.apply_action(u_t_tube)
            
    x_tube = cartpole_tube.get_observation()
    x_tube.shape = (4,1)
    # add plant state to trajectory vector
    x_traj_tube = np.hstack((x_traj_tube,x_tube))

    x_hat_tube = estim_tube.get_estimate()

    #### update cartpole that uses extended tube MPC ####
    if ext_tube_feasible:

        # update plant state of plant controller by the 
        cartpole_Exttube.apply_action(u_t_Exttube)
                    
        x_Exttube = cartpole_Exttube.get_observation()
        x_Exttube.shape = (4,1)
        # add plant state to trajectory vector
        x_traj_Exttube = np.hstack((x_traj_Exttube, x_Exttube))

        x_hat_Exttube = estim_Exttube.get_estimate()

    #### update cartpole that uses track MPC ####

    if track_feasible:

        # update plant state of plant controller by the 
        cartpole_track.apply_action(u_t_track)
                
        x_track = cartpole_track.get_observation()
        x_track.shape = (4,1)
        # add plant state to trajectory vector
        x_traj_track = np.hstack((x_traj_track,x_track))
                
        x_hat_track = estim_track.get_estimate()
        
    
# %% 
# create animations for the cartpoles for each controller
print("Making GIF for remote tube MPC trajectory")
cartpole_tube.make_gif(x_traj_tube,"RemoteTubeMPC")
print("Making GIF for extended remote tube MPC trajectory")
cartpole_tube.make_gif(x_traj_Exttube,"ExtendedRemoteTubeMPC")
print("Making GIF for remote tracking MPC trajectory")
cartpole_track.make_gif(x_traj_track,"RemoteMPC")