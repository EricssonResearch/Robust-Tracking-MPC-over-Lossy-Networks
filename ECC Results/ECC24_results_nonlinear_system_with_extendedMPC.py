# %%
'''
Script to reproduce the results for the control of the cartpole system in Section V of our ECC submission
Here, we added the extended tube MPC to the results as well.
'''

import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True
import LinearMPCOverNetworks.TubeTrackingMPC as TubeTrackMPC
import LinearMPCOverNetworks.TrackingMPC as TrackMPC
import LinearMPCOverNetworks.SmartActuator as SmartActuator
import LinearMPCOverNetworks.Estimator as Estimator
import polytope as pc
import control as ct
from Cartpole.cartpole import Cartpole

# fix random seeds for reproducibility
np.random.seed(1)
rng_gamma = np.random.default_rng(3467)
rng_theta = np.random.default_rng(124)

# sampling time
Th=0.02
# sampling time for the physics simulation (run at 500 Hz)
physics_timestep = 1/500
# define number of time steps for simulation 
total_time = 5 # in seconds
# trajectory length in discrete time steps
T = int(total_time / physics_timestep)
# counter for zero order hold
lim_zoh = Th/physics_timestep

# system parameters
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

# cost matrices 
Q = np.diag([100, 10, 100, 10])
R = 0.1*np.eye(nu)

# convex polytopes for the disturbance, the state and the input

# disturbance polytope
Hw=np.r_[np.eye(nx),-np.eye(nx)]
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
Hz_tube = Z_tube.A
hz_tube = Z_tube.b

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
Hz_Exttube = Z_Exttube.A
hz_Exttube = Z_Exttube.b

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

# %%

# Number of monte carlo simulations
N_MC = 20
# initialize vector of packet loss probabilities we want to investigate
prob_packet_loss = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# initialize dictionaries to save trajectories for each packet loss probability
trajectory_dict_tube = {}
trajectory_dict_tube_discrete = {}

trajectory_dict_Exttube = {}
trajectory_dict_Exttube_discrete = {}

trajectory_dict_track = {}
trajectory_dict_track_discrete = {}

# initialize matrices to save control trajectories
u_traj_tube = np.zeros((len(prob_packet_loss),T))
u_traj_Exttube = np.zeros((len(prob_packet_loss),T))
u_traj_track = np.zeros((len(prob_packet_loss),T))

# initialize matrices to save tracking errors
tracking_error_tube = np.zeros((len(prob_packet_loss),N_MC))
tracking_error_tube_discrete = np.zeros((len(prob_packet_loss),N_MC))

tracking_error_Exttube = np.zeros((len(prob_packet_loss),N_MC))
tracking_error_Exttube_discrete = np.zeros((len(prob_packet_loss),N_MC))

tracking_error_track = np.zeros((len(prob_packet_loss),N_MC))
tracking_error_track_discrete = np.zeros((len(prob_packet_loss),N_MC))

# initialize matrices to save tracking errors in the position
tracking_error_tube_pos = np.zeros((len(prob_packet_loss),N_MC))
tracking_error_tube_pos_discrete = np.zeros((len(prob_packet_loss),N_MC))

tracking_error_Exttube_pos = np.zeros((len(prob_packet_loss),N_MC))
tracking_error_Exttube_pos_discrete = np.zeros((len(prob_packet_loss),N_MC))

tracking_error_track_pos = np.zeros((len(prob_packet_loss),N_MC))
tracking_error_track_pos_discrete = np.zeros((len(prob_packet_loss),N_MC))

tracking_error_track_pos = {}

# initialize array that counts infeasibilities of the tracking MPC and extended tube MPC approach
is_Exttube_infeasible = np.zeros((len(prob_packet_loss),1))
is_track_infeasible = np.zeros((len(prob_packet_loss),1))

# set initial value of the plant
x0 = np.array([[0],[0],[0],[0]])

# initialize cartpoles
cartpole_tube = Cartpole(timeStep=physics_timestep, initState=x0)
cartpole_Exttube = Cartpole(timeStep=physics_timestep, initState=x0)
cartpole_track = Cartpole(timeStep=physics_timestep, initState=x0)

largest_violation_of_constraint_exttube = np.zeros((len(prob_packet_loss),1))
largest_time_of_violation_exttube = np.zeros((len(prob_packet_loss),1))
largest_violation_of_constraint_tube = np.zeros((len(prob_packet_loss),1))
largest_time_of_violation_tube = np.zeros((len(prob_packet_loss),1))
is_consistent_during_violation = np.zeros((len(prob_packet_loss),1))

Theta_t_traj_Exttube_discrete = [1]

for i in range(len(prob_packet_loss)):
    # iterate through the different packet loss probabilities
    print(f"Running iteration i = {i} with probability {prob_packet_loss[i]}")
    
    for l_mc in range(N_MC):
        if l_mc % 10 == 0:
            print(f"At Monte Carlo Simulation {l_mc}")

        # reset states of the cartpoles when starting a new simulation
        cartpole_tube.reset_state(x0)    
        cartpole_Exttube.reset_state(x0)      
        cartpole_track.reset_state(x0)     
        # initialize a discrete-time version of the reference     
        ref_discrete = ref[0,0]

        ### TUBE MPC ###
        # initialize tube MPC variables

        #initialize trajectory of state for tube MPC
        x_traj_tube = x0[:]

        #initialize trajectory of discrete-time state for tube MPC
        x_traj_tube_discrete = x0[:]    

        #initialize trajectory of nominal state for tube MPC
        x_nom_traj_tube_discrete = x0[:]

        # initialize plant state
        x_tube=x0[:]

        # initialize nominal trajectory for the tube MPC
        x_nom_traj_tube=x0[:]

        # intitialize estimator for tube MPC
        estim_tube = Estimator.Estimator(A, B, K_steady_state_tube, x0[:], N)

        # initialize smart actuator for tube MPC
        smart_act_tube = SmartActuator.ConsistentActuator(A, B, K_steady_state_tube, K_plant_tube, x0[:])

        # initialize state estimate for tube MPC
        x_hat_tube = estim_tube.get_estimate()

        ### EXT TUBE MPC ###
        # initialize tube MPC variables
        x_traj_Exttube = x0[:]
        x_traj_Exttube_discrete = x0[:]
        x_nom_traj_Exttube_discrete = x0[:]

        # initialize plant state
        x_Exttube=x0[:]

        # initialize nominal trajectory for the extended tube MPC
        x_nom_traj_Exttube=x0[:]

        # intitialize estimator for the extended tube MPC
        estim_Exttube = Estimator.RobustEstimator(A, B, K_steady_state_Exttube, K_plant_Exttube, x0[:], N)

        # initialize smart actuator for the extended tube MPC
        smart_act_Exttube = SmartActuator.ConsistentActuator(A, B, K_steady_state_Exttube, K_plant_Exttube, x0[:], is_extended_MPC_used = True)

        # initialize state estimate for the extended tube MPC
        x_hat_Exttube = estim_Exttube.get_estimate()

        ### TRACK MPC ###
        # initialize track MPC variables
        x_traj_track = x0[:]
        x_traj_track_discrete = x0[:]

        # initialize plant state
        x_track=x0[:]

        # intitialize estimator for the tracking MPC
        estim_track = Estimator.Estimator(A, B, K_steady_state_track, x0[:], N)

        # initialize smart actuator for the tracking MPC
        smart_act_track = SmartActuator.SmartActuator(K_steady_state_track)

        # initialize state estimate for the tracking MPC
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

                    ref_discrete =  np.hstack((ref_discrete, ref[0,t]))

                    Theta_t_traj_Exttube_discrete.append(smart_act_Exttube.get_Theta_t())

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
                        is_Exttube_infeasible[i,:] += 1
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
                        is_track_infeasible[i,:] += 1
                        track_feasible = False
                    else:                        
                        # store controller signals in controller packet to list of packets in estimator
                        estim_track.store_sent_control_sequence(controller_packet_track['U_t'])

                ##### process controller packet in smart actuator and apply control input to plant #####

                # determine if controller packet is sent over the network in the current time step if it is not the first time step
                if t>0:
                    # determine if a packet transmission is successful based on drawing a uniform random variable
                    if (rng_theta.uniform()<prob_packet_loss[i]):
                        theta_t=0
                    else:
                        theta_t=1

                ### TUBE MPC ###
                                
                # process controller packet on plant side to determine u_t and the plant packet
                u_t_tube, plant_packet_tube = smart_act_tube.process_packet(controller_packet_tube, x_tube, theta_t)

                # check if the plant state is in a tube around the nominal state 
                if not (x_traj_tube_discrete[:,i_discrete]-x_nom_traj_tube_discrete[:,i_discrete] in Z_tube):
                    print(f"At k = {t} in for probability loss p_{i}={prob_packet_loss[i]} the real trajectory is not in a tube around the nominal one") 
                    e_nom = x_traj_tube_discrete[:,i_discrete]-x_nom_traj_tube_discrete[:,i_discrete]
                    e_nom.shape = (nx,1)
                    He=Hz_tube@e_nom
                    He.shape = (He.shape[0],)
                    violation = He-hz_tube
                        
                    if np.max(violation) > largest_violation_of_constraint_tube[i,0]:
                        largest_violation_of_constraint_tube[i,0] = np.max(violation)

                    if t>largest_time_of_violation_tube[i,0]:
                        largest_time_of_violation_tube[i,0] = t
                
                ### EXTENDED TUBE MPC ###

                if ext_tube_feasible:
                    # process controller packet on plant side to determine u_t and the return packet
                    u_t_Exttube, plant_packet_Exttube = smart_act_Exttube.process_packet(controller_packet_Exttube, x_Exttube, theta_t)
                    
                    # check if the plant state is in a tube around the nominal state 
                    if not (x_traj_Exttube_discrete[:,i_discrete]-x_nom_traj_Exttube_discrete[:,i_discrete] in Z_Exttube):
                        print(f"At k = {t} in for probability loss p_{i}={prob_packet_loss[i]} the real trajectory is not in a tube around the nominal one for the extended MPC") 
                        e_nom = x_traj_Exttube_discrete[:,i_discrete]-x_nom_traj_Exttube_discrete[:,i_discrete]
                        e_nom.shape = (nx,1)
                        He=Hz_Exttube@e_nom
                        He.shape = (He.shape[0],)
                        violation = He-hz_Exttube
                        
                        if Theta_t_traj_Exttube_discrete[i_discrete] == 1:
                            is_consistent_during_violation[i,0]= 1

                        if np.max(violation) > largest_violation_of_constraint_exttube[i,0]:
                            largest_violation_of_constraint_exttube[i,0] = np.max(violation)

                        if t>largest_time_of_violation_exttube[i,0]:
                            largest_time_of_violation_exttube[i,0] = t

                ### TRACKING MPC ### 

                if track_feasible:
                    # process controller packet on plant side to determine u_t and the return packet
                    u_t_track, plant_packet_track = smart_act_track.process_packet(controller_packet_track, x_track, theta_t)


                ##### send plant packet to controller #####

                # determine if plant packet is sent over the network in the current time step if it is not the first time step
                if (t>0):
                    # determine if a packet transmission is successful based on drawing a uniform random variable
                    if (rng_gamma.uniform()<prob_packet_loss[i]):
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

            u_traj_tube[i,t] = u_t_tube

            # update plant state of plant controller by the 
            cartpole_tube.apply_action(u_t_tube)
            
            x_tube = cartpole_tube.get_observation()
            x_tube.shape = (4,1)
            # add plant state to trajectory vector
            x_traj_tube = np.hstack((x_traj_tube,x_tube))
            
            x_nom_traj_tube = np.hstack((x_nom_traj_tube, smart_act_tube.get_x_nom()))

            x_hat_tube = estim_tube.get_estimate()

            #### update cartpole that uses extended tube MPC ####
            if ext_tube_feasible:

                u_traj_Exttube[i,t] = u_t_Exttube

                # update plant state of plant controller by the 
                cartpole_Exttube.apply_action(u_t_Exttube)
                
                x_Exttube = cartpole_Exttube.get_observation()
                x_Exttube.shape = (4,1)
                # add plant state to trajectory vector
                x_traj_Exttube = np.hstack((x_traj_Exttube, x_Exttube))
                
                x_nom_traj_Exttube = np.hstack((x_nom_traj_Exttube, smart_act_Exttube.get_x_nom()))

                x_hat_Exttube = estim_Exttube.get_estimate()

            #### update cartpole that uses track MPC ####

            if track_feasible:

                u_traj_track[i,t] = u_t_track

                # update plant state of plant controller by the 
                cartpole_track.apply_action(u_t_track)
                
                x_track = cartpole_track.get_observation()
                x_track.shape = (4,1)
                # add plant state to trajectory vector
                x_traj_track = np.hstack((x_traj_track,x_track))
                
                x_hat_track = estim_track.get_estimate()
        
        # save the tracking errors
        tracking_error_tube[i,l_mc] = 1/T*np.sqrt(np.sum((x_traj_tube[0,0:-1]-ref)**2+(x_traj_tube[1,0:-1])**2+(x_traj_tube[2,0:-1])**2+(x_traj_tube[3,0:-1])**2))
        T_discrete = x_traj_tube_discrete.shape[1]
        tracking_error_tube_discrete[i,l_mc] = 1/T_discrete*np.sqrt(np.sum((x_traj_tube_discrete[0,:]-ref_discrete)**2+(x_traj_tube_discrete[1,:])**2+(x_traj_tube_discrete[2,:])**2+(x_traj_tube_discrete[3,:])**2))

        # save the tracking errors for the first state only
        tracking_error_tube_pos[i,l_mc] = 1/T*np.sqrt(np.sum((x_traj_tube[0,0:-1]-ref)**2))
        tracking_error_tube_pos_discrete[i,l_mc] = 1/T_discrete*np.sqrt(np.sum((x_traj_tube_discrete[0,:]-ref_discrete)**2))

        if ext_tube_feasible:
            # save the tracking errors
            tracking_error_Exttube[i,l_mc] = 1/T*np.sqrt(np.sum((x_traj_Exttube[0,0:-1]-ref)**2+(x_traj_Exttube[1,0:-1])**2+(x_traj_Exttube[2,0:-1])**2+(x_traj_Exttube[3,0:-1])**2))
            T_discrete = x_traj_Exttube_discrete.shape[1]
            tracking_error_Exttube_discrete[i,l_mc] = 1/T_discrete*np.sqrt(np.sum((x_traj_Exttube_discrete[0,:]-ref_discrete)**2+(x_traj_Exttube_discrete[1,:])**2+(x_traj_Exttube_discrete[2,:])**2+(x_traj_Exttube_discrete[3,:])**2))

            # save the tracking errors for the first state only
            tracking_error_Exttube_pos[i,l_mc] = 1/T*np.sqrt(np.sum((x_traj_Exttube[0,0:-1]-ref)**2))
            tracking_error_Exttube_pos_discrete[i,l_mc] = 1/T_discrete*np.sqrt(np.sum((x_traj_Exttube_discrete[0,:]-ref_discrete)**2))

        else:
            # save the tracking errors
            tracking_error_Exttube[i,l_mc] = np.nan
            tracking_error_Exttube_discrete[i,l_mc] = np.nan

            # save the tracking errors for the first state only
            tracking_error_Exttube_pos[i,l_mc] = np.nan
            tracking_error_Exttube_pos_discrete[i,l_mc] = np.nan


        if track_feasible:
            # save the tracking errors
            tracking_error_track[i,l_mc] = 1/T*np.sqrt(np.sum((x_traj_track[0,0:-1]-ref)**2+(x_traj_track[1,0:-1])**2+(x_traj_track[2,0:-1])**2+(x_traj_track[3,0:-1])**2))
            tracking_error_track_discrete[i,l_mc] = 1/T_discrete*np.sqrt(np.sum((x_traj_track_discrete[0,:]-ref_discrete)**2+(x_traj_track_discrete[1,:])**2+(x_traj_track_discrete[2,:])**2+(x_traj_track_discrete[3,:])**2))

            # save the tracking errors for the first state only
            tracking_error_track_pos[i,l_mc] = 1/T_discrete*np.sqrt(np.sum((x_traj_track[0,0:-1]-ref)**2))
            tracking_error_track_pos_discrete[i,l_mc] = 1/T_discrete*np.sqrt(np.sum((x_traj_track_discrete[0,:]-ref_discrete)**2))
        else:
            # save the tracking errors
            tracking_error_track[i,l_mc] = np.nan
            tracking_error_track_discrete[i,l_mc] = np.nan

            # save the tracking errors for the first state only
            tracking_error_track_pos[i,l_mc] = np.nan
            tracking_error_track_pos_discrete[i,l_mc] = np.nan

        if l_mc==np.min([5,N_MC-1]):
            # save the trajectories
            trajectory_dict_tube[prob_packet_loss[i]] = x_traj_tube
            trajectory_dict_tube_discrete[prob_packet_loss[i]] = x_traj_tube_discrete
            trajectory_dict_Exttube[prob_packet_loss[i]] = x_traj_Exttube
            trajectory_dict_Exttube_discrete[prob_packet_loss[i]] = x_traj_Exttube_discrete
            trajectory_dict_track[prob_packet_loss[i]] = x_traj_track
            trajectory_dict_track_discrete[prob_packet_loss[i]] = x_traj_track_discrete
    

comp_times = TubeMPC.get_computational_times()

# scale computational times to obtain milliseconds
comp_times_scaled = [1000*i for i in comp_times]
print(f"Number of computational times available {len(comp_times_scaled)}")
print(f"Max: {np.max(comp_times_scaled)}")
print(f"95% Quantile: {np.quantile(comp_times_scaled,0.95)}")
print(f"90% Quantile: {np.quantile(comp_times_scaled,0.9)}")
print(f"75% Quantile: {np.quantile(comp_times_scaled,0.75)}")
print(f"Median: {np.median(comp_times_scaled)}")
print(f"Mean: {np.mean(comp_times_scaled)}")

print("Extended MPC Execution Time")
comp_times = ExtTubeMPC.get_computational_times()

comp_times_scaled = [1000*i for i in comp_times]
print(f"Max: {np.max(comp_times_scaled)}")
print(f"95% Quantile: {np.quantile(comp_times_scaled,0.95)}")
print(f"90% Quantile: {np.quantile(comp_times_scaled,0.9)}")
print(f"75% Quantile: {np.quantile(comp_times_scaled,0.75)}")
print(f"Median: {np.median(comp_times_scaled)}")
print(f"Mean: {np.mean(comp_times_scaled)}")

# Print number of failed executions for each packet loss probability
print("Failed executions for extended tube MPC:")
print(is_Exttube_infeasible)

print("Failed executions for tracking MPC:")
print(is_track_infeasible)

# %% Plot results for the execution time

# Filter the tracking errors to remove the results of the simulations, which resulted in infeasible problems
# for extended tube MPC
tracking_error_Exttube_discrete_filtered = []
for i in range(len(is_Exttube_infeasible)):
    if is_Exttube_infeasible[i,0]<N_MC:
        tracking_error_Exttube_discrete_filtered.append(tracking_error_Exttube_discrete[i,~np.isnan(tracking_error_Exttube_discrete[i,:])])
    else:
        tracking_error_Exttube_discrete_filtered.append(np.nan)
# for tracking MPC
tracking_error_track_discrete_filtered = []
for i in range(len(is_track_infeasible)):
    if is_track_infeasible[i,0]<N_MC:
        tracking_error_track_discrete_filtered.append(tracking_error_track_discrete[i,~np.isnan(tracking_error_track_discrete[i,:])])
    else:
        tracking_error_track_discrete_filtered.append(np.nan)

# Plot results for the tracking error as a box plot
legend_in = []
legend_label = ["RT-MPC", "ERT-MPC", "R-MPC"]

fig3, ax3 = plt.subplots()

bp3=ax3.boxplot(tracking_error_tube_discrete.T, positions=np.array(range(len(tracking_error_tube_discrete)))*2-0.6, widths=0.4, patch_artist=True, sym = 'x', boxprops=dict(facecolor="C0"))
legend_in.append(bp3["boxes"][0])
for median in bp3['medians']:
    median.set_color('black')

bp3=ax3.boxplot(tracking_error_Exttube_discrete_filtered, positions=np.array(range(len(tracking_error_Exttube_discrete_filtered)))*2, widths=0.4, patch_artist=True, sym = 'x', boxprops=dict(facecolor="C2"))
legend_in.append(bp3["boxes"][0])
for median in bp3['medians']:
    median.set_color('black')

bp4=ax3.boxplot(tracking_error_track_discrete_filtered, positions=np.array(range(len(tracking_error_track_discrete_filtered)))*2+0.6, widths=0.4, patch_artist=True, sym = 'x', boxprops=dict(facecolor="C1"))
legend_in.append(bp4["boxes"][0])
for median in bp4['medians']:
    median.set_color('black')

ax3.set_xticks(range(0, 2*(len(prob_packet_loss)), 2))
ax3.set_xticklabels(prob_packet_loss)
ax3.set_xlabel("Packet Loss Probability", fontsize='large')
ax3.set_ylabel("Average Tracking Error", fontsize='large')
ax3.legend(legend_in, legend_label, loc='best', fontsize='large')

# %% 
# plot the trajectories of position and angle

p = 0.4
x_traj_tube_sample = trajectory_dict_tube[p]

x_traj_Exttube_sample = trajectory_dict_Exttube[p]
# get length of Extended Remote Tube MPC trajectory, since it could be shorter due to infeasibilities
traj_length_Exttube = x_traj_Exttube_sample.shape[1]

x_traj_track_sample = trajectory_dict_track[p]
# get length of Remote MPC trajectory, since it could be shorter due to infeasibilities
traj_length_track = x_traj_track_sample.shape[1]
# Scale time steps to seconds
time_tube = [physics_timestep*i for i in range(T+1)]
time_Exttube = [physics_timestep*i for i in range(traj_length_Exttube)]
time_track = [physics_timestep*i for i in range(traj_length_track)]

# plot state trajectories
_, (ax1,ax2) = plt.subplots(nrows=2)
ax1.plot(time_tube,x_traj_tube_sample[0,:],'-.', color = 'C0', label=legend_label[0], linewidth=2)

ax1.plot(time_Exttube,x_traj_Exttube_sample[0,:],'--', color = 'C2', label=legend_label[1], linewidth=2)
if traj_length_Exttube<T+1:
    ax1.plot(time_Exttube[-1],x_traj_Exttube_sample[0,traj_length_track-1], '*', color = 'C2')

ax1.plot(time_track,x_traj_track_sample[0,:],':', color = 'C1', label=legend_label[2], linewidth=2)
if traj_length_track<T+1:
    ax1.plot(time_track[-1],x_traj_track_sample[0,traj_length_track-1], '*', color = 'C1')

ax1.plot(time_tube[0:-1],ref.T, color = 'k', label='Reference')
ax1.set_ylabel("Position $p$ [m]", fontsize='large')
ax1.legend(fontsize='large')
ax1.set_xticks([0, 1, 2, 3, 4, 5])
ax1.grid()


ax2.plot(time_tube,x_traj_tube_sample[2,:],'-.', color = 'C0', linewidth=2)

ax2.plot(time_tube,x_traj_Exttube_sample[2,:],'-.', color = 'C2', linewidth=2)
if traj_length_Exttube<T+1:
    ax2.plot(time_Exttube[-1],x_traj_Exttube_sample[2,traj_length_track-1], '*', color = 'C2')

ax2.plot(time_track,x_traj_track_sample[2,:],':', color = 'C1', linewidth=2)
if traj_length_track<T+1:
    ax2.plot(time_track[-1],x_traj_track_sample[2,traj_length_track-1], '*', color = 'C1')

ax2.plot([0,time_tube[-1]],[0,0], color = 'k', label='Reference')
ax2.plot([0,time_tube[-1]],[0.3,0.3], color = 'r')
ax2.plot([0,time_tube[-1]],[-0.3,-0.3], color = 'r')
ax2.set_xlabel("Time [s]", fontsize='large')
ax2.set_ylabel("Angle $\phi$ [rad]", fontsize='large')
ax2.set_xticks([0, 1, 2, 3, 4, 5])
ax2.grid()

plt.tight_layout()
plt.show()