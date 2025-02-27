'''
Script to reproduce the results for the control of the linearized cartpole system in Section V of [1].

[1]  D. Umsonst and F. S. Barbosa, "Remote Tube-based MPC for Tracking Over Lossy Networks," 2024 IEEE 63rd Conference on Decision and Control (CDC), Milan, Italy, 2024, pp. 1041-1048, doi: 10.1109/CDC56724.2024.10885830.
'''
# %%
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import LinearMPCOverNetworks.TubeTrackingMPC as TubeTrackMPC
import LinearMPCOverNetworks.TrackingMPC as TrackMPC
import LinearMPCOverNetworks.SmartActuator as SmartActuator
import LinearMPCOverNetworks.Estimator as Estimator
import polytope as pc
import control as ct

# fix random seeds for reproducibility
np.random.seed(1)
rng_w = np.random.default_rng(679)
rng_gamma = np.random.default_rng(347)
rng_theta = np.random.default_rng(124)

# sampling time
Th=0.02
# define number of discrete time steps for simulation 
T=int(5/Th) # for a five second simulation

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

# Output matrix
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

### TUBE MPC###
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

### EXTENDED TUBE MPC ###
# initialize extended robust tube-based tracking MPC object
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

# reference 
ref = 0.5*np.ones((1,T))

# %%
# Number of monte carlo simulations
N_MC = 20
# initialize vector of packet loss probabilities we want to investigate
prob_packet_loss = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# initialize dictionaries to save trajectories for each packet loss probability
trajectory_dict_tube = {}
trajectory_dict_Exttube = {}
trajectory_dict_track = {}

# initialize matrices to save tracking errors
tracking_error_tube = np.zeros((len(prob_packet_loss),N_MC))
tracking_error_Exttube = np.zeros((len(prob_packet_loss),N_MC))
tracking_error_track = np.zeros((len(prob_packet_loss),N_MC))

# set initial value of the plant
x0 = np.array([[0],[0],[0],[0]])

is_track_infeasible = np.zeros((len(prob_packet_loss),1))

for i in range(len(prob_packet_loss)):
    # iterate through the different packet loss probabilities
    print(f"Running iteration i = {i} with probability {prob_packet_loss[i]}")
    
    for l_mc in range(N_MC):
        if l_mc % 10 == 0:
            print(f"At Monte Carlo Simulation {l_mc}")

        ### Tube MPC ###
        # initialize trajectory of state for tube MPC
        x_traj_tube = x0[:]

        # initialize plant state
        x_tube=x0[:]
        
        # initialize nominal trajectory for the tube MPC
        x_nom_traj_tube=x0[:]

        # intitialize estimator for tube tracking MPC
        estim_tube = Estimator.Estimator(A, B, K_steady_state_tube, x0[:], N)

        # initialize smart actuator for tube tracking MPC
        smart_act_tube = SmartActuator.ConsistentActuator(A, B, K_steady_state_tube, K_plant_tube, x0[:])

        # initialize state estimate for tube MPC
        x_hat_tube = estim_tube.get_estimate()

        ### Extended Tube MPC ###
        #initialize trajectory of state, input, and estimate for extended tube MPC
        x_traj_Exttube = x0[:]

        # initialize plant state
        x_Exttube=x0[:]
        
        # initialize nominal trajectory for the tube MPC
        x_nom_traj_Exttube=x0[:]

        # intitialize estimator for tube tracking MPC
        estim_Exttube = Estimator.RobustEstimator(A, B, K_steady_state_Exttube, K_plant_Exttube, x0[:], N)
        # initialize smart actuator for tube tracking MPC
        smart_act_Exttube = SmartActuator.ConsistentActuator(A, B, K_steady_state_Exttube, K_plant_Exttube, x0[:], is_extended_MPC_used = True)

        # initialize state estimate for tube MPC
        x_hat_Exttube = estim_Exttube.get_estimate()

        ### Track MPC ###
        # initialize trajectory of state, input, and estimate for tracking MPC
        x_traj_track = x0[:]

        # initialize plant state
        x_track=x0[:]

        # intitialize estimator for tracking MPC
        estim_track = Estimator.Estimator(A, B, K_steady_state_track, x0[:], N)
        # initialize smart actuator
        smart_act_track = SmartActuator.SmartActuator(K_steady_state_track)
        
        # initialize state estimate for tracking MPC
        x_hat_track = estim_track.get_estimate()

        track_feasible = True
        for t in range(T):
            # assume that the first package transmission is successful
            if (t==0):
                gamma_t=1
                theta_t=1
            
            # uniformly draw a disturbance from the disturbance set
            w = np.r_[rng_w.uniform(-w_pos_min,w_pos_max),
                    rng_w.uniform(-w_vel_min,w_vel_max),
                    rng_w.uniform(-w_ang_min,w_ang_max),
                    rng_w.uniform(-w_ang_vel_min,w_ang_vel_max)]
            w.shape = (nx,1)

            ##### Calculate controller packet and store it in estimator #####

            ### TUBE MPC ###
            
            # obtain qt at time t
            qt_tube = estim_tube.get_qt()
            # determine new controller packet to be sent to plant
            controller_packet_tube = TubeMPC.determine_packet(x_hat_tube, np.hstack((ref[0,t],0,0,0)), qt_tube)
            # store controller signals in controller packet to list of packets in estimator
            estim_tube.store_sent_control_sequence(controller_packet_tube['U_t'])

            ### EXTENDED TUBE MPC ###

            # obtain qt at time t
            qt_Exttube = estim_Exttube.get_qt()
            # determine new controller packet to be sent to plant
            controller_packet_Exttube, x_nom_0_Exttube = ExtTubeMPC.determine_packet(x_hat_Exttube, np.hstack((ref[0,t],0,0,0)), qt_Exttube, gamma_t)
            # store controller signals in controller packet to list of packets in estimator
            estim_Exttube.store_sent_control_sequence(controller_packet_Exttube['U_t'])
            estim_Exttube.store_current_optimal_inital_nominal_plant_states(x_nom_0_Exttube)

            ### TRACKING MPC ###
            
            if track_feasible:
                # obtain qt at time t
                qt_track = estim_track.get_qt()
                
                # determine new controller packet to be sent to plant
                controller_packet_track = TrackMPC.determine_packet(x_hat_track, np.hstack((ref[0,t],0,0,0)), qt_track)
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

            # process controller packet on plant side to determine u_t and the return packet
            u_t_tube, plant_packet_tube = smart_act_tube.process_packet(controller_packet_tube, x_tube, theta_t)
            x_nom_traj_tube = np.hstack((x_nom_traj_tube, smart_act_tube.get_x_nom()))

            # update plant state of plant controller by the 
            x_tube =  A @ x_tube + B @ u_t_tube + w

            # add plant state to trajectory vector
            x_traj_tube = np.hstack((x_traj_tube,x_tube))

            # check if the plant state is in a tube around the nominal state 
            if not (x_traj_tube[:,t]-x_nom_traj_tube[:,t] in Z_tube):
                print(f"At k = {t} in for probability loss p_{i}={prob_packet_loss[i]} the real trajectory is not in a tube around the nominal one") 
            
            ### EXTENDED MPC ###

            # process controller packet on plant side to determine u_t and the return packet
            u_t_Exttube, plant_packet_Exttube = smart_act_Exttube.process_packet(controller_packet_Exttube, x_Exttube, theta_t)
            x_nom_traj_Exttube = np.hstack((x_nom_traj_Exttube, smart_act_Exttube.get_x_nom()))

            # update plant state of plant controller by the 
            x_Exttube =  A @ x_Exttube + B @ u_t_Exttube + w

            # add plant state to trajectory vector
            x_traj_Exttube = np.hstack((x_traj_Exttube,x_Exttube))
            
            # check if the plant state is in a tube around the nominal state 
            if not (x_traj_Exttube[:,t]-x_nom_traj_Exttube[:,t] in Z_Exttube):
                print(f"At k = {t} in for probability loss p_{i}={prob_packet_loss[i]} the real trajectory is not in a tube around the nominal one for the extended MPC") 

            ### TRACKING MPC ###

            if track_feasible:
                # process controller packet on plant side to determine u_t and the return packet
                u_t_track, plant_packet_track = smart_act_track.process_packet(controller_packet_track, x_track, theta_t)
                    
                # update plant state of the plant controlled by the tracking MPC
                x_track =  A @ x_track + B @ u_t_track + w

                # add plant state to trajectory vector
                x_traj_track = np.hstack((x_traj_track, x_track))


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
            estim_tube.update_estimate(plant_packet_tube,gamma_t)
            x_hat_tube = estim_tube.get_estimate()

            ### EXTENDED TUBE MPC ###            

            # update estimator state
            estim_Exttube.update_estimate(plant_packet_Exttube,gamma_t)
            x_hat_Exttube = estim_Exttube.get_estimate()

            ### TRACKING MPC ###

            if track_feasible:
                # update estimator state
                estim_track.update_estimate(plant_packet_track,gamma_t)
                x_hat_track = estim_track.get_estimate()         
        
        # save the tracking errors
        tracking_error_tube[i,l_mc] = 1/T*np.sqrt(np.sum((x_traj_tube[0,0:-1]-ref)**2+(x_traj_tube[1,0:-1])**2+(x_traj_tube[2,0:-1])**2+(x_traj_tube[3,0:-1])**2))
        tracking_error_Exttube[i,l_mc] = 1/T*np.sqrt(np.sum((x_traj_Exttube[0,0:-1]-ref)**2+(x_traj_Exttube[1,0:-1])**2+(x_traj_Exttube[2,0:-1])**2+(x_traj_Exttube[3,0:-1])**2))
        
        if track_feasible:
            tracking_error_track[i,l_mc] = 1/T*np.sqrt(np.sum((x_traj_track[0,0:-1]-ref)**2+(x_traj_track[1,0:-1])**2+(x_traj_track[2,0:-1])**2+(x_traj_track[3,0:-1])**2))
        else:
            tracking_error_track[i,l_mc] = np.nan

        if l_mc==np.min([5,N_MC-1]):
            # save the trajectories of the fifth simulation or the last simulation if there are less than five simulation.
            trajectory_dict_tube[prob_packet_loss[i]] = x_traj_tube
            trajectory_dict_Exttube[prob_packet_loss[i]] = x_traj_Exttube
            trajectory_dict_track[prob_packet_loss[i]] = x_traj_track

# %%
# Plot histogram of computational times
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

fig, ax = plt.subplots()
# define bins for the histogram
bins = np.linspace(2.5,20,50)
ax.hist(np.clip(comp_times_scaled, bins[0], bins[-1]), bins=bins, alpha = 0.7, label  = 'RT-MPC')
ax.set_xlabel('Execution time [ms]', fontsize='large')
ax.set_ylabel('Count', fontsize='large')

# Plot execution times of the extended MPC
print("Extended MPC Execution Time")
comp_times = ExtTubeMPC.get_computational_times()

bins = np.linspace(2.5,20,50)
comp_times_scaled = [1000*i for i in comp_times]
print(f"Max: {np.max(comp_times_scaled)}")
print(f"95% Quantile: {np.quantile(comp_times_scaled,0.95)}")
print(f"90% Quantile: {np.quantile(comp_times_scaled,0.9)}")
print(f"75% Quantile: {np.quantile(comp_times_scaled,0.75)}")
print(f"Median: {np.median(comp_times_scaled)}")
print(f"Mean: {np.mean(comp_times_scaled)}")
ax.hist(np.clip(comp_times_scaled, bins[0], bins[-1]), bins=bins, alpha = 0.7, color='C2', label = 'ERT-MPC')
ax.set_xlabel('Execution time [ms]', fontsize='large')
ax.set_ylabel('Count', fontsize='large')
ax.legend()

# Print number of failed executions for each packet loss probability
print("Failed executions for tracking MPC:")
print(is_track_infeasible)
# %% 
# Plot results

# Filter the tracking errors to remove the results of the simulations, which resulted in infeasible problems
tracking_error_track_filtered = []
for i in range(len(is_track_infeasible)):
    if is_track_infeasible[i,0]<N_MC:
        tracking_error_track_filtered.append(tracking_error_track[i,~np.isnan(tracking_error_track[i,:])])
    else:
        tracking_error_track_filtered.append(np.nan)

# Plot results for the tracking error as a box plot

# define legends
legend_in = []
legend_label = ["RT-MPC", "ERT-MPC", "R-MPC"]

fig3, ax3 = plt.subplots()

bp3=ax3.boxplot(tracking_error_tube.T, positions=np.array(range(len(tracking_error_tube)))*2-0.6, widths=0.4, patch_artist=True, sym = 'x', boxprops=dict(facecolor="C0"))
legend_in.append(bp3["boxes"][0])
for median in bp3['medians']:
    median.set_color('black')

bp3=ax3.boxplot(tracking_error_Exttube.T, positions=np.array(range(len(tracking_error_Exttube)))*2, widths=0.4, patch_artist=True, sym = 'x', boxprops=dict(facecolor="C2"))
legend_in.append(bp3["boxes"][0])
for median in bp3['medians']:
    median.set_color('black')

bp4=ax3.boxplot(tracking_error_track_filtered,positions=np.array(range(len(tracking_error_track_filtered)))*2+0.6, widths=0.4, patch_artist=True, sym = 'x', boxprops=dict(facecolor="C1"))
legend_in.append(bp4["boxes"][0])
for median in bp4['medians']:
    median.set_color('black')


ax3.set_xticks(range(0, 2*(len(prob_packet_loss)), 2))
ax3.set_xticklabels(prob_packet_loss)
ax3.set_xlabel("Packet Loss Probability", fontsize='large')
ax3.set_ylabel("Average Tracking Error", fontsize='large')
ax3.legend(legend_in, legend_label, loc='best', fontsize='large')

# %% plot the tracking tracking errors
p = 0.4
x_traj_tube_sample = trajectory_dict_tube[p]
x_traj_Exttube_sample = trajectory_dict_Exttube[p]
x_traj_track_sample = trajectory_dict_track[p]
# get length of Remote MPC trajectory, since it could be shorter due to infeasibilities
traj_length_track = x_traj_track_sample.shape[1]
# plot state trajectories
_, (ax1,ax2) = plt.subplots(nrows=2)
time_tube = [Th*i for i in range(T+1)]
time_track = [Th*i for i in range(traj_length_track)]
ax1.plot(time_tube,x_traj_tube_sample[0,:],'-.', color = 'C0', label=legend_label[0], linewidth=2)
ax1.plot(time_tube,x_traj_Exttube_sample[0,:],'-.', color = 'C2', label=legend_label[1], linewidth=2)
ax1.plot(time_track,x_traj_track_sample[0,:],'--', color = 'C1', label=legend_label[2], linewidth=2)
if traj_length_track<T+1:
    ax1.plot(time_track[-1],x_traj_track_sample[0,traj_length_track-1], '*', color = 'C1')
ax1.plot(time_tube[0:-1],ref.T, color = 'k', label='r(k)')
ax1.set_ylabel("Position $p$ [m]", fontsize='large')
ax1.legend()
ax1.set_xticks([0, 1, 2, 3, 4, 5])
ax1.grid()

ax2.plot(time_tube,x_traj_tube_sample[2,:],'-.', color = 'C0', linewidth=2)
ax2.plot(time_tube,x_traj_Exttube_sample[2,:],'-.', color = 'C2', linewidth=2)
ax2.plot(time_track,x_traj_track_sample[2,:],'--', color = 'C1', linewidth=2)
if traj_length_track<T+1:
    ax2.plot(time_track[-1],x_traj_track_sample[2,traj_length_track-1], '*', color = 'C1')
# Plot reference
ax2.plot([0,time_tube[-1]],[0,0], color = 'k', label='Reference')
# Plot bounds for angle
ax2.plot([0,time_tube[-1]],[0.3,0.3], color = 'r')
ax2.plot([0,time_tube[-1]],[-0.3,-0.3], color = 'r')
ax2.set_xlabel("Time [s]", fontsize='large')
ax2.set_ylabel("Angle $\phi$ [rad]", fontsize='large')
ax2.set_xticks([0, 1, 2, 3, 4, 5])
ax2.grid()

plt.tight_layout()
plt.show()