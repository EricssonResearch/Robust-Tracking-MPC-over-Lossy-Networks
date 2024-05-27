'''
This script is used to estimate the model error, when linearizing the cartpole system around the origin. 
We draw random initial conditions and from there stabilize the cartpole around the origin.
Then we use the obtained trajectories to estimate the disturbance w(k)
'''
import numpy as np
import control as ct
from Cartpole.cartpole import Cartpole

# set random seed for reproducibility
np.random.seed(1)
rng_w = np.random.default_rng(456)

# sampling time
Th=0.02
physics_timestep = 1/500
# initialize variable for zero order hold
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
        
# cost matrices 
Q = np.diag([100, 10, 100, 10])
R = 0.1*np.eye(nu)
K, _, _ =ct.dlqr(A, B, Q, R)

# MPC horizon
N = 20
Acl = A-B@K

# set initial value of the plant
cartpole = Cartpole(timeStep=physics_timestep)

# set limits for the initial conditions
pos_max = 1
pos_min = -1
vel_max = 0.5
vel_min = -0.5
ang_max = 0.3
ang_min = -0.3
ang_vel_max = 0.5
ang_vel_min = -0.5

w_traj = np.array([[0],[0],[0],[0]])

for l in range(100):
    if l % 10 == 0:
        print(f"k = {l}")
    # draw random initial condition
    x0 = np.r_[rng_w.uniform(pos_min,pos_max),
                rng_w.uniform(vel_min,vel_max),
                rng_w.uniform(ang_min,ang_max),
                rng_w.uniform(ang_vel_min,ang_vel_max)]
    x0.shape = (nx,1)
    x_traj = x0[:]

    cartpole.reset_state(x0)

    x = cartpole.get_observation()
    i = 0
    for t in range(4000):
        if i == 0:
            u_t= - np.dot(K, x)
            if t != 0:
                # estimate disturbance at discrete time step k
                # get x(k-1)
                x_km1 = x_traj[:,-1]
                x_km1.shape = (nx,1)
                # estimate w(k) = x(k)-A_cl@x(k-1)
                w = x - Acl@x_km1
                w.shape = (nx,1)
                # save discrete-time trajectories
                w_traj = np.hstack((w_traj, w))
                x_traj = np.hstack((x_traj,x))
            i+=1
        if i>=lim_zoh:
            i = 0
        else:
            i += 1
        # update plant state
        cartpole.apply_action(u_t)
        x = cartpole.get_observation()
        x.shape = (nx,1)

    # check if the system was stabilized
    if np.linalg.norm(x) > 0.001:
        print(f"System not stabilized in simulation {l}")
    
# Print interval the disturbance lies in when we consider discard 2.5% of values, which correspond to the values with the largest absoulte values.
# We consider these values outliers and, therefore, discard them.
print(f"w_1 in [{np.quantile(w_traj[0,:],0.0125)}, {np.quantile(w_traj[0,:],0.9875)}]")
print(f"w_2 in [{np.quantile(w_traj[1,:],0.0125)}, {np.quantile(w_traj[1,:],0.9875)}]")
print(f"w_3 in [{np.quantile(w_traj[2,:],0.0125)}, {np.quantile(w_traj[2,:],0.9875)}]")
print(f"w_4 in [{np.quantile(w_traj[3,:],0.0125)}, {np.quantile(w_traj[3,:],0.9875)}]")
