# Files to reproduce the results in the submitted ECC24 paper

## Overview 
In this folder, we find the files that reproduce the figures of the paper _Remote Tube-based MPC for Tracking Over Lossy Networks_ (currently under review)
The folder consist of four scripts and a module `Cartpole`, which implements the cartpole simulator in PyBullet. 
We compare the [Remote MPC of Pezzutto et al.](https://ieeexplore.ieee.org/document/9452064) with our Remote Tube MPC and the extended Remote Tube MPC in our paper and in the plots and animations the scripts in this folder creates.

Below you can see the architecture of our proposed remote tube MPC algorithm (Figure 2 in our paper):

<img src="../figures/Architecture.png" alt="Architecture of our approach" width="450"> 

## Decription of the files
The script [ECC24_results_linear_system.py](./ECC24_results_linear_system.py) simulates the control of the linearized cartpole model over a lossy network, where we analyze different constant packet loss probabilities for the network. This script reproduces Figure 3a and will plot a histogram similar to Figure 3d of our paper. 
Note that the histogram might be slightly different, since the computational times depend on the machine used to execute the script.

The script [ECC24_results_nonlinear_system.py](./ECC24_results_nonlinear_system.py) simulates the control of the nonlinear cartpole model over a lossy network, where we analyze different constant packet loss probabilities for the network.
This script reproduces Figure 3b and Figure 3c of our paper.

In Section IV.F of our paper, we describe an extended Tube MPC approach, which uses state feedback from the plant. The scripts [ECC24_results_linear_system_with_extendedMPC.py](./ECC24_results_linear_system_with_extendedMPC.py) and [ECC24_results_nonlinear_system_with_extendedMPC.py](./ECC24_results_nonlinear_system_with_extendedMPC.py) include the results for the extended MPC as well, while producing similar figures as the ones presented in Figure 3.
Note that while for the linear dynamics the plant state is guaranteed to be in a tube around the nominal state, this is not necessarily guaranteed for the nonlinear plant dynamics. This is especially the case if the disturbance set does not capture all potential modelling errors.

Below you can see the reproduction of Figures 3b and 3c, where we have also added the Extended Remote Tube MPC algorithm.

<img src="../figures/TrackingErrorNonlinearECC24.png" alt="Average tracking error of different MPC algorithms (Figure 3b in our paper)" width="300"> <img src="../figures/TrajectoriesNonlinearECC24.png" alt="One example trajectory for a packet loss of 40%" width="300">

The script [estimate_W_for_Cartpole.py](./ECC24_Estimate_W_for_Cartpole.py) estimates the disturbance set $\mathbb{W}$ as described in Section V.A of our paper. 

Finally, the script [ECC24_create_animations.py](./ECC24_create_animations.py) creates animations of the inverted pendulum when it is controlled over a lossy network with the [Remote MPC of Pezzutto et al.](https://ieeexplore.ieee.org/document/9452064) and with our two proposed Remote Tube MPC approaches.
Below we show the animation of trajectories of the Remote MPC (left), our Remote Tube MPC (middle), and the extended Remote Tube MPC (right) when there are 40% of packet drops in the network.

<img src="../figures/RemoteMPC.gif" alt="One example trajectory for the Remote MPC by Pezzutto et al. with a packet loss of 40%" width="300"> <img src="../figures/RemoteTubeMPC.gif" alt="One example trajectory for our Remote Tube MPC with a packet loss of 40%" width="300"> <img src="../figures/ExtendedRemoteTubeMPC.gif" alt="One example trajectory for our extended Remote Tube MPC with a packet loss of 40%" width="300">

> :exclamation: To solve the optimization problem in the different MPC algorithms, we use the `CVXPY` package with the CLARABEL solver. 
However, we have encountered that the CLARABEL solver fails sometimes. 
Therefore, when one runs the scripts it could happen that the execution of the solver fails. 
We have not yet determined the reason for the failure, but when one re-runs the script after failure it often works.