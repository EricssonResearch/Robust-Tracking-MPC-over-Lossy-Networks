# Remote Tube-based MPC for Tracking Over Lossy Networks

## About this repository
This repository contains the code related to the paper _Remote Tube-based MPC for Tracking Over Lossy Networks_, which has been submitted to the 2024 European Control Conference.
This paper addresses the problem of controlling constrained systems subject to disturbances in the case where controller and system are connected over a lossy network.
To do so, we propose a novel framework that splits the concept of tube-based model predictive control in two parts.
One runs locally on the system and is responsible for disturbance rejection, while the other runs remotely and provides optimal input trajectories that satisfy the system’s state and input constraints. Key to our approach is the presence of a nominal model and an ancillary controller on the local system.
Theoretical guarantees regarding the recursive feasibility and the tracking capabilities in the presence of disturbances and packet losses in both directions are provided.

In addition to the novel MPC algorithm, this repository also contains the implementation of several other MPC approach for constrained linear systems, implementation of two algorithms to approximate the minimal robust positively invariant set, an algorithm to determine the maximum admissible set, and also several functions that implement the Minkowski sum and Pontryagin difference for convex polytopes. For more details see the more detailed description of the repository below.

## Installation

First, clone the repository as follows:

```bash
git clone https://github.com/EricssonResearch/Robust-Tracking-MPC-over-Lossy-Networks.git
cd Robust-Tracking-MPC-over-Lossy-Networks
```
Set up a virtual environment (this step is **optional**)
```bash
python -m venv .LinMPCoverNet
source .LinMPCoverNet/bin/activate
```
Next, run 
```bash
pip install --upgrade pip
pip install .
```
to install the `LinearMPCOverNetworks` package.
Finally, if one wants to reproduce the results in our submitted paper in [ECC Results](./ECC%20Results/), one needs to run 
```bash
pip3 install PyBullet
```

## Content of the repository

### Model predictive control:

The [src/LinearMPCOverNetwoks](./src/LinearMPCOverNetworks/) folder contains several different MPC implementations for linear system with additive disturbances under constraints that lie in convex polytopes including the origin. The algorithms either bring the plant to the origin or track a piece-wise constant reference.
The MPC algorithms included are:

- D. Umsonst, F. Barbosa, "Remote Tube-based MPC for Tracking Over Lossy Networks" _(currently under review)_
- [D. Limon, I. Alvarado, T. Alamo, E.F. Camacho, "MPC for tracking piecewise constant references for constrained linear systems," 2008](https://www.sciencedirect.com/science/article/abs/pii/S0005109808001106)
- [M. Pezzutto, M. Farina, R. Carli and L. Schenato, "Remote MPC for Tracking Over Lossy Networks," 2022](https://ieeexplore.ieee.org/abstract/document/9452064)
- [D.Q. Mayne, M.M. Seron, S.V. Raković, "Robust model predictive control of constrained linear systems with bounded disturbances," 2005](https://www.sciencedirect.com/science/article/abs/pii/S0005109804002870)
- [D. Limon, I. Alvarado, T. Alamo, E.F. Camacho, "Robust tube-based MPC for tracking of constrained linear systems with additive disturbances," 2010](https://www.sciencedirect.com/science/article/abs/pii/S0959152409002169)

Examples on how these algorithms can be used are provided in the folder [Examples of Model Predictive Controllers](./Examples%20of%20Model%20Predictive%20Controllers/).

### ECC 2024 Results

The folder [ECC Results](./ECC%20Results/) contains scripts to reproduce the results of the paper _Remote Tube-based MPC for Tracking Over Lossy Networks_ and also a script to generate animations of the cartpole.

### Operations on convex polytopes

To be able to implement the robust tube-based MPC algorithms, we needed to implement several operations on convex polytopes, such as the Minkowski sum, the Pontryagin difference, and the support function of a polytope.
The algorithms build up on the [`polytope` package](https://github.com/tulip-control/polytope).

Furthermore, we implemented two algorithms to approximate the minimum robust positively invariant set of a linear discrete-time system with bounded additive disturbance. 
The algorithms implemented are found in 

- [S. V. Rakovic, E. C. Kerrigan, K. I. Kouramas and D. Q. Mayne, "Invariant approximations of the minimal robust positively Invariant set," 2005](https://ieeexplore.ieee.org/document/1406138)
- [M. S. Darup and D. Teichrib, "Efficient computation of RPI sets for tube-based robust MPC," 2019](https://ieeexplore.ieee.org/document/8796265)

Finally, we also implemented a method to calculate the terminal set of a linear MPC algorithm, via the maximum output admissible set.
This algorithm is based on
- [E. G. Gilbert and K. T. Tan, "Linear systems with state and control constraints: the theory and application of maximal output admissible sets," 1991](https://ieeexplore.ieee.org/document/83532) 

All these methods can be found in [`utils_polytope.py`](./src/LinearMPCOverNetworks/utils_polytope.py) and examples on how to use them are presented in the folder [Examples of Set Operations](./Examples%20of%20Set%20Operations/)

## Authors
David Umsonst [david.umsonst@ericsson.com](mailto:david.umsonst@ericsson.com) and Fernando S. Barbosa [fernando.dos.santos.barbosa@ericsson.com](mailto:fernando.dos.santos.barbosa@ericsson.com)