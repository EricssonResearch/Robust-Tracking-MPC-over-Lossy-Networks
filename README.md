# Remote Tube-based MPC for Tracking Over Lossy Networks

This repository will contain code for the related ECC 2024 submission, currently under peer review.

## Abstract
This paper addresses the problem of controlling constrained systems subject to disturbances in the case where controller and system are connected over a lossy network.
To do so, we propose a novel framework that splits the concept of tube-based model predictive control in two parts. 
One runs locally on the system and is responsible for disturbance rejection, while the other runs remotely and provides optimal input trajectories that satisfy the system's state and input constraints.
Key to our approach is the presence of a nominal model and an ancillary controller on the local system.
Theoretical guarantees regarding the recursive feasibility and the tracking capabilities in the presence of disturbances and packet losses in both directions are provided.
To test the efficacy of the proposed approach, we compare it to a state-of-the-art solution in the case of controlling a cartpole system. 
Extensive simulations are carried with both linearized and nonlinear system dynamics, as well as different packet loss probabilities and disturbances.

## Authors
David Umsonst [david.umsonst@ericsson.com](mailto:david.umsonst@ericsson.com) and Fernando S. Barbosa [fernando.dos.santos.barbosa@ericsson.com](mailto:fernando.dos.santos.barbosa@ericsson.com)
