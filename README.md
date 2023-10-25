# Remote Tube-based MPC for Tracking Over Lossy Networks

This repository will contain code for the related ECC 2024 submission, currently under peer review.

## Abstract
In this paper, we present a remote reference tracking approach for constrained systems, which handles both network imperfections as well as a disturbance acting on the local plant.
Our approach consists of a remote model predictive controller, which acts on an estimate of the nominal plant state. 
The optimal input trajectory is sent over an imperfect network to the local plant.
In the local plant, the input trajectory is checked for consistency and then applied to a nominal process model. 
The nominal process model together with an ancillary controller are used to determine the actual control input to the plant.

## Authors
David Umsonst [david.umsonst@ericsson.com](mailto:david.umsonst@ericsson.com) and Fernando S. Barbosa [fernando.dos.santos.barbosa@ericsson.com](mailto:fernando.dos.santos.barbosa@ericsson.com)
