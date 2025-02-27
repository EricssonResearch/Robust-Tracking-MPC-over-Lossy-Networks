'''
In this file, we implement a class for a smart actuator as proposed in [1].
Furthermore, we also implement a child class of the smart actuator, which implements the consistent actuator described in Section IV.B of [2] 
together with the nominal plant model and ancillary controller described in Section IV.C in [2].

[1] M. Pezzutto, M. Farina, R. Carli and L. Schenato, "Remote MPC for Tracking Over Lossy Networks," in IEEE Control Systems Letters, vol. 6, pp. 1040-1045, 2022, doi: 10.1109/LCSYS.2021.3088749.
[2] D. Umsonst and F. S. Barbosa, "Remote Tube-based MPC for Tracking Over Lossy Networks," 2024 IEEE 63rd Conference on Decision and Control (CDC), Milan, Italy, 2024, pp. 1041-1048, doi: 10.1109/CDC56724.2024.10885830.
'''
import numpy as np

class SmartActuator:

    def __init__(self, K: np.ndarray):
        # initialize steady_state controller gain
        self._K=K
        # initialize internal timer
        self._t=0
        # intialize vector that saves the sequence of binary numbers that indicate if a packet has been received or not
        self._theta_t_vec=np.array([])
        # initialize internal variable that keeps track of the last time instance the remote controller has received a packet
        self._q_t=0
        # initialize variable that keeps track of the last time instance the smart actuator has accepted a received packet
        self._s_t=0

    def update_time(self):
        '''
        This function updates the internal timer
        '''
        self._t += 1

    def process_packet(self, packet: dict, x_t: np.ndarray, theta_t):
        '''
        This function processes a received packet on the local plant side, which is used to determine the next control input
        '''
        # extract control sequence
        U_t = packet['U_t']
        # extract q_t
        q_t = packet['q_t']
        
        # update \Theta_t 
        self.update_Theta_t(theta_t,q_t)
        # update st
        self.update_s_t()

        self.update_local_information(U_t)

        # compute control input based on the received control sequence and the current state
        u_t=self.compute_u_t(x_t)
        # create packet to be sent from the local plant to the remote controller
        plant_packet = self.encapsulate(x_t)
        # update internal timer
        self.update_time()

        return u_t, plant_packet


    def update_Theta_t(self, theta_t: int, q_t: int):
        '''
        This function calculates the consistency metric \Theta_t for a received packet according to equation (17) in [1] 
        '''
        # append the new value for theta_t
        self._theta_t_vec=np.append(self._theta_t_vec,theta_t)

        # calculate Theta_t according to equation (17) in [1]
        if theta_t==1:
            self._q_t=q_t
            self._Theta_t=np.prod(self._theta_t_vec[self._q_t+1:])
        else:
            self._Theta_t=0

        return self._Theta_t
    
    def update_s_t(self):
        '''
        This function updates s_t according to equation (18) in [1]
        '''
        self._s_t=int(self._Theta_t*self._t+(1-self._Theta_t)*self._s_t)

        return self._s_t
    
    def update_local_information(self, u_mpc_traj: np.ndarray):
        '''
        This function updates the local information stored on the smart actuator.
        '''
        # update locally stored trajectory 
        # if the real state and estimated state are consistent
        if self._Theta_t==1:
            self._u_traj=u_mpc_traj

    def compute_u_t(self, x_t: np.ndarray):
        '''
        This function computes the actual control input to the plant
        '''
        # obtain horizon of MPC controller
        N=self._u_traj.shape[1]-1
        # obtain number of inputs
        nu=self._u_traj.shape[0]

        # implementation smart actuator according to equation (19) in [1]
        if (int(self._t-self._s_t)<N):
            u=self._u_traj[:,int(self._t-self._s_t)]
        else:
            u=self._u_traj[:,-1]-self._K@x_t
        
        # make sure actuator signal has the right dimensions
        u.shape=(nu,1)
        return u
    
    def get_s_t(self):
        return self._s_t
    
    def get_Theta_t(self):
        return self._Theta_t
        
    def encapsulate(self, x_t: np.ndarray):
        '''
        This function encapsulates the plant state x_t at time t and s_t according to equation (4) in [1]
        '''
        packet = {}
        packet['x_t'] = x_t
        packet['s_t'] = self._s_t

        return packet
    
class ConsistentActuator(SmartActuator):
    '''
    This class is an extension of the SmartActuator class, which enables the remote robust tube-based tracking MPC over lossy networks
    '''
    def __init__(self, A: np.ndarray, B: np.ndarray, K: np.ndarray, K_plant: np.ndarray, x0: np.ndarray, is_extended_MPC_used: bool = False):

        super().__init__(K)
        # initialize nominal trajectory
        self._x_nom=x0
        # initialize nominal plant matrices
        self._A = A
        self._B = B
        # initialize ancillary controller gain
        self._K_plant = K_plant
        # get the dimensions of the input and state
        self._nu = B.shape[1]
        self._nx = A.shape[1]
        
        # flag to indicate if the extended MPC described in Section IV.F of [2] is used
        self._is_extended_MPC_used = is_extended_MPC_used

    def update_x_nom(self, u_nom: np.ndarray):
        '''
        This function updates the nominal trajectory x_nom on the plant side based on the nominal control input u_nom
        according to equation (4) of [2]
        '''
        u_nom.shape = (self._nu,1)
        self._x_nom = self._A @ self._x_nom + self._B @ u_nom

    def reset_x_nom(self, x_nom_0: np.ndarray):
        '''
        This function resets the nominal trajectory x_nom on the plant side to the received nominal plant state received over the network from the remote controller
        '''
        self._x_nom = x_nom_0

    def get_x_nom(self):
        '''
        This function returns the current state of the nominal plant
        '''
        return self._x_nom

    def ancillary_controller(self, u_nom_t: np.ndarray, x_t: np.ndarray, x_nom_t: np.ndarray):
        '''
        This function calculates the controller input determined by the ancillary controller based in the nominal control input u_nom_t, the nominal state x_nom_t, and the state x_t
        according to equation (5) of [2]
        '''
        u_t = u_nom_t - self._K_plant @ (x_t-x_nom_t)
        return u_t
    
    def process_packet(self, packet: dict, x_t: np.ndarray, theta_t: int):
        '''
        This function processes a received packet on the local plant side, which is used to determine the next control input as described in Section IV.B+C of [2]
        NOTE: The main difference to the SmartActuator class is that here we need to apply the ancillary controller to guarantee that the constraints are satisfied and the plant state
        evolves in a tube around the nominal state.
        '''
        # extract control sequence
        U_t = packet['U_t']
        # extract q_t
        q_t = packet['q_t']
        if 'x_nom_0' in packet.keys():
            x_nom_0_received = packet['x_nom_0']
            x_nom_0_received.shape = self._x_nom.shape
        else:
            x_nom_0_received = None

        # update \Theta_t 
        self.update_Theta_t(theta_t,q_t)
        # update st
        self.update_s_t()
        # update local information
        self.update_local_information(u_mpc_traj=U_t, x_nom_t_received=x_nom_0_received)
        
        # get current nominal state
        x_nom_t = self.get_x_nom()
        # compute nominal control input based on the smart actuator from the SmartActuator class
        u_nom_t = self.compute_u_t(x_nom_t)
        # calculate the control input based on the ancillary controller
        u_t = self.ancillary_controller(u_nom_t,x_t,x_nom_t)
        # create packet to be sent from the local plant to the remote controller
        if self._is_extended_MPC_used:
            plant_packet = self.encapsulate_extended(x_t)
        else:
            plant_packet = self.encapsulate(x_nom_t)
        # update nominal state based on the nominal control input u_nom_t
        self.update_x_nom(u_nom_t)
        # update internal timer
        self.update_time()

        return u_t, plant_packet
    
    def update_local_information(self, u_mpc_traj: np.ndarray, x_nom_t_received: np.ndarray = None):
        '''
        This function updates the local information stored on the consistent actuator, which includes the actuator input trajectory and the first state of the nominal plant
        '''
        if self._Theta_t==1:
            self._u_traj=u_mpc_traj
            if x_nom_t_received is not None:
                self.reset_x_nom(x_nom_t_received)

    def encapsulate_extended(self, x_t: np.ndarray):
        '''
        This function implements the extended encapsulation needed for the extended tube MPC proposed in Section IV.F of [2] 
        '''
        packet = super().encapsulate(x_t)
        packet['x_nom_t'] = self.get_x_nom()

        return packet