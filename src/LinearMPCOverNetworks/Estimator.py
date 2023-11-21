'''
This file implements a class for a state estimator as proposed in Section III.B of [1] and the robust state estimator proposed in Section IV.F of [2].

[1] M. Pezzutto, M. Farina, R. Carli and L. Schenato, "Remote MPC for Tracking Over Lossy Networks," in IEEE Control Systems Letters, vol. 6, pp. 1040-1045, 2022, doi: 10.1109/LCSYS.2021.3088749.
[2] D. Umsonst, F. Barbosa, "Remote Tube-based MPC for Tracking Over Lossy Networks", under review
'''
import numpy as np

class Estimator:

    def __init__(self, A: np.ndarray, B: np.ndarray, K: np.ndarray, x0: np.ndarray, N: int):
        # initialize system matrices
        self._A = A
        self._B = B
        # initialize controller matrices
        self._K = K
        # intialize initial state of the estimator
        self._x_hat=x0
        # initialize internal timer
        self._t = 0
        # initialize variable that keeps track of when the last packet has been received
        self._q_t = 0
        # initialize length of the control sequence 
        self._N = N
        # initialize list of control sequences
        self._controlSequences=[]

    def update_time(self):
        '''
        This function updates the internal timer
        '''
        self._t += 1

    def store_sent_control_sequence(self,Ut: np.ndarray):
        '''
        This function stores the control sequence Ut sent from the remote controller to the plant.
        The reason for that is that the estimator needs to have the sent control sequences available when updating the estimate in
        the update_estimate() function.
        For now we simply append the received control sequence to the list of control sequences.
        ''' 
        self._controlSequences.append(Ut)

    def update_estimate(self, packet: dict, gamma_t: int):
        '''
        This function updates the state estimate based on the packet sent by the plant to the remotec ontroller and gamma_t, where gamma_t=1 indicates the packet has been received successfully.
        The estimator dynamics are given by equations (13)-(15) in [1]
        '''
        # extract content of packet
        x_t=packet['x_t']
        s_t=packet['s_t']

        # if the packet has been received...
        if gamma_t == 1:
            # ... obtain the control sequence uses by the plant
            u_sequ=self._controlSequences[s_t]
            # ... determine the control input \hat{u}(k|k)
            if (self._t-s_t<self._N):
                u_t=u_sequ[:,self._t-s_t]
            else:
                u_t=u_sequ[:,-1]-self._K@x_t

            u_t.shape=(u_sequ.shape[0],1)

            # ... update estimator
            self._x_hat=self._A @ x_t + self._B @ u_t

        # if the packet has not been received...
        else:
            # ... apply the first control input of the last sequence
            u_sequ=self._controlSequences[-1]
            u_t=u_sequ[:,0]
            u_t.shape=(u_sequ.shape[0],1)
            # ... update estimator
            self._x_hat=self._A @ self._x_hat + self._B @ u_t

        self.update_q_t(gamma_t)

        self.update_time()
    
    def get_estimate(self):
        '''
        This function returns the current estimate x_hat of the plant state x
        '''
        x_hat = self._x_hat[:]
        return x_hat
    
    def update_q_t(self, gamma_t: int):
        '''
        This function updates q_t for a given gamma_t. 
        Here, q_t keeps track of when the last packet has been received indicated by gamma_t=1
        '''
        self._q_t = gamma_t * self._t + (1-gamma_t) * self._q_t
    
    def get_qt(self):
        '''
        This function that returns the current value of q_t
        '''
        return self._q_t


class RobustEstimator(Estimator):
    '''
    This class implements the extended version of the estimator described in Section IV.F of [2]
    '''
    def __init__(self, A: np.ndarray, B: np.ndarray, K: np.ndarray, K_plant: np.ndarray, x0: np.ndarray, N: int):
        super().__init__(A, B, K, x0, N)

        # initialize the ancillary controller used on the local plant
        self._K_plant = K_plant
        # initialize the optimal initial state of the nominal trajectory from the MPC
        self._initial_nominal_states_from_MPC = None

    def update_estimate(self, packet: dict, gamma_t: int):
        '''
        This function updates the state estimate based on the packet sent by the plant to the remote and gamma_t, where gamma_t=1 indicates the packet has been received successfully.
        The estimator dynamics are given by equations (13)-(15) in [1], where we used the values for \hat{x}(k|k) and \hat{u}(k|k) as proposed in [2]
        '''
        # extract content of packet
        x_t=packet['x_t']
        s_t=packet['s_t']
        x_nom_t = packet['x_nom_t']

        # make sure the received states are in the right shape
        x_t.shape = (self._K_plant.shape[1],1)
        x_nom_t.shape = (self._K_plant.shape[1],1)

        # if the packet has been received...
        if gamma_t == 1:
            # ... obtain the control sequence uses by the plant
            u_sequ=self._controlSequences[s_t]
            # ... determine the nominal control input
            if (self._t-s_t<self._N):
                u_nom_t=u_sequ[:,self._t-s_t]
            else:
                u_nom_t=u_sequ[:,-1]-self._K@x_nom_t

            u_nom_t.shape=(u_sequ.shape[0],1)
            # obtain \hat{u}(k|k)
            u_t = u_nom_t - self._K_plant@(x_t-x_nom_t)
            # ... update estimator
            self._x_hat=self._A @ x_t + self._B @ u_t

        # if the packet has not been received...
        else:
            # ... get current optimal initial nominal state from MPC
            x_nom_0 = self._initial_nominal_states_from_MPC
            x_nom_0.shape =  (self._K_plant.shape[1],1)
            # ... apply the first control input of the last sequence
            u_sequ=self._controlSequences[-1]
            u_nom_t=u_sequ[:,0]
            u_nom_t.shape=(u_sequ.shape[0],1)
            # ... update estimator
            self._x_hat=self._A @ x_nom_0 + self._B @ u_nom_t

        self.update_q_t(gamma_t)

        self.update_time()

    def store_current_optimal_inital_nominal_plant_states(self, x_nom_0: np.ndarray):
        '''
        This function stores the latest optimal intial value of the nominal state trajectory produced by the MPC
        ''' 
        self._initial_nominal_states_from_MPC=x_nom_0