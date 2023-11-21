'''
In this file, we implement a class for a tracking MPC according to [1] and also the functionality proposed in [2], which enables to run the tracking MPC remotely.

NOTE: In [1] and [2], it is assumed that the constraints for the inputs and states are in the same set, e.g., (x,u)\inXU, while we assume here that there are 
separate constraints on the states and inputs, i.e., x \in X and u \in U

[1] D. Limon, I. Alvarado, T. Alamo, E.F. Camacho, MPC for tracking piecewise constant references for constrained linear systems, Automatica, Volume 44, Issue 9, 2008,
[2] M. Pezzutto, M. Farina, R. Carli and L. Schenato, "Remote MPC for Tracking Over Lossy Networks," in IEEE Control Systems Letters, vol. 6, pp. 1040-1045, 2022, doi: 10.1109/LCSYS.2021.3088749.
'''

import numpy as np
import cvxpy as cp
import control as ct
import polytope as pc
import LinearMPCOverNetworks.utils_polytope as up
import LinearMPCOverNetworks.RegulatorMPC as RegulatorMPC
import time

class TrackingMPC(RegulatorMPC.RegulatorMPC):

    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, N: int, lambda_param= 0.99999) -> None:
        super().__init__(A,B,Q,R,N)

        # Calculate stabilizing controller based on LQR with same cost matrices as for MPC
        K, S, _=ct.dlqr(self._A, self._B, self._Q,self._R)
        self._K=K
        
        # Calculate cost matrix for steady state based on steady state controller
        Q_lyap=self._Q+self._K.T@self._R@self._K
        Q_lyap_sym = (Q_lyap+Q_lyap.T)/2
        self._P=ct.dlyap(self._A-self._B@self._K, Q_lyap_sym)

        # steady state cost matrix
        self._Tout=10*self._P 
        # set lambda parameter for calculation of terminal set of the MPC
        self._lambda = lambda_param
        # determine closed loop system matrix when the stabilizing controller is used
        self._Acl = self._A-self._B@self._K

        # initialize terminal set
        self._Xf = None
        # initialize list to save computational times of the solver of the MPC problem
        self._computational_times = []

    def determine_packet(self,x_hat: np.ndarray, ref: np.ndarray, q_t):
        '''
        This function determines the packet to be sent from the controller to the plant when the network is lossy according to (3) in [2]
        '''
        x_hat.shape = (self._nx, )
        # first solve the MPC problem for a given reference ref and a state estimate x_hat
        start = time.time()
        _, u_mpc_traj, x_ss, u_ss = self.solve_optimization_problem(x_hat, ref)
        end = time.time()
        # record the time it took to solve the MPC problem
        self._computational_times.append(end-start)

        # then encapsulate the contoller signal and q_t
        controller_packet = self.encapsulate(u_mpc_traj, x_ss, u_ss, q_t)

        return controller_packet
    
    def generate_optimization_problem(self):
        '''
        This function generates a parametrized version of the optimization problem solved in the MPC
        By doing so we can speed up the optimization, since the optimization problem does not have to be re-instantiated each time it is solved
        '''
        # determine polyhedra constraints for...
        if self._U is not None:        
            Hu, hu =  self._U.A, self._U.b # ... the input
            hu.shape = (hu.shape[0], )
        
        if self._X is not None:
            Hx, hx =  self._X.A, self._X.b # ... the state
            hx.shape = (hx.shape[0], )

        if self._Xf is not None:
            HxN, hxN = self._Xf.A, self._Xf.b # ... the final state
            hxN.shape = (hxN.shape[0], )

        self._x_mpc = cp.Variable((self._nx,self._N+1))
        self._u_mpc = cp.Variable((self._nu,self._N))
        self._x_bar = cp.Variable((self._nx,))
        self._u_bar = cp.Variable((self._nu,))
        self._x_init_param = cp.Parameter((self._nx,))
        self._ref_param = cp.Parameter(self._nx,)

        constraints=[self._x_mpc[:,0]==self._x_init_param]
        cost = 0
        for i in range(self._N):
            cost += cp.quad_form(self._x_mpc[:,i+1]-self._x_bar,self._Q)+cp.quad_form(self._u_mpc[:,i]-self._u_bar,self._R)
            # dynamic constraint x(k+1)=Ax(k)+Bu(k)
            constraints += [self._x_mpc[:,i+1]==self._A @ self._x_mpc[:,i] + self._B @ self._u_mpc[:,i]]
            if self._X is not None:
                constraints += [Hx@self._x_mpc[:,i]<=hx] # state constraint x(k)\in X
            if self._U is not None:
                constraints += [Hu@self._u_mpc[:,i]<=hu] # output constraint u(k)\in U

        # add steady state constraints
        constraints += [(self._A-np.eye(self._nx))@self._x_bar+self._B@self._u_bar==np.zeros((self._nx,))]
        # add cost for steady state, reference and terminal constraint
        cost += cp.quad_form(self._x_mpc[:,-1]-self._x_bar,self._P)
        cost += cp.quad_form(self._x_bar-self._ref_param,self._Tout)

        # add terminal constraints
        if self._Xf is None:
            # if there is no terminal constraint set, enforce that final state is equal to the steady state variable
            constraints += [self._x_mpc[:,-1]==self._x_bar]
        else:
            constraints += [HxN[:,0:self._nx]@self._x_mpc[:,self._N]+HxN[:,self._nx:2*self._nx]@self._x_bar+HxN[:,2*self._nx:]@self._u_bar<=hxN]

        # generate objective
        objective = cp.Minimize(cost)
        # generate the optimization problem
        self._prob = cp.Problem(objective, constraints)
    
    def solve_optimization_problem(self, x_init: np.ndarray, ref: np.ndarray, verbose_MPC=False):
        '''
        This function solves the MPC problem given by equation (12) in [2]
        '''
        # setting the reference parameter in the parameterized optimization problem to ref
        ref.shape = (self._nx,)
        self._ref_param.value = ref

        # setting the initial state parameter in the parameterized optimization problem to x_init
        x_init.shape = (self._nx,)
        self._x_init_param.value = x_init

        # solve the optimization problem
        self._prob.solve(verbose=verbose_MPC, solver = self._solver, tol_gap_abs = 1e-7, tol_gap_rel = 1e-7)

        # print status of solver in case it is not optimal
        if self._prob.status != "optimal":
            print(f"Status of tracking MPC is {self._prob.status}")
        
        return self._x_mpc.value, self._u_mpc.value, self._x_bar.value, self._u_bar.value
    
    def get_steady_state_controller_gain(self):
        '''
        This function returns the steady state controller gain, which is used for determining the terminal set of the MPC
        '''
        return self._K
    
    def encapsulate(self, u_mpc: np.ndarray, x_bar: np.ndarray, u_bar: np.ndarray, q_t: int):
        '''
        This encapsulates the control signals and q_t according to equation (3) in [2]
        '''
        if x_bar is not None:
            u_ss=u_bar+self._K @ x_bar
            u_ss.shape=(u_mpc.shape[0], 1)
            U_t = np.hstack((u_mpc,u_ss))
        else:
            U_t = None

        packet = {}
        packet['U_t'] = U_t
        packet['q_t'] = q_t

        return packet
    
    def determine_Xf(self):
        '''        
        This function determines the terminal set of the MPC as the invariant admissible set for the augmented system according to equations (7), (9), and (10) of [2]
        ''' 
        Hu, hu =  self._U.A.copy(), self._U.b.copy()
        Hx, hx =  self._X.A.copy(), self._X.b.copy()
        # setup augmented system matrix as in equation (7) of [2]
        A_e = np.c_[np.r_[self._Acl, np.zeros((self._nx,self._nx)), np.zeros((self._nu,self._nx))],
                  np.r_[self._B@self._K, np.eye(self._nx), np.zeros((self._nu,self._nx))], 
                  np.r_[self._B, np.zeros((self._nx,self._nu)), np.eye(self._nu)]]
        
        # setup the polytope in which the augmented system state should lie
        Hcl = np.c_[np.r_[Hx,-Hu@self._K,np.zeros((2*self._nx,self._nx)),np.zeros((2*self._nu,self._nx))],
                    np.r_[np.zeros((2*self._nx,self._nx)),Hu@self._K,Hx,np.zeros((2*self._nu,self._nx))],
                    np.r_[np.zeros((2*self._nx,self._nu)),Hu,np.zeros((2*self._nx,self._nu)),Hu]]

        hcl = np.r_[hx,
                    hu,
                    1/self._lambda*hx,
                    1/self._lambda*hu]

        XXbarUbar=pc.Polytope(Hcl,hcl)
        
        # calculate the terminal set as the maximum admissible output set
        Xf = up.calculate_maximum_admissible_output_set(A_e, XXbarUbar)

        self._Xf =  Xf

    def setup_optimization(self):
        # this functions sets up the MPC optimization problem by ...
        # ... determining the terminal state constraint set
        self.determine_Xf()
        # ... generating the parametrized optimization problem of the MPC
        self.generate_optimization_problem()

    def get_computational_times(self):
        return self._computational_times

    def reset_computational_times(self):
        self._computational_times = []