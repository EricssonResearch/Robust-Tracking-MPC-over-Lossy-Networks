'''
In this script, we implement a tube tracking MPC according to [1].
This is then combined with the results of [2], where we enable to use the tube tracking MPC of [1] over a lossy network, which is inspired by [3]

NOTE: In [1], it is assumed that the constraints for the inputs and states are in the same set, e.g., (x,u)\inXU, while we assume here that there are 
separate constraints on the states and inputs, i.e., x \in X and u \in U

[1] D. Limon, I. Alvarado, T. Alamo, E.F. Camacho, Robust tube-based MPC for tracking of constrained linear systems with additive disturbances, Journal of Process Control, Volume 20, Issue 3, 2010
[2] D. Umsonst, F. Barbosa, "Remote Tube-based MPC for Tracking Over Lossy Networks", under review
[3] M. Pezzutto, M. Farina, R. Carli and L. Schenato, "Remote MPC for Tracking Over Lossy Networks," in IEEE Control Systems Letters, vol. 6, pp. 1040-1045, 2022, doi: 10.1109/LCSYS.2021.3088749.
'''

import numpy as np
import cvxpy as cp
import LinearMPCOverNetworks.utils_polytope as up
import polytope as pc
from LinearMPCOverNetworks.TubeRegulatorMPC import TubeRegulatorMPC
import time

class TubeTrackingMPC(TubeRegulatorMPC):

    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, N: int, lambda_param= 0.99999):
        super().__init__(A,B,Q,R,N)
        self._lambda = lambda_param

        # steady state cost matrix
        self._Tout=10*self._P
        # initialize controller matrix for ancillary controller
        self._K_ancillary = None
        #initialize closed-loop system matrix when the ancillary controller is used
        self._Acl_plant = None
        # initialize list to save computational times of the solver of the MPC problem
        self._computational_times = []

    def determine_Xf(self):
        '''        
        This function determines the terminal set of the MPC as the invariant admissible set for the augmented system according to equations (7), (9), and (10) of [3]
        ''' 
        Hu, hu =  self._Uc.A.copy(), self._Uc.b.copy()
        Hx, hx =  self._Xc.A.copy(), self._Xc.b.copy()

        # setup augmented system matrix as in equation (7) of [2]
        A_cl_w_ss = np.c_[np.r_[self._Acl, np.zeros((self._nx,self._nx)), np.zeros((self._nu,self._nx))],
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
        Xf = up.calculate_maximum_admissible_output_set(A_cl_w_ss, XXbarUbar)
   
        self._Xf =  Xf

    def determine_mRPI(self, W: pc.Polytope, epsilon = 1*10**(-4), Acl = None, rpi_method = 0):
        '''
        This function calculates the minimum robust positively invariant (mRPI) for a given disturbance polytope W,
        a precision epsilon, a desired closed-loop system matrix Acl and a chosen method rpi_method.
        '''
        # if closed-loop matrix is None...
        if Acl is None:
            # ... set it to the one of the closed-loop system with the steady-state LQR controller
            # ... if there is no ancillary controller specified 
            if self._Acl_plant is None:
                Acl = self._Acl
            else:
                # otherwise use the specified ancillary controller closed-loop system matric
                Acl = self._Acl_plant

        # if there is no ancillary controller specified...
        if self._K_ancillary is None:
            #... use the steady state LQR controller as the ancillary controller
            K = self._K
        else:
            # ... otherwise use the ancillary controller
            K = self._K_ancillary

        Fs = super().determine_mRPI(W, epsilon, Acl = Acl, K=K, rpi_method = rpi_method)

        return Fs

    def tighten_constraints(self):
        '''
        This function thightens the given constraints on the state and actuator input
        based on the minimum robust positively invariant set via the Pontryagin set difference
        as similar to how it is done in equation (7) of [1]
        '''
        # if there is an ancillary controller, it is used to tighten the controller constraints
        # otherwsie the steady-state LQR controller of the MPC is used as a ancillary controller
        if self._K_ancillary is None:
            self._Uc = up.pont_diff(self._U, up.scale(self._Z,-self._K))
        else:
            self._Uc = up.pont_diff(self._U,up.scale(self._Z,-self._K_ancillary))
        self._Xc = up.pont_diff(self._X, self._Z)
        
    def generate_optimization_problem(self, fixed_initial_state = False):
        '''
        This function generates a parametrized version of the optimization problem solved in the MPC
        by doing so we can speed up the optimization, since the optimization problem does not have to be re-instantiated each time it is solved
        '''
        # determine polyhedra constraints for...        
        Hu, hu =  self._Uc.A, self._Uc.b # ... the input
        hu.shape = (hu.shape[0], )
        Hx, hx =  self._Xc.A, self._Xc.b # ... the state
        hx.shape = (hx.shape[0], )
        HxN, hxN =  self._Xf.A, self._Xf.b # ... the final state
        hxN.shape = (hxN.shape[0], )
        
        self._x_mpc = cp.Variable((self._nx,self._N+1))
        self._u_mpc = cp.Variable((self._nu,self._N))
        self._x_bar = cp.Variable((self._nx,))
        self._u_bar = cp.Variable((self._nu,))
        self._x_init_param = cp.Parameter((self._nx,))
        self._ref_param = cp.Parameter((self._nx,))

        # add constraint on first state of the MPC
        if fixed_initial_state:
            # either fix it to the the initial parameter x_init_param
            constraints=[self._x_init_param-self._x_mpc[:,0]==0]
        else:
            # or let it be in a tube around the the initial parameter
            Hz, hz = self._Z.A, self._Z.b  # ... the mRPI
            hz.shape = (hz.shape[0], )
            constraints=[Hz@(self._x_init_param-self._x_mpc[:,0])<=hz]
        cost = 0

        for i in range(self._N):
            cost += cp.quad_form(self._x_mpc[:,i]-self._x_bar,self._Q)+cp.quad_form(self._u_mpc[:,i]-self._u_bar,self._R)
            # dynamic constraint x(k+1)=Ax(k)+Bu(k)
            constraints += [self._x_mpc[:,i+1]==self._A @ self._x_mpc[:,i] + self._B @ self._u_mpc[:,i]] # dynamics constraint
            constraints += [Hx@self._x_mpc[:,i]<=hx] # state constraint x(k)\in X
            constraints += [Hu@self._u_mpc[:,i]<=hu] # input constraint u(k)\in U

        # add cost for steady state, reference and terminal constraint
        cost += cp.quad_form(self._x_mpc[:,self._N]-self._x_bar,self._P)
        cost += cp.quad_form(self._x_bar-self._ref_param,self._Tout)

        # add steady state constraints
        constraints += [(self._A-np.eye(self._nx))@self._x_bar+self._B@self._u_bar==np.zeros((self._nx,))]
        # add terminal constraints
        constraints += [HxN[:,0:self._nx]@self._x_mpc[:,self._N]+HxN[:,self._nx:2*self._nx]@self._x_bar+HxN[:,2*self._nx:]@self._u_bar<=hxN]
        # generate objective
        objective = cp.Minimize(cost)
        # generate the optimization problem
        self._prob = cp.Problem(objective, constraints)
        
        # resetting shape to keep other functionalities working
        hxN.shape = (hxN.shape[0], 1)

    def setup_optimization(self, W, fixed_initial_state = False, rpi_method = 0):
        # this functions sets up the MPC optimization problem by ...
        
        # ... determining the minimum robust positively invariant (mrpi) set
        self.determine_mRPI(W, rpi_method = rpi_method)
        # ... using the mrpi to tighten the constraints
        self.tighten_constraints()
        # ... determine the terminal set for the MPC trajectory
        self.determine_Xf()
        # ... generate the MPC optimization problem in its parametrized form
        self.generate_optimization_problem(fixed_initial_state)

    def solve_optimization_problem(self, x_init: np.ndarray, ref: np.ndarray):
        '''
        This function solves the optimization problem of the MPC
        '''
        # setting the reference parameter in the parameterized optimization problem to ref
        ref.shape =(self._nx,)
        self._ref_param.value = ref

        # setting the initial state parameter in the parameterized optimization problem to x_init
        x_init.shape =(self._nx,)
        self._x_init_param.value = x_init

        # solve the optimization problem
        self._prob.solve(verbose = False, solver = self._solver, tol_gap_abs = 1e-7, tol_gap_rel = 1e-7)
        
        if self._prob.status != "optimal":
            print(f"Status of tube tracking MPC is: {self._prob.status}")

        # return the optimal nominal state trajectory, the optimal nominal input trajectory, and the optimal steady state and steady-state input 
        u_nom = self._u_mpc.value
        x_nom = self._x_mpc.value
        u_steady_state = self._u_bar.value
        x_steady_state = self._x_bar.value

        return x_nom, u_nom, x_steady_state, u_steady_state
    
    def determine_packet(self, x_hat: np.ndarray, ref: np.ndarray, q_t: int):
        '''
        This function determines the packet to be sent from the controller to the plant when the network is lossy according to (10) in [2]
        '''
        x_hat.shape = (self._nx, )
        # first solve the MPC problem
        start = time.time()
        x_nom_traj, u_nom_traj, x_steady_state, u_steady_state = self.solve_optimization_problem(x_hat, ref)
        end = time.time()
        self._computational_times.append(end-start)
        # then encapsulate the contoller signal and q_t according to equation (9)
        controller_packet = self.encapsulate(u_nom_traj, u_steady_state, x_steady_state, q_t)

        return controller_packet
    
    def encapsulate(self, u_nom_traj: np.ndarray, u_steady_state: np.ndarray, x_steady_state: np.ndarray, q_t: int):
        '''
        This encapsulates the control signals and q_t according to equation (9) in [2]
        '''
        if x_steady_state is not None:
            # encapsulates the control signals and q_t according to equation (9) of [2]
            u_ss=u_steady_state + self._K @ x_steady_state
            u_ss.shape=(u_nom_traj.shape[0], 1)
            U_t = np.hstack((u_nom_traj,u_ss))
        else:
            U_t = None

        packet = {}
        packet['U_t'] = U_t
        packet['q_t'] = q_t

        return packet
    
    def set_ancillary_controller_gain(self, K_ancillary):
        self._K_ancillary = K_ancillary
        self._Acl_plant = self._A-self._B@K_ancillary
    
    def get_ancillary_controller_gain(self):
        if self._K_ancillary is None:
            return self._K
        else:
            return self._K_ancillary
    
    def get_steady_state_controller_gain(self):
        return self._K

    def get_computational_times(self):
        return self._computational_times
    
    def reset_computational_times(self):
        self._computational_times = []


class ExtendedTubeTrackingMPC(TubeTrackingMPC):
    '''
    This class implements the Extended Tube MPC approach proposed in Section IV.F of [2]
    '''
    def generate_optimization_problem_when_packet_received(self, W):
        '''
        This function generates a parametrized version of the optimization problem solved in the MPC when a packet has been received at the remote controller.
        by doing so we can speed up the optimization, since the optimization problem does not have to be re-instantiated each time it is solved
        '''
        # determine polyhedra constraints for...        
        Hu, hu =  self._Uc.A, self._Uc.b # ... the input
        hu.shape = (hu.shape[0], )
        Hx, hx =  self._Xc.A, self._Xc.b # ... the state
        hx.shape = (hx.shape[0], )
        HxN, hxN =  self._Xf.A, self._Xf.b # ... the final state
        hxN.shape = (hxN.shape[0], )
        # Obtain the Pontryagin difference between the mRPI and the disturbance set...
        ZmW = up.pont_diff(self._Z, W)
        Hz, hz = ZmW.A, ZmW.b # ... and get the polyhedral constraints from it
        hz.shape = (hz.shape[0], )
        
        self._x_mpc_packet_received = cp.Variable((self._nx,self._N+1))
        self._u_mpc_packet_received = cp.Variable((self._nu,self._N))
        self._x_bar_packet_received = cp.Variable((self._nx,))
        self._u_bar_packet_received = cp.Variable((self._nu,))
        self._x_init_param_packet_received = cp.Parameter((self._nx,))
        self._ref_param_packet_received = cp.Parameter((self._nx,))

        # add constraint on initial state
        constraints=[Hz@(self._x_init_param_packet_received-self._x_mpc_packet_received[:,0])<=hz]

        cost = 0

        for i in range(self._N):
            cost += cp.quad_form(self._x_mpc_packet_received[:,i]-self._x_bar_packet_received,self._Q)+cp.quad_form(self._u_mpc_packet_received[:,i]-self._u_bar_packet_received,self._R)
            constraints += [self._x_mpc_packet_received[:,i+1]==self._A @ self._x_mpc_packet_received[:,i] + self._B @ self._u_mpc_packet_received[:,i]] # dynamics constraint
            constraints += [Hx@self._x_mpc_packet_received[:,i]<=hx] # state constraint
            constraints += [Hu@self._u_mpc_packet_received[:,i]<=hu] # output constraint

        cost += cp.quad_form(self._x_mpc_packet_received[:,self._N]-self._x_bar_packet_received,self._P)
        cost += cp.quad_form(self._x_bar_packet_received-self._ref_param_packet_received, self._Tout)

        # add steady state constraints
        constraints += [(self._A-np.eye(self._nx))@self._x_bar_packet_received+self._B@self._u_bar_packet_received==np.zeros((self._nx,))]
        constraints += [HxN[:,0:self._nx]@self._x_mpc[:,self._N]+HxN[:,self._nx:2*self._nx]@self._x_bar_packet_received+HxN[:,2*self._nx:]@self._u_bar<=hxN]

        objective = cp.Minimize(cost)
        self._prob_packet_received = cp.Problem(objective, constraints)
        
        # resetting shape to keep other functionalities working
        hxN.shape = (hxN.shape[0], 1)

    def setup_optimization(self, W, fixed_initial_state = False, rpi_method = 0):
        # setup optimization problem when packet has not been received
        super().setup_optimization(W, fixed_initial_state = fixed_initial_state, rpi_method = rpi_method)
        # setup optimization problem when packet has been received
        self.generate_optimization_problem_when_packet_received(W)
    
    def solve_optimization_problem(self, x_init: np.ndarray, ref: np.ndarray, gamma_t = 0):
        '''
        This function solves the optimization problem of the MPC
        '''
        # check if a packet has been received
        if gamma_t == 1:
            # update initial parameter representing the inital state and the reference
            x_init.shape = (self._nx, )
            self._x_init_param_packet_received.value = x_init
            ref.shape = (self._nx, )
            self._ref_param_packet_received.value = ref
            
            # solve the optimization problem
            self._prob_packet_received.solve(verbose = False, solver = self._solver, tol_gap_abs = 1e-7, tol_gap_rel = 1e-7)
            
            # print status of solver in case it is not optimal
            if self._prob_packet_received.status != "optimal":
                print(f"Status of extended tube MPC when packet has been received: {self._prob_packet_received.status}")
            
            # return the optimal nominal state trajectory, the optimal nominal input trajectory, and the optimal steady state and steady-state input 
            u_nom = self._u_mpc_packet_received.value
            x_nom = self._x_mpc_packet_received.value
            u_steady_state = self._u_bar_packet_received.value
            x_steady_state = self._x_bar_packet_received.value
        else:
            # update initial parameter representing the inital state and the reference
            self._x_init_param.value = x_init
            self._ref_param.value = ref
            
            # solve the optimization problem
            self._prob.solve(verbose = False, solver = self._solver, tol_gap_abs = 1e-7, tol_gap_rel = 1e-7)
            
            # print status of solver in case it is not optimal
            if self._prob.status != "optimal":
                print(f"Status of extended tube MPC when packet has not been received: {self._prob.status}")
            
            # return the optimal nominal state trajectory, the optimal nominal input trajectory, and the optimal steady state and steady-state input 
            u_nom = self._u_mpc.value
            x_nom = self._x_mpc.value
            u_steady_state = self._u_bar.value
            x_steady_state = self._x_bar.value

        return x_nom, u_nom, x_steady_state, u_steady_state
    
    def determine_packet(self, x_hat, ref, q_t, gamma_t = 0):
        '''
        This function determines the packet to be sent from the controller to the plant when the network is lossy.
        '''
        x_hat.shape = (self._nx, )
        # first solve the MPC problem
        start = time.time()
        x_nom_traj, u_nom_traj, x_steady_state, u_steady_state = self.solve_optimization_problem(x_hat, ref, gamma_t)
        end = time.time()
        self._computational_times.append(end-start)
        # then encapsulate the contoller signal and q_t according to equation (3)
        controller_packet = self.encapsulate(u_nom_traj, u_steady_state, x_steady_state, q_t)
        if x_nom_traj is not None:
            x_nom_0 = x_nom_traj[:,0]
        else:
            x_nom_0 = None
        # add nominal state to controller packet 
        controller_packet['x_nom_0'] = x_nom_0
        return controller_packet, x_nom_0