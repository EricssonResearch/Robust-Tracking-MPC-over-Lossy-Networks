'''
In this file, we implement a tube-based regulator MPC as proposed in [1].

[1] D.Q. Mayne, M.M. Seron, S.V. Raković, Robust model predictive control of constrained linear systems with bounded disturbances, Automatica, Volume 41, Issue 2, 2005, Pages 219-224, https://doi.org/10.1016/j.automatica.2004.08.019.
'''

import numpy as np
import cvxpy as cp
import control as ct
import polytope as pc
import LinearMPCOverNetworks.utils_polytope as up
from LinearMPCOverNetworks.RegulatorMPC import RegulatorMPC

class TubeRegulatorMPC(RegulatorMPC):

    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, N: int) -> None:
        super().__init__(A,B,Q,R,N)

        K, P, _ =ct.dlqr(A, B, Q, R)
        self._K = K
        Q_lyap=self._Q+self._K.T@self._R@self._K
        Q_lyap_sym = (Q_lyap+Q_lyap.T)/2
        self._P = ct.dlyap(self._A-self._B@self._K,Q_lyap_sym)
        self._Acl = A - B@K

    def determine_mRPI(self, W: pc.Polytope, eps_var = 1.9*10**(-5), Acl = None, rpi_method = 0, K = None):
        '''
        This function calculates the minimum robust positively invariant (mRPI) for a given disturbance polytope W,
        a precision epsilon and a desired closed-loop system matrix Acl
        '''
        # no matrix K is given set it to the steady-state controller matrix
        if K is None:
            K = self._K
        # if no matrix Acl is given, set it to the private closed-loop system matrix
        if Acl is None:
            Acl = self._Acl

        # check if the matrix Acl is stable, i.e., its eigenvalues are strictly in the unit circle 
        if np.max(abs(np.linalg.eigvals(Acl)))>=1:
            # if not the algorithm will not converge and we return None
            print("The matrix Acl is not stable, such that the algorithm will never converge. \n Therefore, None is returned")
            return None

        # introduce variables such that the calculation of the mRPI is not prematurely stopped
        mRPI_determined = False

        # set an initial maximum amount of iterations
        s_max = 200
        # while the function to approximate the mRPI has not successfully completed...
        while not mRPI_determined:
            # ... compute the mrpi for a given maximum number of iteration s_max...
            if rpi_method == 0:
                # ... according to the approach "Invariant Approximations of the Minimal Robust Positively Invariant Set" by Rakovic et al. 
                Fs_temp, status_mrpi = up.calculate_minimal_robust_positively_invariant_set(Acl, W = W, eps_var = eps_var, s_max = s_max)

            elif rpi_method == 1:
                # ... according to the approach in "Efficient computation of RPI sets for tube-based robust MPC" by Darup and Teichrib
                Fs_temp, status_mrpi = up.calculate_RPI(Acl, W, self._X, self._U, K, eps_var = eps_var, s_max = s_max)

            else:
                print("The method chosen to determine the RPI does not exists, so we use the default method 0")
                Fs_temp, status_mrpi = up.calculate_minimal_robust_positively_invariant_set(Acl, W = W, eps_var = eps_var, s_max = s_max)

            # if the status indicates a successful exection...
            if status_mrpi == 0:
                # ... stop while loop
                mRPI_determined = True
            else:
                # ... otherwise increase s_max
                print(f"RPI not determined in {s_max} steps. Increasing s_max to 10*s_max = {10*s_max}")
                s_max = 10*s_max
        
        # remove redundant inequalities from the approximated RPI
        Fs = pc.reduce(Fs_temp)
        # save the rpi in an internal variable
        self._Z = Fs

        return Fs

    def tighten_constraints(self):
        '''
        This function thightens the given constraints on the state and actuator input
        based on the minimum robust positively invariant set via the Pontryagin set difference
        as it is done in equation (9) and (10) of [1]
        '''
        # tighten input constraint set
        self._Uc = up.pont_diff(self._U, up.scale(self._Z,-self._K))
        # tighten state constraint set
        self._Xc = up.pont_diff(self._X,self._Z)

    def determine_Xf(self):
        '''
        This function determines the terminal set used to constrain the terminal state in the MPC optimization problem
        '''
        # set up polytope that depends only on the state by combining the state and input constraints with the control law u = -Kx
        # this guarantees that the 
        [Hu, hu] =  self._Uc.A, self._Uc.b
        [Hx, hx] =  self._Xc.A, self._Xc.b
        Gxu = np.r_[Hx,-Hu@self._K]
        fxu = np.r_[hx,hu]
        XU=pc.Polytope(Gxu,fxu)

        # calculate maximum admissible output set 
        Xf = up.calculate_maximum_admissible_output_set(self._Acl, XU)

        # save the maximum admissible output set
        self._Xf = Xf

    def generate_optimization_problem(self):
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
        Hz, hz = self._Z.A,self._Z.b # ... the mRPI
        hz.shape = (hz.shape[0], )
        
        self._x_mpc = cp.Variable((self._nx,self._N+1))
        self._u_mpc = cp.Variable((self._nu,self._N))
        self._x_init_param = cp.Parameter((self._nx,))

        constraints=[Hz@(self._x_init_param-self._x_mpc[:,0])<=hz]
        cost = 0

        for i in range(self._N):
            cost += cp.quad_form(self._x_mpc[:,i+1],self._Q)+cp.quad_form(self._u_mpc[:,i],self._R)
            # dynamic constraint x(k+1)=Ax(k)+Bu(k)
            constraints += [self._x_mpc[:,i+1]==self._A @ self._x_mpc[:,i] + self._B @ self._u_mpc[:,i]] 
            constraints += [Hx@self._x_mpc[:,i]<=hx] # state constraint x(k)\in X
            constraints += [Hu@self._u_mpc[:,i]<=hu] # input constraint u(k)\in U

        cost += cp.quad_form(self._x_mpc[:,self._N],self._P)
        constraints += [HxN@self._x_mpc[:,self._N]<=hxN]
        objective = cp.Minimize(cost)
        self._prob = cp.Problem(objective, constraints)

        hxN.shape = (hxN.shape[0], 1)

    def setup_optimization(self, W: pc.Polytope):
        # this functions sets up the MPC optimization problem by ...
        # ... determining the minimum robust positively invariant (mrpi) set
        self.determine_mRPI(W)
        # ... using the mrpi to tighten the constraints
        self.tighten_constraints()
        # ... determine the terminal set for the MPC trajectory
        self.determine_Xf()
        # ... generate the MPC optimization problem in its parametrized form
        self.generate_optimization_problem()
    
    def solve_optimization_problem(self, x_init: np.ndarray):
        '''
        This function solves the optimization problem of the MPC
        '''
        # update initial parameter
        self._x_init_param.value = x_init
        # solve the optimization problem    
        self._prob.solve(solver = self._solver)        

        # return both the predicted trajectory and control inputs
        return self._x_mpc.value, self._u_mpc.value
    
    def get_controller_gain(self):
        # returns controller gain of ancillary controller
        return self._K
    
    def get_minimum_robust_positively_invariant_set(self):
        return self._Z