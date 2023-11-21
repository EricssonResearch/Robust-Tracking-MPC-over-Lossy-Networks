'''
In this script, we implement an MPC algorithm, which from an initial condition brings the plant state to the origin.
This MPC algorithm will serve as a parent class for the other MPC 
'''
import numpy as np
import cvxpy as cp
import polytope as pc

class RegulatorMPC:

    def __init__(self,A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, N: int) -> None:
        # initialize system matrices
        self._A=A
        self._B=B
        self._N=int(N)

        # initialize variables for dimension of...
        # ... the state
        self._nx=self._A.shape[1]
        # ... and the input
        self._nu=self._B.shape[1]

        # initialize cost matrices
        self._Q=Q
        self._R=R

        # initialize constraints for the state and input
        self._X=None
        self._U=None
        # set solver of the MPC problem to Clarabel 
        self._solver = cp.CLARABEL

    def set_state_constraints(self, X: pc.Polytope):
        '''
        This function sets the state constraints used in the MPC
        '''
        self._X = X

    def set_input_constraints(self, U: pc.Polytope):
        '''
        This function sets the actuator input constraints used in the MPC
        '''
        self._U = U

    def generate_optimization_problem(self):
        '''
        This function generates a parametrized version of the optimization problem solved in the MPC.
        By doing so we can speed up the optimization, since the optimization problem does not have to be re-instantiated each time it is solved.
        '''
        # determine polyhedra constraints for...         
        if self._U is not None:        
            Hu, hu =  self._U.A, self._U.b # ... the input
            hu.shape = (hu.shape[0], )
        
        if self._X is not None:
            Hx, hx =  self._X.A, self._X.b # ... the state
            hx.shape = (hx.shape[0], )
        
        self._x_mpc = cp.Variable((self._nx,self._N+1))
        self._u_mpc = cp.Variable((self._nu,self._N))
        self._x_init_param = cp.Parameter((self._nx,))

        constraints=[self._x_mpc[:,0]==self._x_init_param]
        cost = 0
        for i in range(self._N):
            cost += cp.quad_form(self._x_mpc[:,i+1],self._Q)+cp.quad_form(self._u_mpc[:,i],self._R)
            
            # dynamic constraint x(k+1)=Ax(k)+Bu(k)
            constraints += [self._x_mpc[:,i+1]==self._A @ self._x_mpc[:,i] + self._B @ self._u_mpc[:,i]]
            if self._X is not None:
                constraints += [Hx@self._x_mpc[:,i]<=hx] # state constraint x(k) \in X
            if self._U is not None:
                constraints += [Hu@self._u_mpc[:,i]<=hu] # input constraint u(k) \in U
        # TODO: Setup Xf here as well to guarantee recursive feasibility
        objective = cp.Minimize(cost)
        self._prob = cp.Problem(objective, constraints)
    
    def solve_optimization_problem(self, x_init: np.ndarray):
        '''
        This function solves the optimization problem of the MPC given an initial state x_init
        '''
        # setting the parameter in the parameterized optimization problem to x_init
        x_init.shape = (self._nx, )

        self._x_init_param.value = x_init

        # solve the optimization problem
        self._prob.solve(solver = self._solver)

        # return both the predicted trajectory and control inputs
        return self._x_mpc.value, self._u_mpc.value
        
    def set_solver(self, solver):
        self._solver = solver