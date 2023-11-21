'''
Script that implements several functions operating on Polytope objects from the polytope package.
It includes the support function of a polytope, the Pontryagin difference and the Minkowski sum of two polytopes as well as 
the calculation of he maximum admissible output set of a linear discrete-time system and several methods to calculate the minimal robust positively invariant set.
'''

import numpy as np
import scipy as sp
import polytope as pc
from typing import Union

def support(poly: pc.Polytope, x: np.ndarray):
    '''
    Function that implements the calculation of the support of the polytope {x|Ax<=b} 
    as, for example, stated in "Theory and Computation of Disturbance Invariant Sets for Discrete-time Linear Systems" 
    '''
    A = poly.A
    b = poly.b
    results = sp.optimize.linprog(c=-x, A_ub=A, b_ub=b, bounds = (None, None))
    if results.status != 0:
        print(f"Status of the support linear program: {results.status}")
        
    return -results.fun

def pont_diff(poly1: pc.Polytope, poly2: pc.Polytope):
    '''
    Function that implements  the pontryagin set difference poly1 - poly2 = {x| x+y in poly1 for all y in poly2} 
    according to Theorem 3.2 in "Theory and Computation of Disturbance Invariant Sets for Discrete-time Linear Systems" 
    '''
    # get matric and vector describing poly1 
    A1 = poly1.A
    nrx = A1.shape[0]
    b1 = poly1.b
    b_diff = np.zeros((nrx,))
    for i in range(nrx):
        b_diff[i] = support(poly2, A1[i,:])
    poly_out = pc.Polytope(A1, b1-b_diff)
    return poly_out

def mink_sum(poly1: pc.Polytope, poly2: Union[pc.Polytope, np.ndarray]):
    '''
    Function that implements  the Minkowski sum of two convex polytopes. 
    The first polytope, poly1, needs to be specified as a polytope object.
    The second polytope, poly2, can be a polytope object, a vector, or a matrix containing the vertices of poly2
    '''
    # get V representation of poly1 if it is not already available
    if poly1.vertices is None:
        V1 = pc.extreme(poly1)
    else:
        V1 = poly1.vertices.copy()
        
    ndim = V1.shape[1]
    nV1 = V1.shape[0]

    if isinstance(poly2, pc.Polytope):
        # if the input is a polytope, get its V representation
        if poly2.vertices is None:
            V2 = pc.extreme(poly2)      
        else:
            V2 = poly2.vertices.copy()

        nV2 = V2.shape[0]
        vertices_of_mink_sum = np.zeros((nV1*nV2,ndim))
        
        # sum up all vertices
        if nV1<=nV2:
            for i in range(nV1):
                v1_vert_i = V1[i,:]
                v1_vert_i.shape = (1,ndim)
                vertices_of_mink_sum[i*nV2:(i+1)*nV2,:] = V2+v1_vert_i
        else:
            for i in range(nV2):
                v2_vert_i = V2[i,:]
                v2_vert_i.shape = (1,ndim)
                vertices_of_mink_sum[i*nV1:(i+1)*nV1,:] = V1+v2_vert_i

        poly_out = determine_convex_hull(vertices_of_mink_sum)     

    elif isinstance(poly2, np.ndarray): 
        if poly2.ndim == 1:
            # if the input is a single point/vector
            A = poly1.A
            b = poly1.b
            poly2.shape = (A.shape[1],)
            poly_out = pc.Polytope(A,b+A@poly2)

        elif poly2.ndim == 2:
            # if the input is two dimensional we will interpret it as a set of vertices of a polytope
            nV2 = poly2.shape[0]

            vertices_of_mink_sum = np.zeros((nV1*nV2,ndim))

            # sum up all vertices
            if nV1<=nV2:
                for i in range(nV1):
                    v1_vert_i = V1[i,:]
                    v1_vert_i.shape = (1,ndim)
                    vertices_of_mink_sum[i*nV2:(i+1)*nV2,:] = poly2+v1_vert_i
            else:
                for i in range(nV2):
                    v2_vert_i = poly2[i,:]
                    v2_vert_i.shape = (1,ndim)
                    vertices_of_mink_sum[i*nV1:(i+1)*nV1,:] = V1+v2_vert_i
            
            poly_out = determine_convex_hull(vertices_of_mink_sum)
            
        else: 
            print("If the input is a numpy array it should have dimension 1 or 2")
            
    else:
        print("Input has the wrong type")
        
    return poly_out

def scale(poly: pc.Polytope, scaling_variable):
    '''
    Function that implements the scaling of a polytope with the scaling variable, which can be a matrix or scalar
    '''
    ndim_in = np.squeeze(scaling_variable).ndim
    
    # check if scaling variable is a scalar
    if np.isscalar(scaling_variable) or ndim_in==0:
        if scaling_variable == 1:
            # if scaling variable is 1 return the polytope poly
            poly_out = poly.copy()
        elif scaling_variable == 0:
            # if scaling variable is 0, return a singleton polytope, where the single value is zero
            n = poly.A.shape[1]
            Aout = np.r_[np.eye(n),-np.eye(n)]
            bout = np.zeros((2*n,1))
            poly_out = pc.Polytope(Aout,bout)
        elif scaling_variable>0:
            A = poly.A
            b = poly.b
            poly_out = pc.Polytope(A,scaling_variable*b)
        else:
            A = poly.A
            b = poly.b
            poly_out = pc.Polytope(1/scaling_variable*A,b)
    elif isinstance(scaling_variable, np.ndarray):
        # if the scaling variable is a matrix

        # get vertex representation
        if poly.vertices is None:
            V1 = pc.extreme(poly)
        else:
            V1 = poly.vertices
        # multiply all vertices with scaling variable
        # NOTE: pc.extreme returns an Nxd array, where d is the dimension of the polytope. The rows of this array are the vertices of the polytope.
        # Hence, we need to transpose the the transposed scaling variable
        vertices_out = V1@scaling_variable.T
       
        poly_out = determine_convex_hull(vertices_out)
        
    else:
        print("The input is neither an np.ndarray nor a scalar. Therefore, we return None")
        poly_out = None
    return poly_out

def determine_convex_hull(vertices: np.ndarray):
    '''
    Function to determine the convex hull of a polytope based on its vertices.
    We assume that the vertices are stored in the rows of the input variable
    '''
    if vertices.shape[1]==1:
        # for one-dimensional vertices, we use the qhull function of the polytope package
        poly_out = pc.qhull(vertices)
    else:
        # for multi-dimensional vertices, we use a function from scipy to speed up the calculations
        hull = sp.spatial.ConvexHull(vertices)

        Abneg = hull.equations
        A = Abneg[:,0:-1]
        b = -Abneg[:,-1]
        vertices_out = vertices[hull.vertices,:]
        poly_out = pc.Polytope(A=A, b=b, vertices = vertices_out)
    
    return poly_out

def calculate_minimal_robust_positively_invariant_set(A, W: pc.Polytope, eps_var: float = 1.9*10**(-5), s_max: int = 20):
    '''
    Function to calculate the minimal robust positively invariant set of a linear system x(k+1)=Ax(k)+w,
    where w \in W and W is a polytope that contains the origin.
    The function implements Algorithm 1 of "Invariant Approximations of the Minimal Robust Positively Invariant Set" by Rakovic et al.
    and the function is inspired by the eps_mRPI() function of the pytope package.
    '''
    if (A.shape[0] != A.shape[1]):
        print("A needs to be a square matrix. Returning None")
        return None
    if (sum(W.b<=0) != 0):
        print("The polytope W does not contain the origin. Therefore, we return None")
        return None
    if not isinstance(W, pc.Polytope):
        print("W needs to be of type Polytope or ExtendedPolytope. Returning None")
        return None
    
    F = W.A.copy()
    g = W.b.copy()
    nw = F.shape[0]
    g.shape = (nw,1)
    nx = A.shape[0]
    # initialize variables to determine mRPI
    alpha_s = np.zeros((nw,1))
    M_n_pos = np.zeros((nx,1))
    M_n_neg = np.zeros((nx,1))

    # pre-calculate matrix powers
    A_pwr = np.stack([np.linalg.matrix_power(A, i) for i in range(s_max)])
    status = -1
    
    s = 0
    
    while s < s_max-1:
        s+=1
        for i in range(nw):
            fi = F[i,:]
            fi.shape = (nx,1)
            alpha_s[i,0] = support(W, A_pwr[s].T @ fi)/g[i,0]
        
        alpha_s_max = np.max(alpha_s)
        A_sm1 =  A_pwr[s-1]

        for j in range(nx):            
            M_n_pos[j,0] += support(W,A_sm1[j,:])
            M_n_neg[j,0] += support(W,-A_sm1[j,:]) 

        M_s = np.max([np.max(M_n_pos),np.max(M_n_neg)])

        if alpha_s_max <= eps_var/(eps_var+M_s):
            status = 0
            break

    if status != 0:
        print("In the RPI calculation, we reached the iteration maximum {s_max} without converging!")
        return None, status

    # compute set Fs, which will be used to approximate the mRPI
    for i in range(s):
        if i == 0:
            Fs = pc.Polytope(W.A,W.b)
            vertices_W = pc.extreme(W)
        else:
            Fs = mink_sum(Fs,vertices_W@( A_pwr[i].T))
            
    return scale(Fs,1/(1-alpha_s_max)), status

def calculate_maximum_admissible_output_set(A :np.ndarray, X: pc.Polytope):
    """  
    This function determines the maximum admissible output set of the discrete time system 
            x(k+1)=Ax(k), where x(k)\in X for all k>=0
    according to Algorithm 3.1 of "Linear systems with state and control constraints: the theory and application of maximal output admissible sets" by Gilbert and Tan
    """
    G_min = X.A
    f_min = X.b
    t = 0
    O = [X]
    while True:
        Ot = O[t]
        Onext = Ot.intersect(pc.Polytope(G_min@np.linalg.matrix_power(A,t+1),f_min))

        if Ot == Onext:
            print(f"Admissible set calculation has converged at t = {t}")
            break
        else:
            O.append(Onext)
        t+=1

    return O[-1]

def calculate_RPI(A: np.ndarray, W: pc.polytope, X: pc.polytope, U: pc.polytope, K: np.ndarray, eps_var: float = 1e-4, s_max: int = 20, return_container: bool = False):
    '''
    Function that appoximates the minimum robust positively invariant set according to 
    the approach in "Efficient computation of RPI sets for tube-based robust MPC" by Darup and Teichrib
    '''
    if (A.shape[0] != A.shape[1]):
        print("A needs to be a square matrix. Returning None")
        return None
    if (sum(W.b<=0) != 0):
        print("The polytope W does not contain the origin. Therefore, we return None")
        return None
    if not isinstance(W, pc.Polytope):
        print("W needs to be of type Polytope or ExtendedPolytope. Returning None")
        return None
    
    # set status variable to -1, which indicates it is not solved
    status = -1
    # get dimension of the state
    nx = A.shape[0]

    # get matrix and vector describing the disturbance polytope
    Hw = W.A.copy()
    hw = W.b.copy()
    nw = Hw.shape[0]
    hw.shape = (nw,)

    # get matrix and vector describing the state polytope
    Hx = X.A.copy()
    hx = X.b.copy()

    # get matrix and vector describing the input polytope
    Hu = U.A.copy()
    hu = U.b.copy()

    # matrices describing polytope D above equation (12)
    Hd = np.r_[Hx,-Hu@K]
    nd = Hd.shape[0]
    hd = np.r_[hx,hu]
    hd.shape = ((nd,))
    
    A_pwr = np.stack([np.linalg.matrix_power(A, i) for i in range(s_max)])

    # Below we implement the proposed algorithm to find k_star that fulfills the conditions (9)

    # initialize variables to determine if k_star is found  
    k_star_found = False
    k_star = 1
    # initialize variable to save support values.
    hw_k = np.zeros((nw,))
    # initialize variable, which helps us to determine the container C described in Theorem 1 later if a suitable k_star has been found

    bc_all = np.zeros((nd,s_max))
    while k_star<s_max and not k_star_found:
        HcAj = Hd@A_pwr[k_star-1]

        HwAj = Hw@A_pwr[k_star]
        
        # check if the condition of equation (10), which indicates condition (9a) is fulfilled
        w_count = 0
        for i_w in range(nw):
            g = HwAj[i_w,:]
            g.shape = (nx,1)
            hw_k[i_w] = support(W,g)
            if (1+eps_var)*hw_k[i_w]<=eps_var*hw[i_w]:
                w_count += 1
            else:
                break
        
        if w_count == nw:
            condition_a_fulfilled = True
        else:
            condition_a_fulfilled = False
        # check if the condition of equation (12), which indicates condition (9b) is fulfilled
        for l in range(nd):
            g = HcAj[l,:]
            g.shape = (nx,1)
            if k_star==1:
                bc_all[l,k_star-1] = support(W,g)
            else:
                bc_all[l,k_star-1] += bc_all[l,k_star-2] + support(W,g)

        if np.sum((1+eps_var)*bc_all[:,k_star-1]<=hd)==nd:
            condition_b_fulfilled = True
        else: 
            condition_b_fulfilled = False
        
        # if both conditions are fulfilled, we have found k_star
        if condition_a_fulfilled and condition_b_fulfilled:
            k_star_found = True
            status = 0
        else:
            k_star +=1
    # print out k_star
    print(f"k_star = {k_star}")

    # if k_star was not found and we reached the maximum number of iteration, we return None
    if not k_star_found:
        print(f"In the RPI calculation, we reached the iteration maximum {s_max} without converging!")
        return None, status
    
    # once we have found k_star for a given eps_var, we determine container C according to Theorem 1
    C = pc.Polytope(Hd,(1+eps_var)*bc_all[:,k_star-1])

    Hc = Hd.copy()
    hc = (1+eps_var)*bc_all[:,k_star-1]
    nc = Hc.shape[0]

    # once the container is determined we proceed to calculate the RPI according to Theorem 2

    # first we check if condition (27) holds
    HcAkstar = Hc @ A_pwr[k_star]
    c_count = 0
    hc_support = np.zeros((nc,))

    for i_c in range(nc):
        g = HcAkstar[i_c,:]
        g.shape = (nx,1)
        hc_support[i_c] = support(C,g)

        if (1+eps_var)*hc_support[i_c]<=eps_var*hc[i_c]:
            c_count += 1
        else:
            break
    
    if c_count != nc:
        print("The container set C does not fulfill the condition for calculating the RPI. Returning None")
        return None, -1
    
    # if the condition (27) holds, we determine the rpi according to equation (28)
    H_Pinfty = []
    h_Pinfty = []

    for i_p in range(k_star):
        if i_p == 0:
            H_Pinfty=Hc
            h_Pinfty=hc
        else:
            H_Pinfty = np.r_[H_Pinfty, Hc@A_pwr[i_p]]
            h_Pinfty = np.r_[h_Pinfty,hc-bc_all[:,i_p-1]]

    rpi = pc.Polytope(H_Pinfty,h_Pinfty)
    if return_container:
        return rpi, C, status
    else:
        return rpi, status