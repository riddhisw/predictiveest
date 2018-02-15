# Reference 
# [1] Laub, J.A. et. all. The Riccatti Equation. (1991). 

import numpy as np
from sympy.matrices import Matrix 


def t_matrix_dare(A, B, R, Q):

    '''
    A discrete time Riccati Equation of the form:
    
    X = A*XA - A*XB(R + B*XB)^-1 B*XA + Q; * == conjugate transpose
     
    is related to the eigen value problem for the matrix T.
    
    The purpose of this function is to create the T matrix associated with DARE
    using known square numpy arrays, A, R, Q.
    
    A - dimensions n x n
    Q -  
    X - dimensions m x n
    B - 
    R - 
    
    Define two intermediary matrices as:
    
    S := A*^-1 Q  - dimensions m x n
    W := BR^-1B*  - dimensions n x m 
    
    Then, the T matrix to solve the DARE is given:
    
    T[ 0 : n, 0 : n] = A + WS dimensions n x n
    T[ 0 : n , n : ] = -W dimensions n x m
    T[ n : , 0 : n] = -S dimensions m x n
    T[ n :, n : ] = A*^-1  dimensions m x m 
    
    
    '''

    
    dim_n = A.shape[0]
    dim_m = A.shape[1]
    
    invAct = np.linalg.inv(A).conj().T
    
    # Allow for scalar R:
    if R.shape[0] != 1:
        inv_R = np.linalg.inv(R)
        W = np.dot(B, np.dot(inv_R, B.conj().T))
        
    else:
        inv_R = 1.0/R
        W = np.outer(B, B.conj().T)*inv_R
     
    S = np.dot(invAct, Q)
    
    print('A*-1', invAct)
    
    T_matrix = np.zeros([int(dim_n+ dim_m), int(dim_n + dim_m)])
    
    T_matrix[0:dim_n, 0:dim_n] = A + np.dot(W, S)
    T_matrix[0 :dim_n, dim_n:] = -np.dot(W, invAct)
    T_matrix[dim_n:, 0:dim_n ] =  -S
    T_matrix[dim_n: , dim_n: ] =  invAct
    
    print('T', T_matrix)
    print('C prime / -S', -S)
    print('W', W)
    print('B prime', -np.dot(W, invAct))
    print('WS', np.dot(W, S))
    
    
    return T_matrix
    
    
def jordan_transform_matrix(matrix):
    
    A_sym = Matrix(matrix)
    P, J = A_sym.jordan_form()
    # (map_, cells_) = A_sym.jordan_cells()
    
    # convert to numpy arrays
    map_np = np.array(P).astype(np.complex128)  # np.array(map_).astype(np.complex128) 
    #cells_np = [np.array(cells_[x]).astype(np.complex128) for x in xrange(len(cells_))]
    
    return map_np, np.array(J).astype(np.complex128)  # cells_np
    
    
def riccatti_solver(A, B, R, Q):
    
    dim_n = A.shape[0]
    dim_m = A.shape[1]
    
    T_matrix = t_matrix_dare(A, B, R, Q)
    general_map, jordan_cells = jordan_transform_matrix(T_matrix)
    
    print general_map.shape
    
    Y_matrix = general_map[0: dim_n, 0: dim_n]
    Z_matrix = general_map[dim_n:, 0: dim_n] # not sure how to construct Z
    X_soln = np.dot(Z_matrix, np.linalg.inv(Y_matrix))
    
    return X_soln, Y_matrix, Z_matrix, general_map
    
    
def check_dare_solution(Xsoln, A, B, R, Q):
    
    inverse_term = np.linalg.inv(R + np.dot(np.dot(B.conj().T, Xsoln), B))
    mirror_term = np.dot(np.dot(B.conj().T, Xsoln), A)
    
    zero = np.dot(np.dot(A.conj().T, Xsoln), A) + Q
    zero += -1*np.dot(np.dot(mirror_term.conj().T, inverse_term), mirror_term)
    
    zero += -Xsoln

    return zero
    
    
