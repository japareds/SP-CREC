#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 17:21:04 2023

@author: jparedes
"""
import numpy as np
import cvxpy as cp
from cvxopt import matrix, solvers, spmatrix, sparse, spdiag
from scipy import linalg
import pickle
import time

import sys


"""
=============================================================================
Sensor placement algorithms
=============================================================================
"""
class SensorsErrors(Exception):
    pass

class SensorPlacement:
    def __init__(self,algorithm:str='rankMax',n:int=100,s:int=10,n_refst:int=10,n_lcs:int=0,n_unmonitored:int=0,var_refst:float=1e-2,var_lcs:int=1):

        """
        Network and sensor placement parameters

        Args:
            algorithm (str): sensor placement algorithm to use
            n (int): number of potential locations
            s (int): signal sparsity
            n_refst (int): number of reference stations
            n_lcs (int): number of low-cost sensors (LCSs)
            n_unmonitored (int): number of unmonitored locations
            var_refst (float): reference stations devices noise variance
            var_lcs (vloat): LCSs devices noise variance
        """
        self.algorithm = algorithm
        self.n = n
        self.s = s
        self.n_refst = n_refst
        self.n_lcs = n_lcs
        self.n_unmonitored = n_unmonitored
        self.var_refst = var_refst
        self.var_lcs = var_lcs

    """ Sensor placement algorithms """

    def JB_placement(self,Psi,locations_monitored:list,locations_unmonitored:list):
            """
            Simple D-optimal sensor placement for single class of sensors

            Parameters
            ----------
            Psi : np array
                Reduced basis of shape (number of locations, number of vectors)
            locations_monitored: list
                indices of locations that have a sensor
            locations_unmonitored: list
                indices of forbidden locations

            Returns
            -------
            None.

            """
            h = cp.Variable(shape=self.n,value=np.zeros(self.n))
            objective = -1*cp.log_det(cp.sum([h[i]*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(self.n)]))
            constraints = [ cp.sum(h) == self.n_refst + self.n_lcs,
                        h >=0,
                        h <= 1,
                        ]
            if len(locations_monitored) != 0:
                print(f'Adding constraint on {len(locations_monitored)} monitored locations')
                constraints += [h[locations_monitored] == 1]
            if len(locations_unmonitored) !=0:
                print(f'Adding constraint on {len(locations_unmonitored)} unmonitored locations')
                constraints += [h[locations_unmonitored]==0]
            problem = cp.Problem(cp.Minimize(objective),constraints)


            if not problem.is_dcp():
                print('Problem not dcp\nCheck constraints and objective function')
            self.h = h
            self.problem = problem
    
    def JB_Trace_placement(self,Psi,locations_monitored:list,locations_unmonitored:list):
            """
            Simple A-optimal sensor placement for single class of sensors
            Minimizes the trace of covariance matrix cov_beta
            but tr(cov_x) == tr(cov_beta)

            Implemented usng cvxpy library

            min     tr(Psi.T@H@Psi)^-1)
            s.t.    sum(h) == p
                    0<= h<= 1

            Parameters
            ----------
            Psi : np array
                Reduced basis of shape (number of locations, number of vectors)
            locations_monitored: list
                indices of locations that have a sensor
            locations_unmonitored: lsit
                indices of forbidden locations

            Returns
            -------
            None.

            """
            h = cp.Variable(shape=self.n,value=np.zeros(self.n))
            objective = cp.tr_inv(cp.sum([h[i]*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(self.n)]))
            constraints = [ cp.sum(h) == self.n_refst + self.n_lcs,
                           h >=0,
                           h <= 1
                           ]
            if len(locations_monitored) != 0:
                print(f'Adding constraint on {len(locations_monitored)} monitored locations')
                constraints += [h[locations_monitored] == 1]
            if len(locations_unmonitored) !=0:
                print(f'Adding constraint on {len(locations_unmonitored)} unmonitored locations')
                constraints += [h[locations_unmonitored]==0]
            problem = cp.Problem(cp.Minimize(objective),constraints)


            if not problem.is_dcp():
                print('Problem not dcp\nCheck constraints and objective function')
            self.h = h
            self.problem = problem

    def JB_worst_variance_placement(self,Psi,locations_monitored:list,locations_unmonitored:list,threshold:float=0.):
            h = cp.Variable(shape=self.n,value=np.zeros(self.n))
            t = cp.Variable(shape=1,value=100+np.zeros(1))
            objective = cp.power(t,1)
            constraints = [cp.sum(h)==self.n_refst+self.n_lcs,
                           h>=0,
                           h<=1]

            for i in range(self.n):
                #e_i = In[:,i]
                e_i = np.zeros(self.n,dtype=np.float32)
                e_i[i] = 1
                r1 = cp.hstack([Psi.T@cp.diag(h)@Psi,Psi.T@e_i[:,None]])
                r2 = cp.hstack([e_i[:,None].T@Psi,t[:,None]])
                #r2 = cp.hstack([e_i[:,None].T@Psi,np.array([t])[:,None]])
                M_i = cp.vstack([r1,r2])
                constraints += [M_i >> 0]

            if threshold != 0:
                constraints += [t<=threshold]
            if len(locations_monitored) != 0:
                print(f'Adding constraint on {len(locations_monitored)} monitored locations')
                constraints += [h[locations_monitored] == 1]
            if len(locations_unmonitored) !=0:
                print(f'Adding constraint on {len(locations_unmonitored)} unmonitored locations')
                constraints += [h[locations_unmonitored]==0]

            problem = cp.Problem(cp.Minimize(objective),constraints)
            if not problem.is_dcp():
                print('Problem not dcp\nCheck constraints and objective function')
            self.h = h
            self.problem = problem


    def rankMax_placement(self,Psi,alpha,substract=False):
        """
        Sensor placement proposed heuristics for unmonitored locations (p_empty !=0)
        Maximize the rank for reference stations locations while minimizing volume of ellipsoid for LCSs

        Parameters
        ----------
        Psi : np array
            Reduced basis of shape (number of locations, number of vectors)
        alpha : float
            regularization parameter for rank minimization constraint

        Returns
        -------
        None.

        """

        h_lcs = cp.Variable(shape=Psi.shape[0],value=np.zeros(Psi.shape[0]))
        h_refst = cp.Variable(shape = Psi.shape[0],value=np.zeros(Psi.shape[0]))
        objective_eps = -1*cp.log_det(cp.sum([h_lcs[i]*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(Psi.shape[0])]))
        if substract:
            print('Objective function for trace will be negative: logdet(LCS) - alpha*Tr(RefSt)')
            objective_zero = -alpha*cp.trace(cp.sum([h_refst[i]*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(Psi.shape[0])]))
        else:
            objective_zero = alpha*cp.trace(cp.sum([h_refst[i]*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(Psi.shape[0])]))

        objective = objective_eps + objective_zero
        constraints = [cp.sum(h_refst) == self.n_refst,
                       cp.sum(h_lcs) == self.n_lcs,
                       h_refst >=0,
                       h_refst <= 1,
                       h_lcs >=0,
                       h_lcs <=1,
                       h_lcs + h_refst <=1]
        problem = cp.Problem(cp.Minimize(objective),constraints)
        
        if not problem.is_dcp():
            print('Problem not dcp\nCheck constraints and objective function')
        self.h_lcs = h_lcs
        self.h_refst = h_refst
        self.problem = problem

   

    def multiClass_joshiBoyd_placement(self,Psi):
        """
        D-optimal sensor placement for two classes of sensors with variances var_zero and var_eps
        Maximize logdet of precision matrix

        Parameters
        ----------
        Psi : np array
            Reduced basis of shape (number of locations, number of vectors)

        Returns
        -------
        None.

        """
        h_refst = cp.Variable(shape = self.n,value=np.zeros(self.n))
        h_lcs = cp.Variable(shape=self.n,value=np.zeros(self.n))

        if self.n_lcs != 0 and self.n_refst != 0: # there are LCSs and ref st.
            objective = -1*cp.log_det(cp.sum([h_lcs[i]*(self.var_lcs**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] + h_refst[i]*(self.var_refst**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(self.n)]))
            constraints = [cp.sum(h_refst) == self.n_refst,
                           cp.sum(h_lcs) == self.n_lcs,
                           h_refst >=0,
                           h_refst <= 1,
                           h_lcs >=0,
                           h_lcs <=1,
                           h_lcs + h_refst <=1]


        elif self.n_lcs == 0 and self.n_refst != 0:# there are no LCSs
            objective = -1*cp.log_det(cp.sum([h_refst[i]*(self.var_refst**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(self.n)]))
            constraints = [cp.sum(h_refst) == self.n_refst,
                           h_refst >=0,
                           h_refst <= 1,
                           ]


        elif self.n_lcs != 0 and self.n_refst == 0:# there are no ref.st.
            objective = -1*cp.log_det(cp.sum([h_lcs[i]*(self.var_lcs**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(self.n)]))
            constraints = [cp.sum(h_lcs) == self.n_lcs,
                           h_lcs >=0,
                           h_lcs <= 1,
                           ]


        problem = cp.Problem(cp.Minimize(objective),constraints)
        if not problem.is_dcp():
            print('Problem not dcp\nCheck constraints and objective function')
        self.h_lcs = h_lcs
        self.h_refst = h_refst
        self.problem = problem

    def JB_feasibility_program(self,Psi:np.ndarray,rho:list,locations_monitored:list=[],locations_unmonitored:list=[]):
        """
        Sensor placement algorithm that constraints the per coordinate variance-covariance (diagonal)
        with a fixed number of sensors. 
        For a low number of sensors the problem would be unfeasible. The problem should be iterated until the problem is feasible and a solution is found.
        Alhorithm is similar to network design proposed algorithm.
        Implemented using cvxpy

        min     1
        s.t.    (cov_x)_ii<=rho ===> M_i>=0 (PSD)
                sum(h) == p
                0<=h_i<=1
                

        Args:
            Psi (_type_): _description_
        """
        t = cp.Variable(shape=1,value=1+np.zeros(1))
        h = cp.Variable(shape=self.n,value=np.zeros(self.n))
        objective = 1#cp.power(t,1)

        constraints = [ h >=0,
                        h <= 1,
                        cp.sum(h) == self.n_refst + self.n_lcs
                        ]
        
        # PSD constraints
        if len(rho)!=self.n:
            raise ValueError(f'Coordinate constraints is not a list of length equivalent to network size.')
        for i in range(self.n):
            e_i = np.zeros(self.n,dtype=np.float32)
            e_i[i] = 1
            r1 = cp.hstack([Psi.T@cp.diag(h)@Psi,Psi.T@e_i[:,None]])
            r2 = cp.hstack([e_i[:,None].T@Psi,np.array([rho[i]])[:,None]])
            M_i = cp.vstack([r1,r2])
            constraints += [M_i >> 0]

        if len(locations_monitored) !=0:
            constraints += [h[locations_monitored]==1]
        if len(locations_unmonitored) != 0:
            constraints += [h[locations_unmonitored]==0]
        
        problem = cp.Problem(cp.Minimize(objective),constraints)
        if not problem.is_dcp():
            print('Problem not dcp\nCheck constraints and objective function')
        self.h = h
        self.problem = problem

    def JB_feasibility_diagonal(self,Psi:np.ndarray,rho:list,locations_monitored:list=[],locations_unmonitored:list=[]):
        """
        Sensor placement algorithm that minimizes the diagonal elements of the 
        per-coordinate covariance matrix.
        Implemented using cvxpy library
        min     sum(t_i)_i=1^n
        s.t.    t_i >= (Cov_x)_ii
                rho >= t_i
                t_i >= 0
                sum(h) == p
                0<=h<=1

        Constraint t_i <= (Cov_x)_ii is equivalent to PSD constraint on matrix M_i

        Args:
            Psi (np.ndarray): _description_
            rho (list): _description_
            locations_monitored (list, optional): _description_. Defaults to [].
            locations_unmonitored (lsit, optional): _description_. Defaults to [].
        """
        t = cp.Variable(shape=self.n,value=np.ones(self.n))
        h = cp.Variable(shape=self.n,value=np.zeros(self.n))
        objective = cp.sum(t)

        constraints = [ h >=0,
                        h <= 1,
                        cp.sum(h) == self.n_refst + self.n_lcs,
                        t >= 0
                        ]
        
        if len(rho)!=self.n:
            raise ValueError(f'Coordinate constraints is not a list of length equivalent to network size.')
        # threshold constraints
        constraints += [rho>=t]
        # PSD constraints
        for i in range(self.n):
            e_i = np.zeros(self.n,dtype=np.float32)
            e_i[i] = 1
            r1 = cp.hstack([Psi.T@cp.diag(h)@Psi,Psi.T@e_i[:,None]])
            r2 = cp.hstack([e_i[:,None].T@Psi,cp.reshape(t[i],(1,1))])
            M_i = cp.vstack([r1,r2])
            constraints += [M_i >> 0]
        
        # constraints forcing certain indices to be (un)monitored
        if len(locations_monitored) !=0:
            constraints += [h[locations_monitored]==1]
        if len(locations_unmonitored) != 0:
            constraints += [h[locations_unmonitored]==0]
        
        problem = cp.Problem(cp.Minimize(objective),constraints)
        if not problem.is_dcp():
            print('Problem not dcp\nCheck constraints and objective function')
        self.h = h
        self.problem = problem


    def networkPlanning_singleclass(self,Psi:np.ndarray,rho:float):
        """
        Implemented using cvxpy

        min |h|_1
        s.t. -  0<=h<=1
             -  sum(h) >=s
             -  M_i >= 0(PSD), i=1,...,n

        Args:
            Psi (np.ndarray): Low-rank basis
            rho (float): accuracy threshold
        """
        h = cp.Variable(shape = self.n,value=np.zeros(self.n))
        objective = cp.norm(h,1)
        constraints = [h >=0,
                       h <= 1,
                       cp.sum(h) >= self.s,
                       ]
        In = np.identity(self.n)
        for i in range(self.n):
            e_i = In[:,i]
            r1 = cp.hstack([Psi.T@cp.diag(h)@Psi,Psi.T@e_i[:,None]])
            r2 = cp.hstack([e_i[:,None].T@Psi,np.array([rho])[:,None]])
            M_i = cp.vstack([r1,r2])
            constraints += [M_i >> 0]

        problem = cp.Problem(cp.Minimize(objective),constraints)


        if not problem.is_dcp():
            print('Problem not dcp\nCheck constraints and objective function')
        self.h = h
        self.problem = problem

    def networkPlanning_singleclass_iterative(self,Psi:np.ndarray,rho:list,w:np.array,p_min:int=0,locations_monitored:list=[],locations_unmonitored:list=[]):
        """
        Network design algorithm using Candes IRL1 objective function and Rusu method to ensure binary solution.
        The lists must be updated iteratively when calling this method from the solution h.
        Implemented using cvxpy library
        
        min     w.T @ h
        s.t.    - 0<=h<=1
                - h[S] == 1
                - h[Sc] == 0
                - sum(h)>=s
                - M_i >>= (PSD),i=1...n 

        Args:
            Psi (np.ndarray): Low-rank basis
            rho (list):accuracy threshold
            w (np.array): weights vector
            locations_monitored (list, optional): indices of monitored locations from previous steps. Defaults to [].
            locations_unmonitored (list, optional): indices of unmonitored locations from previous steps. Defaults to [].
        """
        
        #In = np.identity(self.n,dtype=np.float32)
        if len(locations_monitored)!=0:
            h_init = np.zeros(self.n)
            h_init[locations_monitored] = 1
            h = cp.Variable(shape = self.n,value=h_init) 
        else:
            h = cp.Variable(shape = self.n,value=np.zeros(self.n))
        objective = w.T@h
        constraints = [h >=0,
                    h <= 1,
                    cp.sum(h) >= self.s
                    ]
        if len(locations_monitored) !=0:
            constraints += [h[locations_monitored]==1]
        if len(locations_unmonitored) != 0:
            constraints += [h[locations_unmonitored]==0]
        if p_min !=0:
            constraints+=[cp.sum(h)>=p_min]

        for i in range(self.n):
            #e_i = In[:,i]
            e_i = np.zeros(self.n,dtype=np.float32)
            e_i[i] = 1
            r1 = cp.hstack([Psi.T@cp.diag(h)@Psi,Psi.T@e_i[:,None]])
            if type(rho) in [float,np.float32,np.float64]:
                r2 = cp.hstack([e_i[:,None].T@Psi,np.array([rho])[:,None]])
            elif len(rho) == 1:
                r2 = cp.hstack([e_i[:,None].T@Psi,np.array([rho[0]])[:,None]])
            else:
                r2 = cp.hstack([e_i[:,None].T@Psi,np.array([rho[i]])[:,None]])
            M_i = cp.vstack([r1,r2])
            constraints += [M_i >> 0]
        problem = cp.Problem(cp.Minimize(objective),constraints)

        if not problem.is_dcp():
            print('Problem not dcp\nCheck constraints and objective function')
        self.h = h
        self.problem = problem
        
    def networkPlanning_singleclass_iterative_LMI(self,Psi:np.ndarray,rho:float,w:np.array,locations_monitored:list=[],locations_unmonitored:list=[]):
        """
        Network design algorithm using Candes IRL1 objective function and Rusu method to ensure binary solution.
        The lists must be updated iteratively when calling this method from the solution h.
        Implemented using cvxpy library
        
        Alternate method.
        Differs from non-LMI version in that the M_i >> 0, i=1,...,n is replaced with the LMI from Rn -> cone Sn+
        
        Args:
            Psi (np.ndarray): Low-rank basis
            rho (float):accuracy threshold
            w (np.array): weights vector
        """
        self.rho = cp.Parameter(value=rho,name='variance_threshold')
        self.Psi = cp.Parameter(shape=Psi.shape,value=Psi,name='Basis')
        self.w = cp.Parameter(shape=w.shape,value=w,name='weights')
        self.h = cp.Variable(shape = self.n,value=np.zeros(self.n))
        
        objective = self.w.T@self.h
        # value constraints
        constraints = [self.h >=0,
                    self.h <= 1,
                    cp.sum(self.h) >= self.s
                    ]
        
        if len(locations_monitored) !=0:
            constraints += [self.h[locations_monitored]==1]
        if len(locations_unmonitored) != 0:
            constraints += [self.h[locations_unmonitored]==0]

        
        # LMI constraints
        sum = cp.sum([self.h[i]*self.rho*self.Psi[i,:][None,:].T@self.Psi[i,:][None,:] for i in range(self.n)])
        for i in range(self.n):
            M_i = sum - self.Psi[i,:][None,:].T@self.Psi[i,:][None,:]
            constraints += [M_i >> 0]

        self.problem = cp.Problem(cp.Minimize(objective),constraints)

        if not self.problem.is_dcp():
            raise TypeError('Problem not dcp\nCheck constraints and objective function')
    
    def IRL1_networkDesign(self,Psi:np.ndarray,rho:float,w:np.array,locations_monitored:list=[],locations_unmonitored:list=[],epsilon:float=1e-2,primal_start={'x':[],'sl':[],'ss':[]}):
        """
        Network design algorithm using Candes IRL1 objective function and Rusu method to ensure binary solution.
        The lists must be updated iteratively when calling this method from the solution h.
        Implemented using cvxopt library

        Equality constraints are unsupported by cvxopt solver. Using ineq constraint of epsilon tolerance
        min     w.T @ h
        s.t.    - 0<=h<=1
                - sum(h)>=s
                - h[S] == 1
                - h[Sc] == 0
                - M_i >>= (PSD),i=1...n 

        Args:
            Psi (np.ndarray): Low-rank basis
            rho (float):accuracy threshold
            w (np.array): weights vector
            locations_monitored (list, optional): indices of monitored locations from previous steps. Defaults to [].
            locations_unmonitored (list, optional): indices of unmonitored locations from previous steps. Defaults to [].
        """
        
        c = matrix(w)
        # equality constraint
        In = spmatrix(1.,range(self.n),range(self.n))
        C_monitored = In[locations_monitored,:]
        C_unmonitored = In[locations_unmonitored,:]
        #C_forbidden = In[locations_forbidden,:]
        #matrix_eq = sparse([C_monitored,C_unmonitored])
        #vector_eq = matrix([matrix(np.tile(1,len(locations_monitored))),matrix(np.tile(0,len(locations_unmonitored)))],tc='d')
        """ inequality constraints 
            order of constraints: h>=0 | h<=1 | sum(h)>= s | h[S]>= 1-eps | h[Sc] <= eps | h[Sf] <=eps
        """
        matrix_ineq = sparse([spdiag(matrix(-1,(self.n,1))),
                              spdiag(matrix(1,(self.n,1))),
                              matrix(-1,(1,self.n)),
                              sparse(-1*C_monitored),
                              sparse(C_unmonitored)])
                              #sparse(C_forbidden)])

        vector_ineq = matrix([matrix(np.tile(0,self.n)),
                              matrix(np.tile(1,self.n)),
                              -self.s.astype(np.double),
                              matrix(np.tile(-(1-epsilon),len(locations_monitored))),
                              matrix(np.tile(epsilon,len(locations_unmonitored)))],tc='d')
                              #matrix(np.tile(epsilon,len(locations_forbidden))) ],tc='d')

        """ SDP constraints
            order of constraints: -rho_i*sum_j (Psi_j.T*Psi_j)*h_j <= -Psi_i.T*Psi_i       """
        # LMI constraint
        # constant rho
        if type(rho) in [float,np.float32,np.float64]:
            matrix_sdp = [sparse([np.reshape(np.tril(-rho*Psi[i,:][None,:].T@Psi[i,:][None,:]),newshape=self.s*self.s,order='F').tolist() for i in range(self.n)])]*self.n
            #matrix_sdp = [sparse([np.reshape(np.tril(-rho*Psi[i,:][None,:].T@Psi[i,:][None,:]),newshape=s*s,order='F').tolist() for i in range(n)])]*n
        elif len(rho) == 1:
            matrix_sdp = [sparse([np.reshape(np.tril(-rho[0]*Psi[i,:][None,:].T@Psi[i,:][None,:]),newshape=self.s*self.s,order='F').tolist() for i in range(self.n)])]*self.n
        # different rho at different locations
        elif len(rho) != self.n:
            raise ValueError(f'Design threshold rho is a list of {len(rho)} elements but there are {self.n} potential locations.')
        else:
            matrix_sdp = [sparse([np.reshape(np.tril(-rho[i]*Psi[j,:][None,:].T@Psi[j,:][None,:]),newshape=self.s*self.s,order='F').tolist() for j in range(self.n)]) for i in range(self.n)]
        print(f'Basis type: {type(Psi)}. Basis dtype: {Psi.dtype}')
        vector_sdp = [matrix(np.tril(-1*Psi[i,:][None,:].T@Psi[i,:][None,:]).astype(float)) for i in range(self.n)]
        
        
        # solver and solution
        print('Calling SDP solver')
        try:
            solvers.options['show_progress'] = False
            self.problem = solvers.sdp(c,Gl=matrix_ineq,hl=vector_ineq,Gs=matrix_sdp,hs=vector_sdp,verbose=False,solver='dsdp',primalstart=primal_start)
            print('dsdp solver found')
        except:    
            print('Solving using non-specialized solver')
            solvers.options['show_progress'] = False
            self.problem = solvers.sdp(c,Gl=matrix_ineq,hl=vector_ineq,Gs=matrix_sdp,hs=vector_sdp,verbose=False)
        self.h = np.array(self.problem['x'])
        


    def IRNet_ROIs(self,Psi:np.ndarray,rho:float,w:np.array,locations_monitored:list=[],locations_unmonitored:list=[],epsilon:float=1e-2,primal_start={'x':[],'sl':[],'ss':[]},include_sparsity_constraint:str=True):
        """
        Network design algorithm using Candes IRL1 objective function and Rusu method to ensure binary solution.
        Method for constrinaing the maximum error variance over a subset ROI
        The lists must be updated iteratively when calling this method from the solution h.
        Equality constraints are unsupported by cvxopt solver. Using ineq constraint of epsilon tolerance.
        Implemented using cvxopt library

        min     w.T @ h
        s.t.    - 0<=h<=1
                - sum(h)>=s
                - h[S] == 1
                - h[Sc] == 0
                - M_i >>= (PSD),i=1...n 

        Args:
            Psi (np.ndarray): Low-rank basis. The basis is restrained to ROI indices
            rho (float):accuracy threshold
            w (np.array): weights vector
            locations_monitored (list, optional): indices of monitored locations from previous steps. Defaults to [].
            locations_unmonitored (list, optional): indices of unmonitored locations from previous steps. Defaults to [].
        """
        
        c = matrix(w)
        # equality constraint
        In = spmatrix(1.,range(self.n),range(self.n))
        C_monitored = In[locations_monitored,:]
        C_unmonitored = In[locations_unmonitored,:]
        #C_forbidden = In[locations_forbidden,:]
        #matrix_eq = sparse([C_monitored,C_unmonitored])
        #vector_eq = matrix([matrix(np.tile(1,len(locations_monitored))),matrix(np.tile(0,len(locations_unmonitored)))],tc='d')
        """ inequality constraints """
        # order of constraints: h>=0 | h<=1 | sum(h)>= s | h[S]>= 1-eps | h[Sc] <= eps | h[Sf] <=eps
        if include_sparsity_constraint:
            matrix_ineq = sparse([spdiag(matrix(-1,(self.n,1))),
                                spdiag(matrix(1,(self.n,1))),
                                matrix(-1,(1,self.n)),
                                sparse(-1*C_monitored),
                                sparse(C_unmonitored)])
                                #sparse(C_forbidden)])
            
            vector_ineq = matrix([matrix(np.tile(0,self.n)),
                                matrix(np.tile(1,self.n)),
                                -self.s,
                                matrix(np.tile(-(1-epsilon),len(locations_monitored))),
                                matrix(np.tile(epsilon,len(locations_unmonitored)))],tc='d')                              
                                #matrix(np.tile(epsilon,len(locations_forbidden))) ],tc='d')
        else:# order of constraints: h>=0 | h<=1 | h[S]>= 1-eps | h[Sc] <= eps | h[Sf] <=eps
            # constraint |sum(h)>= s| NOT included
            matrix_ineq = sparse([spdiag(matrix(-1,(self.n,1))),
                                spdiag(matrix(1,(self.n,1))),
                                sparse(-1*C_monitored),
                                sparse(C_unmonitored)])
                                
            vector_ineq = matrix([matrix(np.tile(0,self.n)),
                                matrix(np.tile(1,self.n)),
                                matrix(np.tile(-(1-epsilon),len(locations_monitored))),
                                matrix(np.tile(epsilon,len(locations_unmonitored)))],tc='d')
            
            
        """ LMI constraint """
        #matrix_sdp = [sparse([np.reshape(-rho*Psi[i,:][None,:].T@Psi[i,:][None,:],newshape=self.s*self.s,order='F').tolist() for i in range(self.n)])]*self.n
        #vector_sdp = [matrix(np.tril(-1*Psi[i,:][None,:].T@Psi[i,:][None,:]).astype(float)) for i in range(self.n)]
        matrix_sdp = [sparse([np.reshape(np.tril(-rho*Psi[i,:][None,:].T@Psi[i,:][None,:]),newshape=self.s*self.s,order='F').tolist() for i in range(self.n)])]
        print(f'Basis type: {type(Psi)}. Basis dtype: {Psi.dtype}')
        vector_sdp = [matrix(np.tril(-1*Psi.T@Psi).astype(float))]
        
        # solver and solution
        print('Calling SDP solver')
        try:
            self.problem = solvers.sdp(c,Gl=matrix_ineq,hl=vector_ineq,Gs=matrix_sdp,hs=vector_sdp,verbose=True,solver='dsdp',primalstart=primal_start)
        except:    
            self.problem = solvers.sdp(c,Gl=matrix_ineq,hl=vector_ineq,Gs=matrix_sdp,hs=vector_sdp,verbose=True,solver='dsdp')
        self.h = np.array(self.problem['x'])


    # other algorithms
    def WCS_placement(self,Psi):
        """
        Worst case scenario sensor placement algorithm
        Solve PSD to minimize maximum diagonal entry of covariance matrix

        Parameters
        ----------
        Psi : np array
            reduced basis


        Returns
        -------
        None.
        self.problem added to class

        """
        # compute covariance matrix
        h_eps = cp.Variable(shape=self.n,value=np.zeros(self.n))
        h_zero = cp.Variable(shape = self.n,value=np.zeros(self.n))
        t = cp.Variable(shape=(1,1))

        #S = np.zeros((self.r,self.r))
        # precision matrix as sum of both: LCSs and ref.st
        # for i in range(self.n):
        #     psi = Psi[i,:][None,:]
        #     S+=h_eps[i]*(var_eps**-1)*psi.T@psi + h_zero[i]*(var_zero**-1)*psi.T@psi
        if self.p_eps != 0:
            S = cp.sum([h_eps[i]*(self.var_eps**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] + h_zero[i]*(self.var_zero**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(self.n)])
        else:
            S = cp.sum([h_zero[i]*(self.var_zero**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(self.n)])

        constraints = [t>=0,
                       cp.sum(h_zero)==self.p_zero,
                       h_zero >= 0,
                       h_zero <=1]

        if self.p_eps !=0:
            constraints += [cp.sum(h_eps)==self.p_eps,
                            h_eps >= 0,
                            h_eps <=1,
                            h_eps + h_zero <=1]


        Ir = np.identity(self.r)
        # PSD constraints: r block matrices
        for j in np.arange(self.r):
            constraints += [cp.bmat([[t,Ir[:,j][:,None].T],[Ir[:,j][:,None],S]]) >> 0]
        problem = cp.Problem(cp.Minimize(t),constraints)
        if not problem.is_dcp():
            print('Problem not dcp\nCheck constraints and objective function')

        self.h_eps = h_eps
        self.h_zero = h_zero
        self.problem = problem

    def Liu_placement_uncorr(self,Psi):
        """
        Liu-based sensor placement for weakly-correlated noise sensors

        Parameters
        ----------
        Psi : numpy array
            low-rank basis

        Returns
        -------
        None.

        """
        R = np.identity(self.n)
        diag = np.concatenate((np.repeat(self.var_zero**-1,self.p_zero),
                               np.repeat(self.var_eps**-1,self.p_eps),
                               np.repeat(0,self.p_empty)))
        np.fill_diagonal(R, diag)
        h = cp.Variable(shape=self.n,value = np.zeros(self.n))
        H = cp.Variable(shape=(self.n,self.n),value=np.zeros((self.n,self.n)))

        F_mat = Psi.T@(cp.multiply(H, R))@Psi

        Z = cp.Variable(shape=(self.r,self.r))
        Ir = np.identity(self.r)
        M = cp.bmat([F_mat,Ir],[Ir,Z])

        objective = cp.trace(Z)


        constraints = [M >= 0,
                       cp.trace(H)<= self.p_zero + self.p_eps,
                       cp.diag(H) == h,
                       cp.bmat([ [H,h[:,None]],[h[:,None].T,np.ones((1,1))] ] )>= 0]


        problem = cp.Problem(cp.Minimize(objective),constraints)
        if not problem.is_dcp():
            print('Problem not dcp\nCheck constraints and objective function')
        self.problem = problem
        self.h = h


# =============================================================================
#        Problem initialization
# =============================================================================


    def check_consistency(self):
        """
        Check number of sensors consistency

        Raises
        ------
        SensorsErrors
            Error if the proposed configuration is not correct

        Returns
        -------
        None.

        """

        # check number of sensors consistency
        # if self.p_zero >self.r and self.p_eps!=0:
        #     raise SensorsErrors(f'The number of reference stations is larger than the dimension of the low-rank basis {self.r}')
        if self.n_unmonitored + self.s > self.n:
            raise SensorsErrors('Not enough sensors for basis sampling')
        elif any(np.array([self.n_lcs,self.n_refst,self.n_lcs])<0):
            raise SensorsErrors('Negative number of sensors')
        elif all(np.array([self.n_lcs,self.n_refst,self.n_unmonitored])>=0) and np.sum([self.n_refst,self.n_lcs,self.n_unmonitored]) != self.n:
            raise SensorsErrors('Number of sensors and empty locations mismatch total number of possible locations')
        else:
            print(f'No initialization errors found.\nNetwork summary:\n- network size: {self.n}\n- signal sparsity: {self.s}\n- number of reference stations: {self.n_refst}\n- number of LCSs: {self.n_lcs}\n- number of unmonitored locations: {self.n_unmonitored}\n- reference staions variance: {self.var_refst}\n- LCSs variance: {self.var_lcs}')

    def initialize_problem(self,Psi:np.ndarray,alpha:float=0.,rho:float=100.,w = cp.Variable(),p_min:int=0,locations_monitored:list = [],locations_unmonitored:list = [],epsilon:float=1e-2,primal_start:dict={},include_sparsity_constraint=False):
        """
        Initializes sensor placement problem

        Args:
            Psi (numpy array): low-rank basis size: (num_locations,sparsity)
            alpha (float, optional): rankMax regularization hyperparameter. Defaults to 0.
            rho (float, optional): network planning sensor placement accuracy threshold
        """
        if self.algorithm not in ['IRNet_ROI','IRL1ND']:
            self.check_consistency()
        
        algorithms = ['rankMax','MCJB','JB','JB_trace','JB_worst_variance','JB_feasibility_program','JB_feasibility_diagonal','NetworkPlanning','NetworkPlanning_iterative','NetworkPlanning_iterative_LMI','IRL1ND','IRL1ND_candes','IRNet_ROI']

        if self.algorithm not in algorithms:
            print(f'Sensor placement algorithm {self.algorithm} not implemented yet')

        elif self.algorithm == 'rankMax':
            print(f'Setting {self.algorithm} sensor placement algorithm with regularization value {alpha}')
            if self.n_unmonitored == 0:
                self.rankMax_placement(Psi,alpha)
            elif self.n_lcs == 0: # no LCS - use Doptimal
                self.var_refst = 1e0
                self.multiClass_joshiBoyd_placement(Psi)
            else:
                self.rankMax_placement(Psi,alpha)
        
        # Joshi-Boyd based algorithms
        elif self.algorithm == 'MCJB':
            print(f'Setting {self.algorithm} sensor placement algorithm')
            self.multiClass_joshiBoyd_placement(Psi)
        elif self.algorithm == 'JB':
            print(f'Setting {self.algorithm} sensor placement algorithm')
            self.JB_placement(Psi,locations_monitored,locations_unmonitored)
        elif self.algorithm == 'JB_trace':
            print(f'Setting {self.algorithm} sensor placement algorithm')
            self.JB_Trace_placement(Psi,locations_monitored,locations_unmonitored)
        elif self.algorithm == 'JB_worst_variance':
            print(f'Setting {self.algorithm} sensor placement algorithm')
            self.JB_worst_variance_placement(Psi,locations_monitored,locations_unmonitored)
        elif self.algorithm == 'JB_feasibility_program':
            print(f'Setting {self.algorithm} sensor placement algorithm')
            self.JB_feasibility_program(Psi,rho,locations_monitored,locations_unmonitored)
        elif self.algorithm == 'JB_feasibility_diagonal':
            print(f'Setting {self.algorithm} sensor placement algorithm')
            self.JB_feasibility_diagonal(Psi,rho,locations_monitored,locations_unmonitored)

        elif self.algorithm == 'NetworkPlanning':
            print(f'Setting {self.algorithm} sensor placement algorithm')
            self.networkPlanning_singleclass(Psi,rho)
        elif self.algorithm == 'NetworkPlanning_iterative':#cvxpy
            print(f'Setting {self.algorithm} sensor placement algorithm')
            self.networkPlanning_singleclass_iterative(Psi,rho,w,p_min,locations_monitored,locations_unmonitored)
        elif self.algorithm == 'NetworkPlanning_iterative_LMI':#cvxpy
            print(f'Setting {self.algorithm} sensor placement algorithm')
            self.networkPlanning_singleclass_iterative_LMI(Psi,rho,w,locations_monitored,locations_unmonitored)
        elif self.algorithm == 'IRL1ND' or self.algorithm == 'IRL1ND_candes':#cvxopt
            print(f'{self.algorithm} sensor placement algorithm')
            self.IRL1_networkDesign(Psi,rho,w,locations_monitored,locations_unmonitored,epsilon,primal_start)
        elif self.algorithm == 'IRNet_ROI':#cvxopt
            print(f'IRNet algorithm for large-scale deployments using ROIs')
            self.IRNet_ROIs(Psi,rho,w,locations_monitored,locations_unmonitored,epsilon,primal_start,include_sparsity_constraint)

        # others
        elif self.algorithm == 'WCS':
            self.WCS_placement(Psi)

        elif self.algorithm == 'rankMax_FM':
            self.simple_Dopt_placement(Psi)
            self.Psi_phase1 = Psi.copy()
            self.alpha_phase2 = alpha

        elif self.algorithm == 'Dopt-Liu':
            self.Liu_placement_uncorr(Psi)


    """Solve sensor problem"""

    def LoadLocations(self,dicts_path,alpha_reg,var_zero):
        """
        Load locations from previous training

        Parameters
        ----------
        dicts_path : str
            path to files
        alpha_reg : flaot
            regularization value
        var_zero : float
            refst covariance

        Returns
        -------
        self.dict_results : dict
            (LCSs, ref.st,empty) locations

        """
        if self.algorithm == 'rankMax' or self.algorithm == 'rankMax_FM' :
            fname = dicts_path+f'DiscreteLocations_{self.algorithm}_vs_p0_{self.n}N_r{self.r}_pEmpty{self.p_empty}_alpha{alpha_reg:.1e}.pkl'
            with open(fname,'rb') as f:
                self.dict_locations = pickle.load(f)

            fname = dicts_path+f'Weights_{self.algorithm}_vs_p0_{self.n}N_r{self.r}_pEmpty{self.p_empty}_alpha{alpha_reg:.1e}.pkl'
            with open(fname,'rb') as f:
                self.dict_weights = pickle.load(f)
        else:
            fname = dicts_path+f'DiscreteLocations_{self.algorithm}_vs_p0_{self.n}N_r{self.r}_pEmpty{self.p_empty}_varZero{var_zero:.1e}.pkl'
            with open(fname,'rb') as f:
                self.dict_locations = pickle.load(f)

            fname = dicts_path+f'Weights_{self.algorithm}_vs_p0_{self.n}N_r{self.r}_pEmpty{self.p_empty}_varZero{var_zero:.1e}.pkl'
            with open(fname,'rb') as f:
                self.dict_weights = pickle.load(f)

    def solve(self,n_it=50):
        """
        Solve sensor placement problem and print objective function optimal value
        and sensors weights

        Returns
        -------
        None.

        """
        time_init = time.time()
        print(f'Solving sensor placement using {self.algorithm} strategy')

        # some special steps for special algorithms
        if self.algorithm == 'WCS':
            solver = cp.MOSEK
            try:
                self.problem.solve(verbose=True,solver=solver)
                if self.problem.status not in ["infeasible", "unbounded"]:
                    print(f'Optimal value: {self.problem.value}')
            except:
                print('Problem not solved.\nIt is possible solver failed or problem status is unknown.')

        elif self.algorithm == 'rankMax_FM':
            try:
                # phase 1) solve simple D-optimal to determine unmonitored locations
                print('Solving basic D-optimal problem')
                self.problem_1.solve(verbose=True)
                if self.problem_1.status not in ["infeasible", "unbounded"]:
                    print(f'Optimal value: {self.problem_1.value}')
                # phase2) solve fully monitored rankMax to distribute sensors
                self.loc_unmonitored = np.sort(np.argsort(self.h.value)[:self.p_empty])
                self.loc_monitored = np.sort(np.argsort(self.h.value)[-(self.p_zero + self.p_eps):])
                Psi_restricted = self.Psi_phase1[self.loc_monitored,:]
                self.rankMax_placement(Psi_restricted, self.alpha_phase2)
                self.problem.solve(verbose=True)
                print('Solving rankMax for sensor distribution')
                if self.problem.status not in ["infeasible", "unbounded"]:
                    print(f'Optimal value: {self.problem.value}')

            except:
                print('Problem not solved.\nIt is possible solver failed or problem status is unknown.')
        
        # general solving process
        else:
            try:
                solver = cp.MOSEK
                self.problem.solve(verbose=True,solver=solver)
                if self.problem.status not in ["infeasible", "unbounded"]:
                    print(f'Optimal value: {self.problem.value}')
            except:
                print('Problem not solved.\nIt is possible solver failed or problem status is unknown.')

        time_end = time.time()
        self.exec_time = time_end - time_init
        print(f'Optimization problem solved in {self.exec_time:.2f}s')


    def discretize_solution(self):
        """
        Convert continuous weights [0,1] to discrete values {0,1}
        Get maximum entries of h and use those locations to place sensors

        Returns
        -------
        None.

        """
        
        if self.algorithm == 'rankMax_FM':
            order_refst = self.loc_monitored[np.sort(np.argsort(self.h_zero.value)[-self.n_refst:])]
            order_lcs = np.sort(np.setdiff1d(self.loc_monitored,order_refst))
            order_unmonitored = np.sort(self.loc_unmonitored)
        
        # sensor placement algorithms with single class of sensors
        elif self.algorithm == 'JB':
            order_lcs = []
            order_refst = np.sort(np.argsort(self.h.value)[-self.n_refst:])
            order_unmonitored = np.sort([i for i in np.arange(self.n) if i not in order_refst])
        elif self.algorithm == 'JB_trace':
            order_lcs = []
            order_refst = np.sort(np.argsort(self.h.value)[-self.n_refst:])
            order_unmonitored = np.sort([i for i in np.arange(self.n) if i not in order_refst])
        elif self.algorithm == 'JB_worst_variance':
            order_lcs = []
            order_refst = np.sort(np.argsort(self.h.value)[-self.n_refst:])
            order_unmonitored = np.sort([i for i in np.arange(self.n) if i not in order_refst])
        elif self.algorithm in ['JB_feasibility_program','JB_feasibility_diagonal']:
            order_lcs = []
            order_refst = np.sort(np.argsort(self.h.value)[-self.n_refst:])
            order_unmonitored = np.sort([i for i in np.arange(self.n) if i not in order_refst])
        elif self.algorithm == 'NetworkPlanning':
            self.n_refst = int(np.round(self.h.value.sum()))
            order_lcs = []
            order_refst = np.sort(np.argsort(self.h.value)[-self.n_refst:])
            order_unmonitored = np.sort([i for i in np.arange(self.n) if i not in order_refst])

        else:
            if self.h_lcs.value.sum() == 0.0 and self.h_refst.value.sum() == 0.0: # placement failure
                print('Sensor placement failed. Setting to zero')
                order_refst, order_lcs = np.zeros(self.n), np.zeros(self.n)
            elif self.n_refst == 0:# no ref.st.
                print('There are no Reference stations in the network. Placing LCSs')
                order_refst = []
                order_lcs = np.sort(np.argsort(self.h_lcs.value)[-self.n_lcs:])
            elif self.n_lcs == 0:# no LCSs
                print('There are no LCSs in the network. Placing reference stations')
                order_refst= np.sort(np.argsort(self.h_refst.value)[-self.n_refst:])
                order_lcs = []
            else:
                order_refst = np.sort(np.argsort(self.h_refst.value)[-self.n_refst:])
                order_lcs = np.sort([i for i in np.flip(np.argsort(self.h_lcs.value)) if i not in order_refst][:self.n_lcs])

            order_unmonitored = np.sort([i for i in np.arange(self.n) if i not in order_lcs and i not in order_refst])

            if (set(order_lcs).issubset(set(order_refst)) or set(order_refst).issubset(set(order_lcs))) and (self.n_lcs!=0 and self.n_refst!=0) :
                print('Some locations of LCSs are already occupied by a reference station')
                order_lcs = [i for i in np.arange(self.n) if i not in order_refst]

        self.locations = [order_lcs,order_refst,order_unmonitored]
        print(f'Monitoring locations:\n- reference stations ({len(self.locations[1])} locations in total): {self.locations[1]}\n- LCSs ({len(self.locations[0])} locations in total): {self.locations[0]}\n- unmonitored locations ({len(self.locations[2])} locations in total): {self.locations[2]}')
        

    def compute_Doptimal(self,Psi,alpha):
        """
        Computes D-optimal metric of continuous index obtained from rankMin_reg solution

        Parameters
        ----------
        Psi : numpy array
            low-rank basis.
        alpha : float
            reference station rank regularization parameter

        Returns
        -------
        None.

        """
        precision_matrix = (self.var_eps**-1)*Psi.T@np.diag(self.h_eps.value)@Psi + (self.var_zero**-1)*Psi.T@np.diag(self.h_zero.value)@Psi
        self.Dopt_metric = -1*np.log(np.linalg.det(precision_matrix))

        self.logdet_eps = np.log(np.linalg.det(Psi.T@np.diag(self.h_eps.value)@Psi))
        self.trace_zero = alpha*np.trace(Psi.T@np.diag(self.h_zero.value)@Psi)





# =============================================================================
#         Compute covariance matrix and regressor from solutions
# =============================================================================


    def C_matrix(self):
        """
        Convert indexes of LCSs, RefSt and Emtpy locations into
        measurement matrix C

        Returns
        -------
        self.C: list
            C matrix for LCSs, RefSt, Empty


        """
        # select certain rows from identity matrix
        In = np.identity(self.n)
        C_lcs = In[self.locations[0],:]
        C_refst = In[self.locations[1],:]

        if self.n_unmonitored != 0:
            C_unmonitored = In[self.locations[2],:]
        else:
            C_unmonitored = []
        self.C = [C_lcs,C_refst,C_unmonitored]

    def compute_convex_covariance_matrix(self,Psi,weights,var_zero,var_eps,metric='logdet'):
        C_eps = np.diag(weights[0])
        C_refst = np.diag(weights[1])
        Theta_eps = C_eps@Psi
        Theta_refst = C_refst@Psi

        Precision_matrix = (var_eps**(-1)*Theta_eps.T@Theta_eps) + (var_zero**(-1)*Theta_refst.T@Theta_refst)
        try:
            self.Cov_convex = np.linalg.inv(Precision_matrix)
        except:
            print('Computing pseudo-inverse')
            self.Cov_convex = np.linalg.pinv(Precision_matrix)

        if metric=='logdet':
            self.metric_convex = np.log(np.linalg.det(self.Cov_convex))


    def covariance_matrix(self,Psi,metric='logdet',alpha=0.1,activate_error_solver=True):
        """
        Compute covariance matrix from C matrices and compute a metric

        Parameters
        ----------
        Psi : np array
            low-rank basis used
        metric : str
            metric to compute covariance optimizality: D-optimal, E-optimal, WCS
        alpha: float
            regularization parameter for proposed algorithm

        Returns
        -------
        None.

        """
        C_eps = self.C[0]
        C_zero = self.C[1]

        Theta_eps = C_eps@Psi
        Theta_zero = C_zero@Psi
        if self.p_eps !=0:
            self.Precision_matrix = (self.var_eps**(-1)*Theta_eps.T@Theta_eps) + (self.var_zero**(-1)*Theta_zero.T@Theta_zero)
            try:
                self.Cov = np.linalg.inv( self.Precision_matrix )
            except:
                print('Computing pseudo-inverse')
                self.Cov = np.linalg.pinv( self.Precision_matrix )

        else:
            self.Precision_matrix = (self.var_zero**(-1)*Theta_zero.T@Theta_zero)
            try:
                self.Cov = np.linalg.inv(self.Precision_matrix)
            except:
                print('Computing pseudo-inverse')
                self.Cov = np.linalg.pinv(self.Precision_matrix)

        # compute cov-matrix metric

        if metric == 'logdet':
            self.metric = np.log(np.linalg.det(self.Cov))
            self.metric_precisionMatrix = np.log(np.linalg.det(self.Precision_matrix))

        elif metric == 'eigval':
            self.metric = np.max(np.real(np.linalg.eig(self.Cov)[0]))
            self.metric_precisionMatrix = np.min(np.real(np.linalg.eig(self.Precision_matrix)[0]))

        elif metric == 'WCS':
            self.metric = np.diag(self.Cov).max()

        elif metric == 'logdet_rank':
            self.metric = np.log(np.linalg.det(self.var_eps**(-1)*Theta_eps.T@Theta_eps)) + alpha*np.trace(self.var_zero**(-1)*Theta_zero.T@Theta_zero)

        if activate_error_solver:
            if type(self.problem.value) == type(None):# error when solving
                self.metric = np.inf
                self.metric_precisionMatrix = -np.inf

    def covariance_matrix_GLS(self,Psi):
        """
        Compute covariance matrix from GLS. Use pseudo-inverse to account for unstable behavior

        Parameters
        ----------
        Psi : numpy array
            low-rank basis

        Returns
        -------
        None.

        """
        C_lcs = self.C[0]
        C_refst = self.C[1]
        Theta_lcs = C_lcs@Psi
        Theta_refst = C_refst@Psi

        if C_lcs.shape[0] == 0:# no LCSs
            Precision_matrix = (self.var_zero**-1)*Theta_refst.T@Theta_refst
        elif C_refst.shape[0] == 0:#no Ref.St.
            Precision_matrix = (self.var_eps**-1)*Theta_lcs.T@Theta_lcs
        else:
            Precision_matrix = (self.var_zero**-1)*Theta_refst.T@Theta_refst + (self.var_eps**-1)*Theta_lcs.T@Theta_lcs

        try:
            S = np.linalg.svd(Precision_matrix)[1]
            rcond_pinv = rcond_pinv = (S[-1]+S[-2])/(2*S[0])
            Cov = np.linalg.pinv(Precision_matrix,rcond_pinv)
        except:
            Cov = np.linalg.pinv(Precision_matrix,hermitian=True)

        self.Cov = Cov

    def covariance_matrix_limit(self,Psi):
        """
        Compute covariance matrix in the limit var_zero = 0

        Parameters
        ----------
        Psi : numpy array
            low-rank basis


        Returns
        -------
        None.

        """
        C_lcs = self.C[0]
        C_refst = self.C[1]
        Theta_lcs = C_lcs@Psi
        Theta_refst = C_refst@Psi


        if C_lcs.shape[0] == 0:#no LCSs
            self.Cov = np.zeros(shape=(self.r,self.r))
            return
        elif C_refst.shape[0] == 0:#no RefSt
            Precision_matrix = (self.var_eps**-1)*Theta_lcs.T@Theta_lcs
            S = np.linalg.svd(Precision_matrix)[1]
            rcond_pinv = rcond_pinv = (S[-1]+S[-2])/(2*S[0])
            self.Cov = np.linalg.pinv( Precision_matrix,rcond_pinv)

            return

        else:
            # compute covariance matrix using projector
            refst_matrix = Theta_refst.T@Theta_refst

            Is = np.identity(self.r)
            try:
                P = Is - refst_matrix@np.linalg.pinv(refst_matrix)
                #P = Is - refst_matrix@np.linalg.pinv(refst_matrix,hermitian=True,rcond=1e-10)
            except:
                P = Is - refst_matrix@np.linalg.pinv(refst_matrix,hermitian=True,rcond=1e-10)

            rank1 = np.linalg.matrix_rank(Theta_lcs@P,tol=1e-10)
            rank2 = np.linalg.matrix_rank(P@Theta_lcs.T,tol=1e-10)

            S1 = np.linalg.svd(Theta_lcs@P)[1]
            S2 = np.linalg.svd(P@Theta_lcs.T)[1]

            if rank1==min((Theta_lcs@P).shape):
                try:
                    rcond1_pinv = (S1[-1]+S1[-2])/(2*S1[0])
                except:
                    rcond1_pinv = 1e-15
            else:
                rcond1_pinv = (S1[rank1]+S1[rank1-1])/(2*S1[0])

            if rank2==min((P@Theta_lcs.T).shape):
                try:
                    rcond2_pinv = (S2[-1]+S2[-2])/(2*S2[0])
                except:
                    rcond2_pinv = 1e-15
            else:
                rcond2_pinv = (S2[rank2]+S2[rank2-1])/(2*S2[0])

            self.Cov = self.var_eps*np.linalg.pinv(Theta_lcs@P,rcond=rcond1_pinv)@np.linalg.pinv(P@Theta_lcs.T,rcond=rcond2_pinv)
            #self.Cov = self.var_eps*P@np.linalg.pinv(Theta_lcs.T@Theta_lcs,hermitian=True,rcond=1e-10)@P




    def beta_estimated_GLS(self,Psi,y_refst,y_lcs):
        """
        Compute estimated regressor (beta) from sensor measurements

        Parameters
        ----------
        Psi : numpy array
            sparse basis
        y_refst : numpy array
            reference stations vector of measurements
        y_lcs : numpy array
            LCSs vector of measurements

        Returns
        -------
        self.beta_hat : numpy array
                estimated regressor
                Regressor evolution over time (r,num_samples)

        """
        C_eps = self.C[0]
        C_zero = self.C[1]
        Theta_eps = C_eps@Psi
        Theta_zero = C_zero@Psi
        second_term = (self.var_zero**-1)*Theta_zero.T@y_refst + (self.var_eps**-1)*Theta_eps.T@y_lcs
        self.beta_hat = self.Cov@second_term


    def beta_estimated_limit(self,Psi,y_refst,y_lcs):
        """
        Compute estimated regressor (beta) from sensor measurements
        in the limit variances refst goes to zero (limit of GLS)

        Parameters
        ----------
        Psi : numpy array
            sparse basis
        y_refst : numpy array
            reference stations vector of measurements
        y_lcs : numpy array
            LCSs vector of measurements

        Returns
        -------
        self.beta_hat : numpy array
                estimated regressor over time (r,num_samples)

        """
        C_lcs = self.C[0]
        C_refst = self.C[1]
        Theta_lcs = C_lcs@Psi
        Theta_refst = C_refst@Psi
        refst_matrix = Theta_refst.T@Theta_refst

        Is = np.identity(self.r)
        P = Is - refst_matrix@np.linalg.pinv(refst_matrix)

        term_refst = np.linalg.pinv(Theta_refst)
        term_lcs = np.linalg.pinv(Theta_lcs@P)@np.linalg.pinv(P@Theta_lcs.T)@Theta_lcs.T

        self.beta_hat = term_lcs@y_lcs + term_refst@y_refst


    def save_locations(self,files_path):
        """
        Save generated list of locations. Solutions to sensor placement problem for a given basis.
        """
        fname = f'{files_path}SensorsLocations_N{self.n}_RefSt{self.n_refst}_LCSs{self.n_lcs}_Sparsity{self.s}.pkl'
        with open(fname,'wb') as f:
            pickle.dump(self.locations, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Sensor placement locations results saved at: {fname}')

if __name__ == '__main__':
    """
    Network parameters:
        - number of sensors per class
        - signal sparsity
        - variance of sensors
    """
    N = 50#691150
    #SIGNAL_SPARSITY = 25#280
    N_LCS, N_REFST = 0,N
    N_UNMONITORED = N - N_REFST - N_LCS
    VAR_LCS, VAR_REFST = 1,1e-2
    NUM_SAMPLES = int(5e2)
    """ Create test measurement matrix, snapshots matrix and low-rank basis """
    rng = np.random.default_rng(seed=40)
    X = rng.normal(loc=0.0,scale=1.0,size=(NUM_SAMPLES,N))
    snapshots_matrix = X.T
    snapshots_matrix_centered = snapshots_matrix - snapshots_matrix.mean(axis=1)[:,None]
    U,S,Vt = np.linalg.svd(snapshots_matrix,full_matrices=False)
    cumulative_energy = np.cumsum(S)/np.sum(S)
    energy_threshold = 0.9
    sparsity_energy = np.where(cumulative_energy>=energy_threshold)[0][0]
    print(f'Cumulative energy surpasses the threshold of {energy_threshold:.2f} at index: {sparsity_energy}')
    Psi = U[:,:sparsity_energy]
    print(f'Network parameters:\n - network size: {N}\n - signal sparsity: {sparsity_energy}\n - snapshots matrix shape: {snapshots_matrix.shape}\n - Psi basis shape: {Psi.shape}')
    
    """ compute variance-covariance matrix """
    coordinate_error_variance_fullymonitored = np.diag(Psi@np.linalg.inv(Psi.T@Psi)@Psi.T)
    maxvariance_fullymonitored = coordinate_error_variance_fullymonitored.max()
    variance_threshold_ratio = 10.
    design_threshold = variance_threshold_ratio*maxvariance_fullymonitored
    print(f'Network design parameters:\n - worst coordinate error variance fully monitored network: {maxvariance_fullymonitored:.3f}\n - design threshold: {variance_threshold_ratio:.2f}\n - design variance threshold: {design_threshold:.3f}')
    
    """ Solve network planning problem """
    algorithm = 'IRL1ND'#['NetworkPlanning_iterative','IRL1ND']
    locations_monitored = []
    locations_unmonitored = []
    sensor_placement = SensorPlacement(algorithm, N, sparsity_energy,N_REFST,N_LCS,N_UNMONITORED)
    epsilon = 5e-2
    n_it = 20
    h_prev = np.zeros(N)
    if len(locations_monitored) !=0:
        h_prev[locations_monitored] = 1
    w = 1/(h_prev+epsilon)
    it = 0

    if algorithm == 'IRL1ND':
        """ Uses cvxopt library """
        primal_start = {'x':[],'sl':[],'ss':[]}
        # iterative method
        time_init = time.time()
        while len(locations_monitored) + len(locations_unmonitored) != N:
            # solve sensor placement with constraints
            sensor_placement.initialize_problem(Psi,rho=design_threshold,
                                                w=w,locations_monitored=locations_monitored,locations_unmonitored=locations_unmonitored,
                                                epsilon=epsilon,primal_start=primal_start)
            if sensor_placement.problem['status'] == 'optimal':
                # get solution
                primal_start['x'] = sensor_placement.problem['x']
                primal_start['sl'] = sensor_placement.problem['sl']
                primal_start['ss'] = sensor_placement.problem['ss']
                # update sets
                new_monitored = [int(i[0]) for i in np.argwhere(sensor_placement.h >= 1-epsilon) if i[0] not in locations_monitored]
                new_unmonitored = [int(i[0]) for i in np.argwhere(sensor_placement.h <= epsilon) if i[0] not in locations_unmonitored]
                locations_monitored += new_monitored
                locations_unmonitored += new_unmonitored
                # check convergence
                if np.linalg.norm(sensor_placement.h - h_prev)<=epsilon or it==n_it:
                    locations_monitored += [[int(i[0]) for i in np.argsort(sensor_placement.h,axis=0)[::-1] if i not in locations_monitored][0]]
                    it = 0
                h_prev = sensor_placement.h
                w_old = w.copy()
                w = 1/(h_prev + epsilon)
                it +=1
                print(f'Iteration results\n ->Primal objective: {sensor_placement.problem["primal objective"]:.6f}\n ->{len(locations_monitored) + len(locations_unmonitored)} locations assigned\n ->{len(locations_monitored)} monitored locations\n ->{len(locations_unmonitored)} unmonitored locations\n')
            else:
                # solver fails at iteration
                w = w_old
                #locations_monitored = locations_monitored[:-len(new_monitored)]
                locations_unmonitored = locations_unmonitored[:-len(new_unmonitored)]
                it+=1
        time_end = time.time()
        
        sensor_placement.locations = [[],np.sort(locations_monitored),np.sort(locations_unmonitored)]
        sensor_placement.C_matrix()
        """
        pool = multiprocessing.Pool(processes=4)
        print(pool.map(func2, range(n)))
        pool.close()
        pool.join()
        """
    
    elif algorithm == 'IRL1ND_candes':
        """ No wrapper for iterating until all locations are assigned"""
        h_prev = np.zeros(N)
        epsilon = 1e-2
        w = 1/(h_prev+epsilon)
        n_it = 100
        it = 0
        locations_monitored = []
        locations_unmonitored = []
        # iterative method
        time_init = time.time()
        
        while it < n_it:
            # solve sensor placement with constraints
            sensor_placement.initialize_problem(Psi,rho=design_threshold,
                                                w=w,locations_monitored=locations_monitored,locations_unmonitored=locations_unmonitored)
        
            if np.linalg.norm(sensor_placement.h - h_prev)<=epsilon:
                print(f'Convergence criteria reached: {np.linalg.norm(sensor_placement.h - h_prev):.2e}')
                break
            h_prev = sensor_placement.h.copy()
            w = 1/(h_prev + epsilon)
            it +=1
        
        time_end = time.time()
        n_sensors = int(np.ceil(sensor_placement.problem['primal objective']))
        locations_monitored = np.argsort([i[0] for i in sensor_placement.h])[-n_sensors:]
        locations_unmonitored = [i for i in np.arange(sensor_placement.n)if i not in locations_monitored]
        sensor_placement.locations = [[],np.sort(locations_monitored),np.sort(locations_unmonitored)]
        sensor_placement.C_matrix()
        print(f'{len(locations_monitored)} Locations monitored: {locations_monitored}\n{len(locations_unmonitored)} Locations unmonitored: {locations_unmonitored}\n')

    elif algorithm == 'NetworkPlanning_iterative':
        """ Uses cvxpy library. LMIs are not explicitly expressed """

        # iterative method
        time_init = time.time()
        while len(locations_monitored) + len(locations_unmonitored) != N:
            # solve sensor placement with constraints
            sensor_placement.initialize_problem(Psi,rho=design_threshold,
                                                w=w,locations_monitored=locations_monitored,locations_unmonitored=locations_unmonitored)
            sensor_placement.solve()
            # update sets
            
            locations_monitored += [i[0] for i in np.argwhere(sensor_placement.h.value >= 1-epsilon) if i[0] not in locations_monitored]
            locations_unmonitored += [i[0] for i in np.argwhere(sensor_placement.h.value <= epsilon) if i[0] not in locations_unmonitored]
            # check convergence
            if np.linalg.norm(sensor_placement.h.value - h_prev)<=epsilon or it==n_it:
                locations_monitored += [[i for i in np.argsort(sensor_placement.h.value)[::-1] if i not in locations_monitored][0]]
                it = 0
            h_prev = sensor_placement.h.value
            w = 1/(h_prev + epsilon)
            it +=1
            print(f'{len(locations_monitored)} Locations monitored: {locations_monitored}\n{len(locations_unmonitored)} Locations unmonitored: {locations_unmonitored}\n')
        
        time_end = time.time()
        sensor_placement.locations = [[],np.sort(locations_monitored),np.sort(locations_unmonitored)]
        sensor_placement.C_matrix()

    elif algorithm == 'NetworkPlanning':
        sensor_placement.initialize_problem(Psi,rho=design_threshold)
        time_init = time.time()
        sensor_placement.solve()
        sensor_placement.discretize_solution()
        time_end = time.time()
        sensor_placement.C_matrix()


    elif algorithm == 'IRNet_ROI':
        """ algorithm for large-scale deployments. Uses cvxopt library"""
        algorithm = 'IRL1ND'#['IRL1ND','IRNet_ROI']
        epsilon = 1e-2
        n_it = 20
        locations_monitored = []
        locations_unmonitored = []
        locations_monitored_roi = []
        locations_unmonitored_roi = []

        idx_roi = np.array([],dtype=int)
        n_elements_roi = 20
        idx_range = np.arange(0,Psi.shape[0],1)
        rng.shuffle(idx_range)

        #if N%n_elements_roi != 0:
        #    raise ValueError(f'Number of elements per ROI ({n_elements_roi}) mismatch total number of potential locations in network ({N})')

        # randomly select group of entries of Psi matrix
        time_init = time.time()
        for i in np.arange(0,N,n_elements_roi):
            # select new ROI and append to previous ROI
            idx_roi_new = idx_range[i:i+n_elements_roi]
            print(f'\n\nNumber of elements in new ROI: {len(idx_roi_new)}')
            print(f'New ROI covers indices {idx_roi_new}')
            if len(idx_roi_new) == 0:
                print(f'Empty ROI')
                continue
            # idx_roi contains indices from large matrix that compose the ROI
            idx_roi = np.sort(np.unique(np.concatenate((idx_roi,idx_roi_new),axis=0,dtype=np.int64)))
            print(f'Indices in union of cumulative ROI: {idx_roi}')
            if len(idx_roi) < sparsity_energy:
                print(f'Current ROI has less elements ({len(idx_roi)}) than signal sparsity ({sparsity_energy}).\nSkipping to new ROI to increase the number of elements')
                continue
            Psi_roi = Psi[idx_roi,:]
            fully_monitored_network_max_variance_roi = np.diag(Psi_roi@np.linalg.inv(Psi_roi.T@Psi_roi)@Psi_roi.T).max()
            #fully_monitored_network_max_variance_roi = np.diag(Psi@np.linalg.inv(Psi.T@Psi)@Psi.T)[idx_roi].max()
            # determine design threshold on current ROI
            deployed_network_variance_threshold_roi = variance_threshold_ratio*fully_monitored_network_max_variance_roi
            n_roi = Psi_roi.shape[0]
            print(f'Number of potential locations in ROI: {n_roi}')
            print(f'Fully monitored max error variance in ROI: {fully_monitored_network_max_variance_roi:.2f}\nDesign max error variance in ROI: {deployed_network_variance_threshold_roi:.2f}')
            print(f'Overall fully monitored max error variance: {maxvariance_fullymonitored:.2f}\nOverall design max error variance: {design_threshold:.2f}')

            # IRNet method parameters            
            sensor_placement = SensorPlacement(algorithm,n_roi,sparsity_energy,n_refst=n_roi,n_lcs=0,n_unmonitored=0)
            epsilon_zero = epsilon#/10
            primal_start_roi = {'x':[],'sl':[],'ss':[]}
            it_roi = 0
            h_prev_roi = np.zeros(n_roi)
            weights_roi = 1/(h_prev_roi+epsilon)
            # carry monitored locations from previous ROI step
            # algorithm output for locations_monitored_roi contains indices (0...len(Psi_roi)). Convert to overall indices in original matrix
            if len(locations_monitored)!=0:
                locations_monitored_roi = np.where(np.isin(idx_roi,locations_monitored))[0].tolist()
            else:
                locations_monitored_roi = []
            print(f'Monitored locations from previous step: {locations_monitored_roi}')

            if len(locations_unmonitored)!=0:
                locations_unmonitored_roi = np.where(np.isin(idx_roi,locations_unmonitored))[0].tolist()
            else:
                locations_unmonitored_roi = []
            print(f'Unmonitored locations from previous step: {locations_unmonitored_roi}')
            
            new_monitored_roi = []
            new_unmonitored_roi = []

            # begin IRNet iterations
            while len(locations_monitored_roi) + len(locations_unmonitored_roi) != n_roi:
                print(f'New monitored indices: {new_monitored_roi}')
                print(f'New unmonitored indices: {new_unmonitored_roi}')
                # solve sensor placement with constraints
                sensor_placement.initialize_problem(Psi_roi,rho=deployed_network_variance_threshold_roi,w=weights_roi,epsilon=epsilon,
                                                locations_monitored=locations_monitored_roi,locations_unmonitored = locations_unmonitored_roi,
                                                primal_start=primal_start_roi)
                
                if sensor_placement.problem['status'] == 'optimal':
                    # get solution dictionary
                    print(f'Optimization locations results: {sensor_placement.h}')
                    primal_start_roi['x'] = sensor_placement.problem['x']
                    primal_start_roi['sl'] = sensor_placement.problem['sl']
                    primal_start_roi['ss'] = sensor_placement.problem['ss']
                    # update sets: get entries (from constrained basis) of monitored and unmonitored locations
                    new_monitored_roi = [int(i[0]) for i in np.argwhere(sensor_placement.h >= 1-epsilon) if i[0] not in locations_monitored_roi]
                    new_unmonitored_roi = [int(i[0]) for i in np.argwhere(sensor_placement.h <= epsilon_zero) if i[0] not in locations_unmonitored_roi]
                    locations_monitored_roi += new_monitored_roi
                    locations_unmonitored_roi += new_unmonitored_roi
                    # check convergence: update entries of monitored locations
                    if np.linalg.norm(sensor_placement.h - h_prev_roi)<=epsilon or it_roi==n_it:
                        new_monitored_roi = [[int(i[0]) for i in np.argsort(sensor_placement.h,axis=0)[::-1] if i not in locations_monitored_roi][0]]
                        locations_monitored_roi += new_monitored_roi
                        it_roi = 0
                        print(f'Maximum iteration reached. Updating locations monitored ({len(locations_monitored_roi)}): {locations_monitored_roi}')
                    h_prev_roi = sensor_placement.h
                    weights_roi = 1/(h_prev_roi + epsilon)
                    it_roi +=1
                    print(f'Iteration results\n ->Primal objective: {sensor_placement.problem["primal objective"]:.6f}\n ->{len(locations_monitored_roi) + len(locations_unmonitored_roi)} locations assigned\n ->{len(locations_monitored_roi)} monitored locations\n ->{len(locations_unmonitored_roi)} unmonitored locations\n Basis indices of monitored locations at iteration: {locations_monitored_roi}\n Basis indices of unmonitored locations at iteration: {locations_unmonitored_roi} ')
                    sys.stdout.flush()
                
                elif sensor_placement.problem['status'] == 'unknown':
                    print('Problem status')
                    print(sensor_placement.problem['status'])

                    if np.linalg.norm(sensor_placement.h - h_prev_roi)<=epsilon or it_roi==n_it:
                        new_monitored_roi = [[int(i[0]) for i in np.argsort(sensor_placement.h,axis=0)[::-1] if i not in locations_monitored_roi][0]]
                        locations_monitored_roi += new_monitored_roi
                        it_roi = 0
                        print(f'Maximum iteration reached. Updating locations monitored ({len(locations_monitored_roi)}): {locations_monitored_roi}')

                    it_roi += 1
                else: # unfeasible solution: step back and free an unmonitored location
                    print('Problem status')
                    print(sensor_placement.problem['status'])
                    # solver fails at iteration
                    #locations_monitored = locations_monitored[:-len(new_monitored)]
                    locations_unmonitored_roi = locations_unmonitored_roi[:-len(new_unmonitored_roi)]
                    it_roi+=1
                
            
            # add monitored locations from ROI to list of overall monitored locations of original basis entries
            locations_monitored = list(idx_roi[locations_monitored_roi])
            locations_unmonitored = list(idx_roi[locations_unmonitored_roi])
            print(f'ROI results:\n Locations monitored ({len(locations_monitored)}): {locations_monitored}\n Unmonitored locations ({len(locations_unmonitored)}): {locations_unmonitored}')
            

        time_end = time.time()
        sensor_placement.locations = [[],np.sort(locations_monitored),np.sort(locations_unmonitored)]
        sensor_placement.C_matrix()

    deployed_network_accuracy = np.diag(Psi@np.linalg.inv(Psi.T@sensor_placement.C[1].T@sensor_placement.C[1]@Psi)@Psi.T).max()
    print(f'Discretized network solution:\n- Number of potential locations: {sensor_placement.n}\n- Number of monitored locations: {sensor_placement.locations[1].shape}\nMonitored locations: {sensor_placement.locations[1]}\n- Number of unmonitored locations: {sensor_placement.locations[2].shape}\n- Fully monitoring network maximum signal variance: {maxvariance_fullymonitored:.2f}\n- Network design worst coordinate error variance threshold: {design_threshold:.2f}\n- Deployed monitoring network maximum variance: {deployed_network_accuracy:.2f}')
    print("\u0332".join(f"Finished in {time_end-time_init:.2f}s"))
    
    