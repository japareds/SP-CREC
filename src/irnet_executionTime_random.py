#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import pickle
import numpy as np
from scipy.spatial.distance import pdist, squareform
from abc import ABC,abstractmethod
from cvxopt import matrix, solvers, spmatrix, sparse, spdiag


# sensor placement class
class OptimizationProblem(ABC):
    @abstractmethod
    def create_matrices_sdp(self, Psi, phi_matrix, constant_matrix, design_threshold,n,signal_sparsity, sensor_variance,**kwargs):
        raise NotImplementedError
    def solve(self,**kwargs):
        raise NotImplementedError

class OptimizationProblemCVXOPT(OptimizationProblem):
    def create_matrices_ineq(self,n:int,signal_sparsity:int,epsilon:float,locations_monitored:list,locations_unmonitored:list):
        """Create cvxopt matrices for Inequality constraints

        Args:
            n (int): _description_
            Psi (np.ndarray): _description_
            signal_sparsity (int): _description_
            epsilon (float): _description_
            locations_monitored (list): _description_
            locations_unmonitored (list): _description_
            n_monitored_previous_batches (int): _description_

        Returns:
            sparse matrix: _description_
        """
        # order of constraints: h>=0 | h<=1 | sum(h)>= s | h[S]>= 1-eps | h[Sc] <= eps | h[Sf] <=eps

        # Used by Rusu's wrapper: h[S] >= 1-epsilon | h[Sc] <= epsilon
        In = spmatrix(1.,range(n),range(n))
        C_monitored = In[locations_monitored,:]
        C_unmonitored = In[locations_unmonitored,:]
        
        # matrix objects
        matrix_ineq = sparse([spdiag(matrix(-1,(n,1))),
                                spdiag(matrix(1,(n,1))),
                                matrix(-1,(1,n)),
                                sparse(-1*C_monitored),
                                sparse(C_unmonitored)])
    
        vector_ineq = matrix([matrix(np.tile(0,n)),
                                matrix(np.tile(1,n)),
                                -np.double(signal_sparsity),
                                matrix(np.tile(-(1-epsilon),len(locations_monitored))),
                                matrix(np.tile(epsilon,len(locations_unmonitored)))],tc='d')
                    
        return matrix_ineq, vector_ineq
                                


    def create_matrices_sdp(self, n, Psi, constant_matrix, design_threshold,signal_sparsity, sensor_variance):
        """ Create cvxopt matrices for LMI constraint
        M1*h1 + ... + Mn*hn <= Mc
        matrix_sdp is a list of matrix() object.
        Each ith matrix_i object corresponds to the ith Conic mapping.
        Each matrix_i is composed from a list of n elements where j-th entry is the vectorized matrix Mj
        
        vector_sdp is a list of matrix() object.
        Each ith matrix_i object corresponds to the ith constant matrix Mc for the ith LMI

        """
        
        # constant design_threshold
        if type(design_threshold) in [float,np.float32,np.float64]:
            matrix_sdp = [sparse([np.reshape(np.tril(-(design_threshold/sensor_variance)*Psi[i,:][None,:].T@Psi[i,:][None,:]),newshape=signal_sparsity*signal_sparsity,order='F').tolist() for i in range(n)])]*n
            #matrix_sdp = [sparse([np.reshape(np.tril(-rho*Psi[i,:][None,:].T@Psi[i,:][None,:]),newshape=s*s,order='F').tolist() for i in range(n)])]*n
        elif len(design_threshold) == 1:
            matrix_sdp = [sparse([np.reshape(np.tril(-(design_threshold[0]/sensor_variance)*Psi[i,:][None,:].T@Psi[i,:][None,:]),newshape=signal_sparsity*signal_sparsity,order='F').tolist() for i in range(n)])]*n
        # different design_threshold for each location
        elif len(design_threshold) != n:
            raise ValueError(f'Design threshold is a list of {len(design_threshold)} elements but there are {n} potential locations.')
        else:
            matrix_sdp = [sparse([np.reshape(np.tril(-(design_threshold[i]/sensor_variance)*Psi[j,:][None,:].T@Psi[j,:][None,:]),newshape=signal_sparsity*signal_sparsity,order='F').tolist() for j in range(n)]) for i in range(n)]
        
        vector_sdp = [matrix(np.tril(-1*(Psi[i,:][None,:].T@Psi[i,:][None,:] - (design_threshold[i]/sensor_variance)*constant_matrix)).astype(float)) for i in range(n)]

        return matrix_sdp,vector_sdp
    
    def solve(self,n,Psi:np.ndarray,signal_sparsity:float,design_threshold:list,sensor_variance:float,constant_matrix:np.ndarray,weights:np.array,epsilon:float=1e-2,locations_monitored:list=[],locations_unmonitored:list=[],n_monitored_previous_batches:int=0,primal_start={'x':[],'sl':[],'ss':[]}):
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
            rho (list): accuracy design threshold
            w (np.array): weights vector
            locations_monitored (list, optional): indices of monitored locations from previous steps. Updated by Rusu's wrapper. Defaults to [].
            locations_unmonitored (list, optional): indices of unmonitored locations from previous steps. Updated by Rusu's wrapper. Defaults to [].
        """
        
        
        # inequality constraints 
        matrix_ineq, vector_ineq = self.create_matrices_ineq(n=n,signal_sparsity=signal_sparsity,epsilon=epsilon,
                                                             locations_monitored=locations_monitored,locations_unmonitored=locations_unmonitored)

        """ SDP constraints
            order of constraints: -rho_i*sum_j (Psi_j.T*Psi_j)*h_j <= -Psi_i.T*Psi_i + Const_matrix. for i=1,...,n
            matrix_sdp = sum_j(Psi_j.T*Psi_j)
            vector_sdp = -Psi_i + Const_matrix
        """
        matrix_sdp,vector_sdp = self.create_matrices_sdp(n=n,Psi=Psi,constant_matrix=constant_matrix,
                                                         design_threshold=design_threshold,signal_sparsity=Psi.shape[1],sensor_variance=sensor_variance
                                                         )
        # solver and solution
        c = matrix(weights)
        print('Calling SDP solver')
        try:
            solvers.options['show_progress'] = True
            self.problem = solvers.sdp(c,Gl=matrix_ineq,hl=vector_ineq,Gs=matrix_sdp,hs=vector_sdp,verbose=False,solver='dsdp',primalstart=primal_start)
            print('dsdp solver found')
        except:    
            print('Solving using non-specialized solver')
            solvers.options['show_progress'] = True
            self.problem = solvers.sdp(c,Gl=matrix_ineq,hl=vector_ineq,Gs=matrix_sdp,hs=vector_sdp,verbose=False)
        self.h = np.array(self.problem['x'])

class NetworkDesign(ABC):
    @abstractmethod
    def design(self,**kwargs):
        raise NotImplementedError
class RusuWrapperCVXOPT(NetworkDesign):
    def __init__(self):
        pass
    def design(self,n:int,signal_sparsity:int,Psi:np.ndarray,constant_matrix:np.ndarray,design_threshold:list,sensor_variance:float,epsilon:float,h_prev:np.ndarray,weights:np.ndarray,n_it:int,locations_monitored:list=[],locations_unmonitored:list=[])->list:
        """
        Rusu's wrapper algorithm implemented using the cvxopt library
        Args:
            Psi (np.ndarray): low-rank basis with (n_rows) locations and (n_cols) vectors/dimension
            
            epsilon (float): IRL1 weights update constant
            h_prev (np.ndarray): network locations initialization
            weights (np.ndarray): IRL1 weights initialization
            n_it (int): IRL1 max iterations
            locations_monitored (list, optional): initialization of set of monitored lcoations. Defaults to [].
            locations_unmonitored (list, optional): initialization of set of unmonitored locaitons. Defaults to [].

        Returns:
            locations (list): indices of monitored and unmonitored locations [S,Sc]
        """
        # iterative method
        epsilon_zero = epsilon
        primal_start = {'x':[],'sl':[],'ss':[]}
        it = 0
        time_init = time.time()
        optimization_problem = OptimizationProblemCVXOPT()
        print(f'Length locations: {len(locations_monitored) + len(locations_unmonitored)}')
        while len(locations_monitored) + len(locations_unmonitored) != n:
            # solve sensor placement with constraints
            optimization_problem.solve(n=n,Psi=Psi,signal_sparsity=signal_sparsity,weights=weights,
                                       design_threshold=design_threshold,sensor_variance=sensor_variance,
                                       constant_matrix=constant_matrix)
            sys.stdout.flush()
            if optimization_problem.problem['status'] == 'optimal':
                # get solution
                primal_start['x'] = optimization_problem.problem['x']
                primal_start['sl'] = optimization_problem.problem['sl']
                primal_start['ss'] = optimization_problem.problem['ss']
                # update sets
                new_monitored = [int(i[0]) for i in np.argwhere(optimization_problem.h >= 1-epsilon) if i[0] not in locations_monitored]
                new_unmonitored = [int(i[0]) for i in np.argwhere(optimization_problem.h <= epsilon_zero) if i[0] not in locations_unmonitored]
                locations_monitored += new_monitored
                locations_unmonitored += new_unmonitored
                # check convergence
                if np.linalg.norm(optimization_problem.h - h_prev)<=epsilon or it==n_it:
                    locations_monitored += [[int(i[0]) for i in np.argsort(optimization_problem.h,axis=0)[::-1] if i not in locations_monitored][0]]
                    it = 0        
                h_prev = optimization_problem.h
                weights = 1/(h_prev + epsilon)
                it +=1
                print(f'Iteration results\n ->Primal objective: {optimization_problem.problem["primal objective"]:.6f}\n ->{len(locations_monitored) + len(locations_unmonitored)} locations assigned\n ->{len(locations_monitored)} monitored locations\n ->{len(locations_unmonitored)} unmonitored locations\n')
                sys.stdout.flush()
            else:
                # solver fails at iteration
                locations_monitored = locations_monitored[:-len(new_monitored)]
                locations_unmonitored = locations_unmonitored[:-len(new_unmonitored)]
                it+=1


        time_end = time.time()
        locations = [locations_monitored,locations_unmonitored]
        print('-'*50)
        print(f'IRL1 algorithm finished in {time_end-time_init:.2f}s.')
        print('-'*50+'\n')
        sys.stdout.flush()
        return locations,time_end - time_init

# generate snapshots matrix
class NormalStandardSampling():
    def sample(self,seed,n,n_samples):
        rng = np.random.default_rng(seed=seed)
        snapshots_matrix = rng.normal(loc=0.0,scale=1.0,size=(n,n_samples))
        return snapshots_matrix
class SeasonalDataSampling():
    def generate_covariance_matrix(self,seed,n_clusters=3):
        rng = np.random.default_rng(seed=seed)
        cluster_centers = np.array([[-1, -1], [0.7, 0.3], [1, 1]])
        # 50%, 18%, 32%
        # Generate points around the cluster centers
        """
        points_per_cluster = [10, 4, 6]  # Assuming 20 total nodes
        points_per_cluster = [25, 9, 16]  # Assuming 50 total nodes
        points_per_cluster = [20, 7, 13]  # Assuming 40 total nodes
        points_per_cluster = [30, 11, 19]  # Assuming 60 total nodes
        points_per_cluster = [35, 13, 22]  # Assuming 70 total nodes
        points_per_cluster = [40, 14, 26]  # Assuming 70 total nodes
        """
        points_per_cluster = [len(i) for i in np.array_split(np.arange(n),n_clusters)]

        positions = []
        for i, center in enumerate(cluster_centers):
            noise = rng.random((points_per_cluster[i], 1))  # No noise added
            points = center + 0.1*noise
            positions.append(points)

        positions = np.concatenate(positions)
        distances = squareform(pdist(positions))
        distances = np.exp(-distances ** 2)
        # Define the adjacency matrix based on a threshold
        threshold = 0.4
        adjacency = np.where(distances < threshold, 0, distances)
        degree = np.diag(np.sum(adjacency, axis=1))
        laplacian = degree - adjacency
        covariance_matrix = np.linalg.pinv(laplacian)
        return covariance_matrix


    def sample(self,seed,n=100,n_years=2,n_measurements_per_day=24):
        n_measurements_total = n_years * 365 * n_measurements_per_day
        # random mean and covariance
        mean_M0 = np.random.uniform(low=55, high=65, size=n)
        mean_A0 = np.random.uniform(low=10, high=35, size=n)
        mean_phi = np.zeros(n)
        # random samples
        covariance_matrix = self.generate_covariance_matrix(seed=seed)
        M0_samples = np.random.multivariate_normal(mean_M0, covariance_matrix, n_measurements_total)
        A0_samples = np.random.multivariate_normal(mean_A0, covariance_matrix, n_measurements_total)
        phi_samples = np.random.multivariate_normal(mean_phi, covariance_matrix, n_measurements_total)

        # Generate time series data
        snapshots_matrix = np.zeros((n, n_measurements_total))
        D = 0
        for i in range(n):
            for j in range(n_measurements_total):
                t = j % n_measurements_per_day  # Get the time within a day (0-23)
                D += 1 if t == 0 else 0  # Increase D by 1 when a new day starts
                M0 = M0_samples[j, i] * (1 + 0.8* np.sin(2 * np.pi * D / 365))
                A0 = A0_samples[j,i] * (1 + 0.7 * np.sin(2 * np.pi * D / 365))
                phi = phi_samples[j, i]
                noise = 0.5 * np.random.normal(0, 1)  # Generate white noise
                snapshots_matrix[i, j] = M0 + A0 * np.sin((2 * np.pi * t / n_measurements_per_day) + 0.1*phi) + noise
        return snapshots_matrix


class snapshotsMatrix():
    def __init__(self,sampler):
        self._sampler = sampler
    def create_snapshots_matrix(self,**kwargs):
        snapshots_matrix = self._sampler.sample(**kwargs)
        return snapshots_matrix

if __name__=='__main__':
    abs_path = os.path.dirname(os.path.realpath(__file__))
    files_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/files/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/test/'
    n_range = [10,50,100]
    seed_range = np.arange(0,5)
    energy_threshold = 0.9
    design_factor = 5.0
    epsilon = 5e-2
    n_it = 20
    sensor_variance = 1.
    compute_times = True
    if compute_times:
        exec_times_size = {el:0 for el in n_range}
        for n in n_range:
            exec_times = []
            for seed in seed_range:
                snapshots_matrix_generator = snapshotsMatrix(SeasonalDataSampling())
                snapshots_matrix = snapshots_matrix_generator.create_snapshots_matrix(seed=seed,n=n,n_years=1,n_measurements_per_day=24)
                snapshots_matrix_centered = snapshots_matrix - snapshots_matrix.mean(axis=1)[:,None]
                # low-rank decomposition
                U,sing_vals,_ = np.linalg.svd(snapshots_matrix_centered,full_matrices=False)
                energy = np.cumsum(sing_vals)/np.sum(sing_vals)
                signal_sparsity = np.where(energy>=energy_threshold)[0][0] +1
                Psi = U[:,:signal_sparsity]
                coordinate_error_variance_fm = np.diag(Psi@Psi.T)
                wcev_fm = coordinate_error_variance_fm.max()

                design_threshold = design_factor*wcev_fm*np.ones(shape=n)
                h_prev = np.zeros(n)
                weights = 1/(h_prev+epsilon)

                irnet = RusuWrapperCVXOPT()
                locations,time_exec = irnet.design(Psi=Psi,n=n,signal_sparsity=signal_sparsity,
                                                design_threshold=design_threshold,sensor_variance=sensor_variance,
                                                epsilon=epsilon,h_prev=h_prev,weights=weights,n_it=n_it,
                                                locations_monitored=[],locations_unmonitored=[],constant_matrix=np.zeros(shape=(signal_sparsity,signal_sparsity)))
                exec_times.append(time_exec)
            exec_times_size[n] = exec_times
        
        # save computational time file
        fname = f'{results_path}executionTimes_randomMatrices_N{[i for i in n_range]}.pkl'
        with open(fname,'wb') as f:
            pickle.dump(exec_times_size,f,protocol=pickle.HIGHEST_PROTOCOL)
        print(f'saved in {fname}')
    else:
        n = 1000
        fname = f'{results_path}executionTimes_randomMatrices_N{n_range[0]}to{n}.pkl'
        with open(fname,'rb') as f:
            exec_times_size = pickle.load(f)
        



