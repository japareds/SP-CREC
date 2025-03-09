#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 17:21:04 2023

@author: jparedes
"""
import os
import time
import pandas as pd
import geopy.distance
from sklearn.model_selection import train_test_split
from abc import ABC,abstractmethod
import numpy as np
import math
import sys
import warnings
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from geopandas import GeoDataFrame

import sensor_placement as sp


""" Obtain signal sparsity and reconstruct signal at different temporal regimes"""

# perturbate measurements
def add_noise_signal(X:pd.DataFrame,seed:int=92,var:float=1.)->pd.DataFrame:
    """
    Add noise to measurements dataset. The noise ~N(0,var).
    The noise is the same for all sensors during all the time.

    Args:
        X (pd.DataFrame): dataset with measurements
        seed (int): random number generator seed
        var (float): noise variance

    Returns:
        pd.DataFrame: _description_
    """
    rng = np.random.default_rng(seed=seed)
    noise = rng.normal(loc=0.0,scale=var,size=X.shape)
    X_noisy = X + noise
    #X_noisy[X_noisy<0] = 0.
    return X_noisy

# ROI classes
class roi_generator(ABC):
    @abstractmethod
    def generate_rois(self,**kwargs):
        raise NotImplementedError
    
class RandomRoi(roi_generator):
    """ Regions of Interest randomly generated from rng seed"""
    def generate_rois(self,**kwargs)->dict:
        seed = kwargs['seed']
        n = kwargs['n']
        n_regions = kwargs['n_regions']
        rng = np.random.default_rng(seed=seed)    
        indices = np.arange(0,n,1)
        indices_perm = rng.permutation(indices)
        roi_idx = {el:[] for el in np.arange(n_regions)}
        indices_split = np.array_split(indices_perm,n_regions)
        for i in np.arange(n_regions):
            roi_idx[i] = indices_split[i]
        return roi_idx
    
class SubSplitRandomRoi(roi_generator):
    """
    Regions of Interest randomly generated. 
    The indices are randomly generated and then some of them are splitted into new sub regions.
    """
    def generate_rois(self,**kwargs):
        seed = kwargs['seed']
        n = kwargs['n']
        n_regions_original = kwargs['n_regions_original']
        rois_split = kwargs['rois_split']
        n_regions_subsplit = kwargs['n_regions_subsplit']
        seed_subsplit = kwargs['seed_subsplit']
        rng = np.random.default_rng(seed=seed)
        indices = np.arange(0,n,1)
        # first split. Original ROIs
        indices_perm = rng.permutation(indices)
        roi_idx = {el:[] for el in np.arange(n_regions_original)}
        indices_split = np.array_split(indices_perm,n_regions_original)
        for i in np.arange(n_regions_original):
            roi_idx[i] = indices_split[i]
        # second split. Maintain some ROIs and split others
        new_roi_idx = {}
        rng_subsplit = np.random.default_rng(seed=seed_subsplit)
        for i in roi_idx:
            if i in rois_split:
                indices_roi = roi_idx[i]
                indices_roi_perm = rng_subsplit.permutation(indices_roi)
                indices_roi_split = np.array_split(indices_roi_perm,n_regions_subsplit)
                new_dict = {}
                for j in np.arange(n_regions_subsplit):
                    new_dict[float(f'{i}.{j+1}')] = indices_roi_split[j]
                new_roi_idx.update(new_dict)
            else:
                new_roi_idx[i] = roi_idx[i]
            
        return new_roi_idx
            
    
class VarianceRoi(roi_generator):
    def generate_rois(self,**kwargs)->dict:
        coordinate_error_variance_fullymonitored = kwargs['coordinate_error_variance_fullymonitored']
        variance_thresholds = kwargs['variance_thresholds']
        n_regions = kwargs['n_regions']
        print(f'Determining indices that belong to each ROI. {n_regions} regions with thresholds: {variance_thresholds}')
        if type(variance_thresholds) is not list:
            variance_thresholds = [variance_thresholds]
        if len(variance_thresholds) != n_regions:
            raise ValueError(f'Number of variance thresholds: {variance_thresholds} mismatch specified number of regions: {n_regions}')
        roi_idx = {el:[] for el in variance_thresholds}
        for i in range(len(variance_thresholds[:-1])):
            print(f'Variance threshold between {variance_thresholds[i]} and {variance_thresholds[i+1]}')
            stations = [j for j in coordinate_error_variance_fullymonitored[np.logical_and(coordinate_error_variance_fullymonitored>=variance_thresholds[i],coordinate_error_variance_fullymonitored<variance_thresholds[i+1])]]
            print(f'{len(stations)} stations')
            idx_stations = np.where(np.isin(coordinate_error_variance_fullymonitored,stations))[0]
            roi_idx[variance_thresholds[i]] = idx_stations
        stations = [j for j in coordinate_error_variance_fullymonitored[coordinate_error_variance_fullymonitored>=variance_thresholds[-1]]]
        print(f'{len(stations)} stations with a distance larger than {variance_thresholds[-1]}')
        idx_stations = np.where(np.isin(coordinate_error_variance_fullymonitored,stations))[0]
        roi_idx[variance_thresholds[-1]] = idx_stations
        return roi_idx
    
class DistanceRoi(roi_generator):
    def generate_rois(self,**kwargs)->dict:
        """
        Generates Regions of Interest (ROIs) based on distance from certain station

        Args:        
            distances (pd.Series): distance of each location from origin station
            distance_thresholds (list): thresholds for each ROI
            n_regions (int): number of ROIs

        Raises:
            ValueError: Check if number of specified distance thresholds matches number of ROIs

        Returns:
            dict: Indices of each ROI. Key specifies the distance threshold
        """
        distances = kwargs['distances']
        distance_thresholds = kwargs['distance_thresholds']
        n_regions = kwargs['n_regions']
        print(f'Determining indices that belong to each ROI. {n_regions} regions with thresholds: {distance_thresholds}')
        if type(distance_thresholds) is not list:
            distance_thresholds = [distance_thresholds]
        if len(distance_thresholds) != n_regions:
            raise ValueError(f'Number of distance thresholds: {distance_thresholds} mismatch specified number of regions: {n_regions}')
        roi_idx = {el:[] for el in distance_thresholds}
        #distance_thresholds = np.insert(distance_thresholds,0,0)
        for i in range(len(distance_thresholds[:-1])):
            print(f'Distance threshold between {distance_thresholds[i]} and {distance_thresholds[i+1]}')
            stations = [j for j in distances[np.logical_and(distances>=distance_thresholds[i],distances<distance_thresholds[i+1])].index]
            print(f'Stations ({len(stations)}): {stations}')
            idx_stations = np.where(np.isin(distances.index,stations))[0]
            roi_idx[distance_thresholds[i]] = idx_stations
        stations = [j for j in distances[distances>=distance_thresholds[-1]].index]
        print(f'Stations with a distance larger than {distance_thresholds[-1]} ({len(stations)}): {stations}')
        idx_stations = np.where(np.isin(distances.index,stations))[0]
        roi_idx[distance_thresholds[-1]] = idx_stations
        
        return roi_idx


class ROI():
    """
    Region of interest (ROI) class. Select a generator from different roigenerator classes.
    Use as:
        roi = ROI(generator())
        roi.deine_ROIs(**kwargs)
    """
    def __init__(self,generator):
        self._generator = generator
    def define_rois(self,**kwargs)->dict:
        self.roi_idx = self._generator.generate_rois(**kwargs)

# file writer classes
class FileWriter(ABC):
    @abstractmethod
    def save(self,**kwargs):
        raise NotImplementedError

class WriteRandomFile(FileWriter):
    def save(self,results_path,locations,**kwargs):
        n = kwargs['n']
        signal_sparsity = kwargs['signal_sparsity']
        variance_threshold_ratio = kwargs['variance_threshold_ratio']
        n_locations_monitored = kwargs['n_locations_monitored']
        random_seed = kwargs['random_seed']
        
        fname = f'{results_path}SensorsLocations_N{n}_S{signal_sparsity}_VarThreshold{variance_threshold_ratio}_nSensors{n_locations_monitored}_randomSeed{random_seed}.pkl'
        with open(fname,'wb') as f:
            pickle.dump(locations,f,protocol=pickle.HIGHEST_PROTOCOL)
        print(f'File saved in {fname}')

class WriteSplitRandomFile(FileWriter):
    def save(self,results_path,locations,**kwargs):
        n = kwargs['n']
        signal_sparsity = kwargs['signal_sparsity']
        variance_threshold_ratio = kwargs['variance_threshold_ratio']
        n_locations_monitored = kwargs['n_locations_monitored']
        random_seed = kwargs['seed']
        seed_subsplit = kwargs['seed_subsplit']
        rois_split = kwargs['rois_split']
        
        fname = f'{results_path}SensorsLocations_N{n}_S{signal_sparsity}_VarThreshold{variance_threshold_ratio}_nSensors{n_locations_monitored}_randomSeed{random_seed}_split{rois_split}_subsplitSeed{seed_subsplit}.pkl'
        with open(fname,'wb') as f:
            pickle.dump(locations,f,protocol=pickle.HIGHEST_PROTOCOL)
        print(f'File saved in {fname}')

class SaveLocations():
    def __init__(self,writer):
        self._writer = writer
    def save_locations(self,results_path,locations,**kwargs):
        self._writer.save(results_path,locations,**kwargs)

# file reader class
class FileReader(ABC):
    @abstractmethod
    def load(self,**kwargs):
        raise NotImplementedError

class ReadRandomFile(FileReader):
    def load(self,file_path,**kwargs):
        n = kwargs['n']
        signal_sparsity = kwargs['signal_sparsity']
        variance_threshold_ratio = kwargs['signal_threshold_ratio']
        n_sensors = kwargs['n_sensors']
        random_seed = kwargs['random_seed']
        fname = f'{file_path}SensorsLocations_N{n}_S{signal_sparsity}_VarThreshold{variance_threshold_ratio}_nSensors{n_sensors}_randomSeed{random_seed}.pkl'
        with open(fname,'rb') as f:
            locations_monitored = np.sort(pickle.load(f))
        return locations_monitored
class ReadSplitRandomFile(FileReader):
    def load(self,file_path,**kwargs):
        n = kwargs['n']
        signal_sparsity = kwargs['signal_sparsity']
        variance_threshold_ratio = kwargs['variance_threshold_ratio']
        n_sensors = kwargs['n_sensors']
        random_seed = kwargs['random_seed']
        seed_subsplit = kwargs['seed_subsplit']
        rois_split = kwargs['rois_split']

        fname = f'{file_path}SensorsLocations_N{n}_S{signal_sparsity}_VarThreshold{variance_threshold_ratio}_nSensors{n_sensors}_randomSeed{random_seed}_split{rois_split}_subsplitSeed{seed_subsplit}.pkl'
        with open(fname,'rb') as f:
            locations_monitored = np.sort(pickle.load(f))
        return locations_monitored
    
class ReadRandomFileBoyd(FileReader):
    def load(self,file_path,**kwargs):
        n = kwargs['n']
        signal_sparsity = kwargs['signal_sparsity']
        variance_threshold_ratio = kwargs['variance_threshold_ratio']
        random_seed = kwargs['random_seed']
        n_sensors_Dopt = kwargs['n_sensors_Dopt']
        fname = f'{file_path}SensorsLocations_Boyd_N{n}_S{signal_sparsity}_VarThreshold{variance_threshold_ratio}_nSensors{n_sensors_Dopt}_randomSeed{random_seed}.pkl'
        with open(fname,'rb') as f:
            locations_monitored = np.sort(pickle.load(f))
        return locations_monitored
    
class ReadSplitRandomFileBoyd(FileReader):
    def load(self,file_path,**kwargs):
        n = kwargs['n']
        signal_sparsity = kwargs['signal_sparsity']
        variance_threshold_ratio = kwargs['variance_threshold_ratio']
        n_sensors_Dopt = kwargs['n_sensors_Dopt']
        random_seed = kwargs['random_seed']
        seed_subsplit = kwargs['seed_subsplit']
        rois_split = kwargs['rois_split']
        fname = f'{file_path}SensorsLocations_Boyd_N{n}_S{signal_sparsity}_VarThreshold{variance_threshold_ratio}_nSensors{n_sensors_Dopt}_randomSeed{random_seed}_split{rois_split}_subsplitSeed{seed_subsplit}.pkl'
        try:
            with open(fname,'rb') as f:
                locations_monitored = np.sort(pickle.load(f))
            print(f'Loaded file {fname}')
        except:
            warnings.warn(f'No file {fname}')
            return 
        return locations_monitored
    
class ReadLocations():
    def __init__(self,reader):
        self._reader = reader
    def load_locations(self,file_path,**kwargs):
        locations_monitored = self._reader.load(file_path,**kwargs)
        return locations_monitored


#%% signal reconstruction functions
def singular_value_hard_threshold(snapshots_matrix:np.ndarray,sing_vals:np.array,noise:float=-1)->float:
    """
    Compute singular value hard threshold from Gavish-Donoho approximation

    Args:
        snapshots_matrix (np.ndarray): snapshots matrix used for computing SVD
        sing_vals (np.array): corresponding array of singular values
        noise (float,optional): noise () deviation from signal

    Returns:
        float: cut-off index
    """
    beta = snapshots_matrix.shape[0]/snapshots_matrix.shape[1]
    if noise == -1:#unknown noise
        c1,c2,c3,c4 = 0.56,0.95,1.82,1.43
        omega = c1*beta**3 - c2*beta**2 + c3*beta + c4
        sing_val_threshold = omega*np.median(sing_vals)
        
    else:#known noise
        t1 = 2*(beta+1)
        t2 = (8*beta) / ( beta + 1 + np.sqrt((beta**2 + 14*beta + 1)) )
        lambda_beta = np.sqrt(t1+t2)
        sing_val_threshold = lambda_beta*noise*np.sqrt(max(snapshots_matrix.shape))
    
    sparsity_gd = np.max(np.where(sing_vals>sing_val_threshold)) + 1
    return sparsity_gd
def signal_reconstruction_svd(U:np.ndarray,snapshots_matrix:np.ndarray,s_range:np.ndarray) -> pd.DataFrame:
    """
    Decompose signal keeping s-first singular vectors using training set data
    and reconstruct validation set.

    Args:
        U (numpy array): left singular vectors matrix
        snapshots_matrix (numpy array): snaphots matrix data.
        s_range (numpy array): list of sparsity values to test

    Returns:
        rmse_sparsity: dataframe containing reconstruction errors at different times for each sparsity threshold in the range
    """
    print(f'Determining signal sparsity by decomposing training set and reconstructing validation set.\nRange of sparsity levels: {s_range}')
    mse_sparsity = pd.DataFrame()
    error_variance_sparsity = pd.DataFrame()
    normalized_frobenius_norm = pd.DataFrame()
    for s in s_range:
        # projection
        Psi = U[:,:s]
        #snapshots_matrix_pred_svd = (Psi@Psi.T@snapshots_matrix_centered) + snapshots_matrix_train.mean(axis=1)[:,None]
        snapshots_matrix_pred_svd = Psi@Psi.T@snapshots_matrix
        
        #RMSE across different signal measurements
        # estimated covariance
        error = snapshots_matrix - snapshots_matrix_pred_svd
        error_variance = error.var(axis=1,ddof=0)# estimated coordiante error variance
        mse = pd.DataFrame((error**2).mean(axis=0),columns=[s])
        coordinate_error_variance = pd.DataFrame(error_variance,columns=[s])

        normalized_frobenius_norm = pd.concat((normalized_frobenius_norm,pd.DataFrame([np.linalg.norm(error,ord='fro')/np.linalg.norm(snapshots_matrix,ord='fro')],columns=[s])),axis=1)
        mse_sparsity = pd.concat((mse_sparsity,mse),axis=1)
        error_variance_sparsity = pd.concat((error_variance_sparsity,coordinate_error_variance),axis=1)

    return mse_sparsity,error_variance_sparsity,normalized_frobenius_norm

def signal_reconstruction_regression(Psi:np.ndarray,locations_measured:np.ndarray,X_test:pd.DataFrame,X_test_measurements:pd.DataFrame=[],snapshots_matrix_train:np.ndarray=[],snapshots_matrix_test_centered:np.ndarray=[],projected_signal:bool=False,sample_covariance:bool=True)->pd.DataFrame:
    """
    Signal reconstyruction from reduced basis measurement.
    The basis Psi and the measurements are sampled at indices in locations_measured.
    Compute reconstruction error


    Args:
        Psi (np.ndarray): low-rank basis
        locations_measured (np.ndarray): indices of locations measured
        X_test (pd.DataFrame): testing dataset which is measured and used for error estimation
        X_test_measurements (pd.DataFrame): testing dataset measurements projected onto subspace spanned by Psi
        snapshots_matrix_train (np.ndarray): training set snapshots matrix used for computing average
        snapshots_matrix_val_centered (np.ndarray): testing set centered snapshots matrix used for signal reconstruction
        

    Returns:
        rmse (pd.DataFrame): mean reconstruction error between validation data set and reconstructed data
        error_max (pd.DataFrame): max reconstruction error when comparing validation data with reconstructed data
    """
    # basis measurement
    n_sensors_reconstruction = len(locations_measured)
    C = np.identity(Psi.shape[0])[locations_measured]
    Psi_measured = C@Psi
    # regression
    if projected_signal:
        beta_hat = np.linalg.pinv(Psi_measured)@X_test_measurements.iloc[:,locations_measured].T
        snapshots_matrix_predicted = Psi@beta_hat
    else:
        beta_hat = np.linalg.pinv(Psi_measured)@snapshots_matrix_test_centered[locations_measured,:]
        snapshots_matrix_predicted_centered = Psi@beta_hat
        snapshots_matrix_predicted = snapshots_matrix_predicted_centered + snapshots_matrix_train.mean(axis=1)[:,None]
    # compute prediction
    X_pred = pd.DataFrame(snapshots_matrix_predicted.T)
    X_pred.columns = X_test.columns
    X_pred.index = X_test.index
    # compute error metrics
    error = X_test - X_pred
    rmse = pd.DataFrame(np.sqrt(((error)**2).mean(axis=1)),columns=[n_sensors_reconstruction],index=X_test.index)
    error_variance = error.var(axis=0,ddof=0)
    """
    error_max = pd.DataFrame(np.abs(error).max(axis=1),columns=[n_sensors_reconstruction],index=X_test.index)
    error_var = np.zeros(shape = error.shape)
    for i in range(error.shape[0]):
        error_var[i,:] = np.diag(error.iloc[i,:].to_numpy()[:,None]@error.iloc[i,:].to_numpy()[:,None].T)
    error_var = pd.DataFrame(error_var,index=X_test.index,columns=X_test.columns)
    """
    return rmse, error_variance

#%%
def networkPlanning_iterative(sensor_placement:sp.SensorPlacement,N:int,Psi:np.ndarray,deployed_network_variance_threshold:float,epsilon:float,h_prev:np.ndarray,weights:np.ndarray,n_it:int,p_min:int=0,locations_monitored:list=[],locations_unmonitored:list=[])->list:
    """
    IRL1 network planning algorithm
    Args:
        sensor_placement (sp.SensorPlacement): sensor placement object containing network information
        N (int): total number of network locations
        deployed_network_variance_threshold (float): error variance ratio for constraining per-coordinate error variance
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
    it = 0
    time_init = time.time()
    new_monitored = []
    new_unmonitored = []
    while len(locations_monitored) + len(locations_unmonitored) != N:
        # solve sensor placement with constraints
        
        sensor_placement.initialize_problem(Psi,rho=deployed_network_variance_threshold,
                                            w=weights,p_min=p_min,locations_monitored=locations_monitored,locations_unmonitored=locations_unmonitored)
        sensor_placement.solve()
        print(f'Problem status: {sensor_placement.problem.status}')
        if sensor_placement.problem.status == 'optimal':
            # update sets with new monitored locations
            new_monitored = [i[0] for i in np.argwhere(sensor_placement.h.value >= 1-epsilon) if i[0] not in locations_monitored]
            new_unmonitored = [i[0] for i in np.argwhere(sensor_placement.h.value <= epsilon) if i[0] not in locations_unmonitored]

            if len(new_monitored) != 0:
                locations_monitored += new_monitored
            else:
                print('No new monitored locations')
            if len(new_unmonitored)!=0:
                locations_unmonitored += new_unmonitored
            else:
                print('No new unmonitored locations')
            # check convergence
            if np.linalg.norm(sensor_placement.h.value - h_prev)<=epsilon or it==n_it:
                locations_monitored += [[i for i in np.argsort(sensor_placement.h.value)[::-1] if i not in locations_monitored][0]]
                it = 0
            h_prev = sensor_placement.h.value
            weights_old = weights.copy()
            weights = 1/(h_prev + epsilon)
            it +=1
        else:
            # solver fails at iteration
            #locations_monitored = locations_monitored[:-len(new_monitored)]
            if len(new_unmonitored) != 0:
                locations_unmonitored = locations_unmonitored[:-len(new_unmonitored)]
                weights = weights_old
            it+=1

        print(f'{len(locations_monitored)} Locations monitored: {locations_monitored}\n{len(locations_unmonitored)} Locations unmonitored: {locations_unmonitored}\n')
    time_end = time.time()
    locations = [locations_monitored,locations_unmonitored]
    print(f'IRL1 algorithm finished in {time_end-time_init:.2f}s.')
    return locations
#%%
def sizing_and_placement(U:np.ndarray,signal_sparsity:list,sigma_noise:float,sigma_threshold:np.array,rois_thresholds:dict = {'urban':9,'rural':18},rois_source:str='random',n_urban:int=0,n_rural:int=0,seed_random:int=0,epsilon_range:list = [20e-2,10e-2,9e-2,8e-2,7e-2,6e-2,5e-2,4e-2,3e-2,2e-2,1e-2]):
    """
    SP-CREC algorithm for given or every sparsity level for a certain basis Psi.
    Specify sigma_threshold for different ROIs and noise sigma value
    Iterates over different epsilon values that gives the minimum number of sensors.
    """

    print(f'\n\nSizing and Placement algorithm with Coordinate Reconstruction Error Constraints (SP-CREC)\n\n')
            
    n = U.shape[0]
    sigma_noise_list = sigma_noise*np.asarray(len(sigma_threshold)*[1])
    variance_ratio = (sigma_threshold**2)/(sigma_noise_list**2)#sigma_th^2 / sigma_noise^2
    n_regions = len(rois_thresholds)
    
    
    """
    Solve SPCREC sensor placement problem
    Algorithm parameters
        As epsilon increases less stations are assigned to be monitored, i.e., it is easier to assign unmonitored locations.
        However, large epsilon values  (epsilon=1e-1) can make the problem infeasible at next iteration as some locations are assigned to be unmonitored.
        epsilon=1e-1 tends to fail at multiple thresholds as threshold <1.1
    """
    if type(signal_sparsity) == int and signal_sparsity > 0:
        s_range = [signal_sparsity]
    elif (type(signal_sparsity)==list or type(signal_sparsity)==np.ndarray) and len(signal_sparsity) !=0:
        s_range = signal_sparsity
    else:
        s_range = np.arange(1,U.shape[1]+1)

    algorithm = 'NetworkPlanning_iterative'    
    n_it = 20
    epsilon_prev = epsilon_range[0]

    # solve for different basis
    for s in s_range:
        Psi = U[:,:s]
        print(f'\n\nSolving SP-CREC algorithm for basis of size: {Psi.shape}')
    
        # epsilon iterator
        for epsilon in epsilon_range:
            success = 0
            #if epsilon_prev<epsilon:
            #    continue
            print(f'Current SP-CREC epsilon value: {epsilon:.2f}')
            sensor_placement = sp.SensorPlacement(algorithm, n, s,
                                                    n_refst=n,n_lcs=0,n_unmonitored=0)
            h_prev = np.zeros(n)
            weights = 1/(h_prev+epsilon)
            locations_monitored = []
            locations_unmonitored = []
        
            locations = networkPlanning_iterative(sensor_placement,n,Psi,variance_ratio,
                                                    epsilon=epsilon,h_prev=h_prev,weights=weights,n_it=n_it,
                                                    locations_monitored=locations_monitored,locations_unmonitored=locations_unmonitored)
            
            """ 
            Compute the solution obtained for the coordinate error variance
            check if the solution meets the constraints
            """
            n_locations_monitored = len(locations[0])
            n_locations_unmonitored = len(locations[1])
            if n_locations_monitored == 0:
                print(f'No sensors deployed')
                continue
            elif n_locations_monitored < s:
                print(f'Number of monitored locations is lower than signal sparisty')
                continue
            sensor_placement.locations = [[],np.sort(locations[0]),np.sort(locations[1])]
            sensor_placement.C_matrix()
            C = sensor_placement.C[1]
            print(f'Shape of C matrix: {C.shape}')
            # evaluate if algorithm was successful (analitically)
            try:
                coordinate_error_variance = (sigma_noise**2)*np.diag(Psi@np.linalg.inv(Psi.T@C.T@C@Psi)@Psi.T)
            except:
                print('Error at computing coordinate error variance from monitored locations')
                continue
            
            if (coordinate_error_variance<sigma_threshold**2).sum() == n:
                    print(f'\nSuccess at computing coordinate error variance at every location with epsilon: {epsilon:.2f}')
                    print(f'Network design results:\n- Total number of potential locations: {n}\n- basis sparsity: {s}\n- Number of monitored locations: {n_locations_monitored}\n- Number of unmonitored locations: {n_locations_unmonitored}')
                    #epsilon_prev = epsilon
                    #success = 1
            else:
                    print('Some locations do not meet threshold constraints')
                    print(f'Try epsilon lower than current value ({epsilon:.2f})')

            # save results
            # if success        

            if rois_source == 'random':
                fname = f'{results_path}SPCREC_SensorsLocations_N[{n}]_S[{s}]_epsilon[{epsilon:.2f}]_nROIS[{n_regions}]_nUrban[{n_urban}]_nRural[{n_rural}]_RandomSeed[{seed_random}]_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}]_nSensors[{n_locations_monitored}].pkl'
            else:
                fname = f'{results_path}SPCREC_SensorsLocations_N[{n}]_S[{s}]_epsilon[{epsilon:.2f}]_nROIS[{n_regions}]_nUrban[{n_urban}]_nRural[{n_rural}]_RandomSeed[{seed_random}]_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}]_nSensors[{n_locations_monitored}].pkl'

            with open(fname,'wb') as f:
                pickle.dump(locations,f,protocol=pickle.HIGHEST_PROTOCOL)
            print(f'File saved in {fname}')            
            #break # leave epsilon iterator

#%%
def validate_spcrec(path_to_files:str,s_range:list,snapshots_matrix:np.ndarray,U:np.ndarray,sigma_noise:float,sigma_threshold:list,rois_thresholds:dict,n_urban:int,n_rural:int,rois_source:str='random',seed_random:int=0,seed_noise:int=0):
    if type(s_range)==list:
        s_range = np.array(s_range)
    n = U.shape[0]
    n_regions = len(rois_thresholds)
    epsilon_range = np.array([20e-2,10e-2,9e-2,8e-2,7e-2,6e-2,5e-2,4e-2,3e-2,2e-2,1e-2])
    n_locations_thresholded = pd.DataFrame(np.zeros((s_range.shape[0],epsilon_range.shape[0])),index=s_range,columns=epsilon_range)
    n_monitored_locations = pd.DataFrame(np.zeros((s_range.shape[0],epsilon_range.shape[0])),index=s_range,columns=epsilon_range)
    for s in s_range:
        print(f'Loading SP-CREC result files for sparsity level: {s}')
        n_locations_range = np.arange(s,n+1,1)
        # load SP-CREC solution
        
        for n_locations_monitored in n_locations_range:
            for epsilon in epsilon_range:
                locations_monitored_spcrec = []
            
                if rois_source == 'random':
                    fname = f'{path_to_files}SPCREC_SensorsLocations_N[{n}]_S[{s}]_epsilon[{epsilon:.2f}]_nROIS[{n_regions}]_nUrban[{n_urban}]_nRural[{n_rural}]_RandomSeed[{seed_random}]_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}]_nSensors[{n_locations_monitored}].pkl'
                else:
                    fname = f'{path_to_files}SPCREC_SensorsLocations_N[{n}]_S[{s}]_epsilon[{epsilon:.2f}]_nROIS[{n_regions}]_nUrban[{n_urban}]_nRural[{n_rural}]_RandomSeed[{seed_random}]_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}]_nSensors[{n_locations_monitored}].pkl'

                try:
                    with open(fname,'rb') as f:
                        locations_monitored_spcrec = np.sort(pickle.load(f)[0])
                    print(f'Loading epsilon: {epsilon:.2f}')
                except:
                    continue
            
                if len(locations_monitored_spcrec) == 0:
                    print(f'No SP-CREC solution file for sparsity level ({s})')
                    continue

                # add noise to snapshots matrix
                C_spcrec = np.identity(n)[locations_monitored_spcrec,:]
                
                rng = np.random.default_rng(seed=seed_noise)
                Z = rng.standard_normal(size=snapshots_matrix.shape)
                snapshots_matrix_noisy = snapshots_matrix + sigma_noise*Z
                snapshots_matrix_noisy_measured = C_spcrec@snapshots_matrix_noisy
                # compute coordinate error variance for snapshots matrix
                Psi = U[:,:s]
                beta_hat = np.linalg.pinv(C_spcrec@Psi)@snapshots_matrix_noisy_measured
                snapshots_matrix_predicted = Psi@beta_hat
                error = snapshots_matrix - snapshots_matrix_predicted
                error_var_spcrec = error.var(axis=1,ddof=0)
        
                
            
                if (error_var_spcrec<=sigma_threshold**2).sum() == n:
                    print(f'Coordinate error variance on snapshots ma/trix is below threshold values.\ns: {s}\nNumber of monitored locations: {n_locations_monitored}\n')
                else:
                    print(f'Failed at constraining coordinate error variance. Success at {(error_var_spcrec<=sigma_threshold**2).sum()} out of {n} locations')
                n_locations_thresholded.loc[s,epsilon] = (error_var_spcrec<=sigma_threshold**2).sum()
                n_monitored_locations.loc[s,epsilon] = len(locations_monitored_spcrec)
                
    return n_locations_thresholded,n_monitored_locations
#%% SP-CREC but the results are iteratively validated on a set
def sizing_and_placement_autovalidated(U:np.ndarray,signal_sparsity:int,epsilon:float,rois_thresholds:dict,n_urban:int,n_rural:int,sigma_noise:float,seed_noise:int,n_locations_monitored_init:int,snapshots_matrix:np.ndarray,sigma_threshold:np.array,rois_source:str,path_to_files:str,save_results:bool,**kwargs):

    """ 
    Load initial SP-CREC results
    """

    n = U.shape[0]
    n_regions = len(rois_thresholds)      

    if rois_source == 'random':
        seed_random = kwargs['seed_random']
        fname = f'{path_to_files}SPCREC_SensorsLocations_N[{n}]_S[{signal_sparsity}]_nROIS[{n_regions}]_nUrban[{n_urban}]_nRural[{n_rural}]_RandomSeed[{seed_random}]_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}]_nSensors[{n_locations_monitored_init}].pkl'
    else:
        fname = f'{path_to_files}SPCREC_SensorsLocations_N[{n}]_S[{signal_sparsity}]_nROIS[{n_regions}]_nUrban[{n_urban}]_nRural[{n_rural}]_RandomSeed[{seed_random}]_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}]_nSensors[{n_locations_monitored}].pkl'

    try:
        with open(fname,'rb') as f:
            locations_monitored_init = np.sort(pickle.load(f))
    except:
        print(f'No file {fname}')

    """ 
    compute coordinate error variance on validation snapshots matrix
    """
    # add noise to snapshots matrix    
    rng = np.random.default_rng(seed=seed_noise)
    Z = rng.standard_normal(size=snapshots_matrix.shape)
    snapshots_matrix_noisy = snapshots_matrix + sigma_noise*Z
    # measure noisy snapshots matrix
    C_spcrec = np.identity(n)[locations_monitored_init,:]
    snapshots_matrix_noisy_measured = C_spcrec@snapshots_matrix_noisy
    # compute coordinate error variance for snapshots matrix
    Psi = U[:,:signal_sparsity]
    beta_hat = np.linalg.pinv(C_spcrec@Psi)@snapshots_matrix_noisy_measured
    snapshots_matrix_predicted = Psi@beta_hat
    error = snapshots_matrix - snapshots_matrix_predicted
    error_var = error.var(axis=1,ddof=0)

    """
    Ensure that it is NOT impossible to meet the thresholds constraints with the number of modes
    """
    beta_hat_fullymonitored = np.linalg.pinv(Psi)@snapshots_matrix_noisy
    snapshots_matrix_predicted_fullymonitored = Psi@beta_hat_fullymonitored
    error_fullymonitored = snapshots_matrix - snapshots_matrix_predicted_fullymonitored
    error_var_fullymonitored = error_fullymonitored.var(axis=1,ddof=0)
    if (error_var_fullymonitored<=sigma_threshold**2).sum()!=n:
        print(f'It is not possible to meet the thresholds even with a fully monitored basis.\nSignal sparsity: {signal_sparsity}')
        return
    
    """
    Compare error variance with threshold
    """

    n_locations_thresholded = (error_var<=sigma_threshold**2).sum()
    if n_locations_thresholded == n:
        print(f'Coordinate error variance on validation snapshots matrix is already below every threshold value.\ns: {signal_sparsity}\nNumber of monitored locations: {n_locations_monitored_init}\n')
        fname = f'{results_path}SPCREC_SensorsLocations_Validated_N[{n}]_S[{signal_sparsity}]_nROIS[{n_regions}]_nUrban[{n_urban}]_nRural[{n_rural}]_RandomSeed[{seed_random}]_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}]_nSensors[{n_locations_monitored_init}].pkl'
        with open(fname,'wb') as f:
            pickle.dump(locations_monitored_init,f,protocol=pickle.HIGHEST_PROTOCOL)
        print(f'File saved in {fname}')            
        return
    
    else:
        print(f'Failed at constraining coordinate error variance at every location.\nSuccess at {n_locations_thresholded} out of {n} locations')

    """ 
    If SP-CREC failed to initially constraint every coordinate
    then comntinue the algorithm adding more locations
    """
    algorithm = 'NetworkPlanning_iterative'
    n_it = 20
    p_min = n_locations_monitored_init
    locations_monitored = [i for i in locations_monitored_init]
    while n_locations_thresholded !=n:
        sensor_placement = sp.SensorPlacement(algorithm, n, signal_sparsity,
                                              n_refst=n,n_lcs=0,n_unmonitored=0)
        locations_unmonitored = []
        h_prev = np.zeros(n)
        h_prev[locations_monitored] = 1
        weights = 1/(h_prev+epsilon)
        p_min += 1
        
        sigma_noise_list = sigma_noise*np.asarray(len(sigma_threshold)*[1])
        variance_ratio = (sigma_threshold**2)/(sigma_noise_list**2)#sigma_th^2 / sigma_noise^2

        locations = networkPlanning_iterative(sensor_placement,n,Psi,variance_ratio,
                                              epsilon=epsilon,h_prev=h_prev,weights=weights,n_it=n_it,p_min=p_min,
                                              locations_monitored=locations_monitored,locations_unmonitored=locations_unmonitored)

        sensor_placement.locations = [[],np.sort(locations[0]),np.sort(locations[1])]
        sensor_placement.C_matrix()
        C = sensor_placement.C[1]
        n_locations_monitored = len(locations[0])
        n_locations_unmonitored = len(locations[1])
        
        # compute coordinate error variance
        snapshots_matrix_noisy_measured = C@snapshots_matrix_noisy
        beta_hat = np.linalg.pinv(C@Psi)@snapshots_matrix_noisy_measured
        snapshots_matrix_predicted = Psi@beta_hat
        error = snapshots_matrix - snapshots_matrix_predicted
        error_var = error.var(axis=1,ddof=0)


        # check if solution meets the constraints
        
        try: # analytical computation
            coordinate_error_variance = (sigma_noise**2)*np.diag(Psi@np.linalg.inv(Psi.T@C.T@C@Psi)@Psi.T)
            print(f'Number of locations that meet analytical coordinate error variance constraint: {(coordinate_error_variance<=sigma_threshold**2).sum()}')
        except:
            print('Error at computing analytical coordinate error variance from monitored locations')

        n_locations_thresholded = (error_var<=sigma_threshold**2).sum()       
        
        if n_locations_thresholded == n:
            print(f'\nSuccess at thresholding coordinate error variance at every location.')
            print(f'Number of monitored locations: {n_locations_monitored}\nNumber of thresholded locations: {n_locations_thresholded}')
            success = 1
        else:
            print(f'Some locations do not meet threshold constraints.\nNumber of thresholded locations: {n_locations_thresholded}')
        

    # save results
    if success and save_results:        
        if rois_source == 'random':
            fname = f'{results_path}SPCREC_SensorsLocations_Validated_N[{n}]_S[{signal_sparsity}]_nROIS[{n_regions}]_nUrban[{n_urban}]_nRural[{n_rural}]_RandomSeed[{seed_random}]_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}]_nSensors[{n_locations_monitored}].pkl'
        else:
            fname = f'{results_path}SPCREC_SensorsLocations_Validated_N[{n}]_S[{signal_sparsity}]_nROIS[{n_regions}]_nUrban[{n_urban}]_nRural[{n_rural}]_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}]_nSensors[{n_locations_monitored}].pkl'

        with open(fname,'wb') as f:
            pickle.dump(locations,f,protocol=pickle.HIGHEST_PROTOCOL)
        print(f'File saved in {fname}')            
        #break # leave epsilon iterator

#%% Alternative sensor placement method
def trace_min_iterative(snapshots_matrix:np.ndarray,U:np.ndarray,signal_sparsity:int,sigma_threshold:np.array,rois_threshold:dict,n_urban:int,n_rural:int,seed_random:int,seed_noise:int,p_break:int=-1,algorithm:str='JB_trace'):
        """
        Solve sensor placement by minimizing the trace of variance-covariance matrix.
        The 
        """
        Psi = U[:,:signal_sparsity]
        n = Psi.shape[0]
        print(f'Low-rank decomposition. Basis shape: {Psi.shape}')        

        # add noise to snapshots matrix        
        rng = np.random.default_rng(seed=seed_noise)
        Z = rng.standard_normal(size=snapshots_matrix.shape)
        snapshots_matrix_noisy = snapshots_matrix + sigma_noise*Z
        
        """
        Solve sensor placement problem
            - Specify number of sensors to deploy
        """
        p_range = np.arange(signal_sparsity,n+1,1)
        for p in p_range:
            sensor_placement = sp.SensorPlacement(algorithm, n, signal_sparsity,
                                                  n_refst=p,n_lcs=0,n_unmonitored=n-p)
            sensor_placement.initialize_problem(Psi)
            sensor_placement.solve()
            sensor_placement.discretize_solution()
            locations = np.sort(sensor_placement.locations[1])
            n_locations_tracemin = len(locations)
            # measure noisy snapshots matrix
            C_tracemin = np.identity(n)[locations,:]
            snapshots_matrix_noisy_measured = C_tracemin@snapshots_matrix_noisy
            # compute coordinate error variance for snapshots matrix
            beta_hat = np.linalg.pinv(C_tracemin@Psi)@snapshots_matrix_noisy_measured
            snapshots_matrix_predicted = Psi@beta_hat
            error = snapshots_matrix - snapshots_matrix_predicted
            error_var = error.var(axis=1,ddof=0)
            # check how many locations are below threshold
            n_locations_thresholded_tracemin = (error_var<=sigma_threshold**2).sum()
            print(f'Number of locations below threshold: {n_locations_thresholded_tracemin}\nNumber of monitored locations: {n_locations_tracemin}')
            if p == p_break:
                break
            elif n_locations_thresholded_tracemin == n:
                break
            else:
                continue

        n_sensors = p
        # save results
        if rois_source == 'random':
            fname = f'{results_path}TraceMin_SensorsLocations_N[{n}]_S[{signal_sparsity}]_nROIS[{n_regions}]_nUrban[{n_urban}]_nRural[{n_rural}]_RandomSeed[{seed_random}]_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}]_nSensors[{n_sensors}].pkl'
        else:
            fname = f'{results_path}TraceMin_SensorsLocations_N[{n}]_S[{signal_sparsity}]_nROIS[{n_regions}]_nUrban[{n_urban}]_nRural[{n_rural}]_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}]_nSensors[{n_sensors}].pkl'

        with open(fname,'wb') as f:
            pickle.dump(locations,f,protocol=pickle.HIGHEST_PROTOCOL)
        print(f'File saved in {fname}')            
    

#%%
def trace_min_iterative_rois(snapshots_matrix:np.ndarray,snapshots_matrix_val:np.ndarray,cumulative_energy_basis:float,rois_source:str='random',sigma_noise:float = 6.75,rois_thresholds:dict = {'urban':9,'suburban':9,'rural':18},n_urban:int=0,n_rural:int=0,seed_random:int=0,start_larger_threshold:bool=True,force_n:int=-1,algorithm:str='JB_trace'):
    
    """
    Iterative sensor placement over multiple Regions of Interest (ROIs).
    Args:
        roi_idx (dict): Dictionary containing indices of locations asociated with each ROI. The keys correspond to thresholds for defining the ROI
        roi_threshold (list): ROI threshold for defining each ROI
        variance_threshold_ratio (list): percentage of worsening of coordinate error variance with respect to fully monitored network
        snapshots_matrix_train_centered (np.ndarray): snapshots matrix of measurements
        start_larger_threshold (bool): reverse ROI order
        force_n (int): stop algorithm when number of sensors reaches specified value.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """    

    """ 
    Set Regions of Interest (ROIs) thresholds
        The rows of the snapshots matrix are previously sorted based on ROIs.
        There are two olptins for defining ROIS:
            1. The ROIs are obtained from a file loaded within the dataset class
            2. Randomly assigned form specified number of urban and rural
    """

    # detect what indices belong to each ROI
    roi_idx = {el:[] for el in rois_thresholds.values()}
    for i in roi_idx:
        roi_idx[i] = np.array([np.array(j) for j in np.arange(0,n,1) if sigma_threshold[j] == i])

    
    """ Iterate over ROIs. The solutions of previous ROIs are used onto the next union of ROIs"""
    locations_monitored = []
    locations_unmonitored = []
    locations_monitored_roi = []
    locations_unmonitored_roi = []       

    iterator = range(len(rois_thresholds))
    indices_rois = [np.array(i) for i in roi_idx.values()]
    thresholds_rois = [i for i in rois_thresholds.values()]
    for i in iterator: # ROI iteration
        if start_larger_threshold:
            indices = np.sort(np.concatenate([roi_idx[j] for j in thresholds_rois[::-1][:i+1]]))
        else:
            indices = np.sort(np.concatenate([roi_idx[j] for j in thresholds_rois[:i+1]]))
        snapshots_matrix_roi = snapshots_matrix[indices,:]
        U_roi,sing_vals_roi,Vt_roi = np.linalg.svd(snapshots_matrix_roi,full_matrices=False)
        energy_roi = np.cumsum(sing_vals_roi)/np.sum(sing_vals_roi)
        signal_sparsity_roi = np.argmin(np.abs(cumulative_energy_basis - energy_roi))+1
        Psi_roi = U_roi[:,:signal_sparsity_roi]
        n_roi = Psi_roi.shape[0]
        print(f'Current ROI has {n_roi} potential locations')
        # carry monitored locations from previous ROI step
        if len(locations_monitored)!=0:
            locations_monitored_roi = np.where(np.isin(indices,locations_monitored))[0]
        else:
            locations_monitored_roi = []
        sigma_threshold_roi = sigma_threshold[indices]
        
        # number of sensors per ROI iteration
        for n_sensors_roi in np.arange(signal_sparsity_roi,n_roi+1,1):
            
            if n_sensors_roi > n_roi:
                raise ValueError(f'Number of deployed sensors in ROI ({n_sensors_roi}) is larger than the number of potential locations in ROI ({n_roi}).')
            elif n_sensors_roi < signal_sparsity_roi:
                raise ValueError(f'Number of deployed sensors in ROI ({n_sensors_roi}) is lower than signal sparsity in ROI ({signal_sparsity_roi})')
            print(f'Signal sparsity of sub-signal in ROI: {signal_sparsity_roi}\nNumber of locations in ROI: {n_sensors_roi}')
            # initialize algorithm
            sensor_placement = sp.SensorPlacement(algorithm, n_roi, signal_sparsity_roi,
                                                    n_refst=n_sensors_roi,n_lcs=0,n_unmonitored=n_roi-n_sensors_roi)
            sensor_placement.initialize_problem(Psi_roi,locations_monitored=locations_monitored_roi)
            sensor_placement.solve()
            sensor_placement.discretize_solution()
            locations = sensor_placement.locations[1]
            sensor_placement.C_matrix()
            C = sensor_placement.C[1]
            
            # Add noise and measure snapshots matrix at locations. Use a validation matrix
            snapshots_matrix_roi_val = snapshots_matrix_val[indices,:]
            rng = np.random.default_rng(seed=seed_noise)
            Z = rng.standard_normal(size=snapshots_matrix_roi_val.shape)
            snapshots_matrix_roi_val_noisy = snapshots_matrix_roi_val + sigma_noise*Z
            snapshots_matrix_roi_val_noisy_measured = C@snapshots_matrix_roi_val_noisy
            # compute error variance
            beta_hat = np.linalg.pinv(C@Psi_roi)@snapshots_matrix_roi_val_noisy_measured
            snapshots_matrix_roi_predicted = Psi_roi@beta_hat
            error_roi = snapshots_matrix_roi_val - snapshots_matrix_roi_predicted
            error_variance_roi = error_roi.var(axis=1,ddof=0)
            # compare error variance with roi threshold
            n_success_roi = (error_variance_roi<sigma_threshold_roi**2).sum()
            if n_sensors_roi != force_n:
                if n_success_roi != n_roi and len(locations)!= n_roi:
                    continue
                elif n_success_roi == n_roi or len(locations)==n_roi:
                    locations_monitored_roi = locations
                    locations_monitored = indices[locations]
                    break
            else:
                locations_monitored_roi = locations
                locations_monitored = indices[locations]
                break

    n_locations_monitored = len(locations_monitored)
    if rois_source == 'random':
        fname = f'{results_path}TraceMinIterative_SensorsLocations_N[{n}]_S[{signal_sparsity_roi}]_nROIS[{n_regions}]_nUrban[{n_urban}]_nRural[{n_rural}]_RandomSeed[{seed_random}]_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}]_nSensors[{n_locations_monitored}].pkl'
    else:
        fname = f'{results_path}TraceMinIterative_SensorsLocations_N[{n}]_S[{signal_sparsity_roi}]_nROIS[{n_regions}]_nUrban[{n_urban}]_nRural[{n_rural}]_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}]_nSensors[{n_locations_monitored}].pkl'

    with open(fname,'wb') as f:
        pickle.dump(locations_monitored,f,protocol=pickle.HIGHEST_PROTOCOL)
    print(f'File saved in {fname}')            




#%% dataset
class Dataset():
    def __init__(self,pollutant:str='O3',N:int=44,start_date:str='2011-01-01',end_date:str='2022-12-31',files_path:str='',synthetic_dataset:bool=False):
        self.pollutant = pollutant
        self.N = N
        self.start_date = start_date
        self.end_date = end_date
        self.files_path = files_path
        self.synthetic_dataset = synthetic_dataset
    
    def load_dataset(self):
        if self.synthetic_dataset:
            fname = f'{self.files_path}SyntheticData_{self.start_date}_{self.end_date}.csv'
        else:
            fname = f'{self.files_path}{self.pollutant}_catalonia_clean_N{self.N}_{self.start_date}_{self.end_date}.csv'
            self.stations_types = pd.read_csv(f'{self.files_path}stations_types.csv',index_col=0)
            self.coordinates = pd.read_csv(f'{self.files_path}coordinates.csv',index_col=0)
            self.coordinates_distances = pd.DataFrame([],index=self.coordinates.index,columns=self.coordinates.index)
            for i in range(self.coordinates.shape[0]):
                for j in range(self.coordinates.shape[0]):
                    self.coordinates_distances.iloc[i,j] = geopy.distance.geodesic(self.coordinates.iloc[i,:],self.coordinates.iloc[j,:]).km
            
        print(f'Loading dataset from {fname}')
        self.ds = pd.read_csv(fname,sep=',',index_col=0)
        self.ds.index = pd.to_datetime(self.ds.index)
        

    def check_dataset(self):
        print(f'Checking missing values in dataset')
        print(f'Percentage of missing values per location:\n{100*self.ds.isna().sum()/self.ds.shape[0]}')
        print(f'Dataset has {self.ds.shape[0]} measurements for {self.ds.shape[1]} locations.\n{self.ds.head()}')

    def sort_stations(self,order='dist',station_center='Ciutadella'):
        
        if order == 'dist':# Sort order of stations based on distance to one of them
            if station_center not in [i for i in self.coordinates_distances.columns]:
                raise ValueError(f'Station used for center is not present in dataset')

            self.distances = dataset.coordinates_distances.loc[station_center]
            self.distances.sort_values(ascending=True,inplace=True)
            self.ds = self.ds.loc[:,[f'O3_{i}' for i in self.distances.index if f'O3_{i}' in self.ds.columns]]
            
        elif order == 'type':#sort stations based on type of urban environment
            self.stations_types.sort_values(by=['Area','Name'],ascending=[False,True],inplace=True)
            self.stations_types['loc'] = range(0,self.stations_types.shape[0])
            self.ds = self.ds.loc[:,[f'O3_{i}' for i in self.stations_types.index if f'O3_{i}' in self.ds.columns]]
        print(f'Order of dataset locations: {self.ds.columns}')


#%%
# figures
class Figures():
    def __init__(self,save_path,figx=2.5,figy=2.5,fs_title=10,fs_label=10,fs_ticks=10,fs_legend=10,marker_size=3,dpi=300,use_grid=False,show_plots=False):
        self.figx = figx
        self.figy = figy
        self.fs_title = fs_title
        self.fs_label = fs_label
        self.fs_ticks = fs_ticks
        self.fs_legend = fs_legend
        self.marker_size = marker_size
        self.dpi = dpi
        self.save_path = save_path
        if show_plots:
            self.backend = 'Qt5Agg'
        else:
            self.backend = 'Agg'
        
        print('Setting mpl rcparams')
        
        font = {'weight':'normal',
                'size':str(self.fs_label),
                }
        
        lines = {'markersize':self.marker_size}
        
        fig = {'figsize':[self.figx,self.figy],
               'dpi':self.dpi
               }
        
        ticks={'labelsize':self.fs_ticks
            }
        axes={'labelsize':self.fs_ticks,
              'grid':False,
              'titlesize':self.fs_title
            }
        if use_grid:
            grid = {'alpha':0.5}
            mpl.rc('grid',**grid)
        
        mathtext={'default':'regular'}
        legend = {'fontsize':self.fs_legend}
        
        mpl.rc('font',**font)
        mpl.rc('figure',**fig)
        mpl.rc('xtick',**ticks)
        mpl.rc('ytick',**ticks)
        mpl.rc('axes',**axes)
        mpl.rc('legend',**legend)
        mpl.rc('mathtext',**mathtext)
        mpl.rc('lines',**lines)        
        mpl.use(self.backend)

    def curve_timeseries_singlestation(self,X:pd.DataFrame,station_name:str,date_init:str='2020-01-20',date_end:str='2021-10-27'):
        date_range = pd.date_range(start=date_init,end=date_end,freq='H')
        date_idx = [i for i in date_range if i in X.index]
        data = X.loc[date_idx,[station_name]]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(data)
        ax.set_xlabel('date')
        ax.set_ylabel('Concentration ($\mu$g/$m^3$)')
        fig.tight_layout()

    def curve_timeseries_allstations(self,X:pd.DataFrame,date_init:str='2020-01-20',date_end:str='2021-10-27',save_fig=False):
        date_range = pd.date_range(start=date_init,end=date_end,freq='H')
        date_idx = [i for i in date_range if i in X.index]
        data = X.loc[date_idx]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.fill_between(x=data.index,y1=np.percentile(X,axis=1,q=25),y2=np.percentile(X,axis=1,q=75))
        ax.set_xlabel('date')
        ax.set_ylabel('O$_3$ ($\mu$g/$m^3$)')
        fig.tight_layout()

        if save_fig:
            fname = self.save_path+'timeseries_Allstations.png'
            fig.savefig(fname,dpi=300,format='png',bbox_inches='tight')
            print(f'Figure saved at {fname}')

    
    def curve_timeseries_dailypattern_singlestation(self,X:pd.DataFrame,station_name:str):
        X_ = X.loc[:,station_name].copy()
        data = X_.groupby(X_.index.hour).median()
        q1,q3 = X_.groupby(X_.index.hour).quantile(q=0.25),X_.groupby(X_.index.hour).quantile(q=0.75)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(data)
        ax.fill_between(x=data.index,y1=q1,y2=q3,alpha=0.5)
        ax.set_xlabel('hour')
        yrange = np.arange(0,110,10)
        ax.set_yticks(yrange)
        ax.set_yticklabels([i for i in ax.get_yticks()])
        ax.set_ylabel('O$_3$ ($\mu$g/$m^3$)')
        ax.set_ylim(0,100)
        fig.tight_layout()
    
    def curve_timeseries_dailypattern_multiplestations(self,X:pd.DataFrame,stations_locs:list=[0,1,2,3],save_fig:bool=False):
        stations_names = [i for i in X.columns[stations_locs]]
        colors = ['#1a5276','orange','#117864','#943126']
        X_ = X.iloc[:,stations_locs].copy()
        data = X_.groupby(X_.index.hour).median()
        q1,q3 = X_.groupby(X_.index.hour).quantile(q=0.25),X_.groupby(X_.index.hour).quantile(q=0.75)

        
        fig = plt.figure()
        curves = {}
        for i in range(len(stations_locs)):
            ax = fig.add_subplot(221+i)
            curves[i] = ax.plot(data.iloc[:,i],label=stations_names[i],color=colors[i])
            ax.fill_between(x=data.index,y1=q1.iloc[:,i],y2=q3.iloc[:,i],alpha=0.5,color=colors[i])
            yrange = np.arange(0,110,10)
            ax.set_yticks(yrange)
            ax.set_yticklabels([i for i in ax.get_yticks()])    
            if (221+i)%2 == 1:
                ax.set_ylabel('O$_3$ ($\mu$g/$m^3$)')
            ax.set_ylim(0,100)
            if i in [2,3]:
                ax.set_xlabel('hour')

        handles = [curves[i][0] for i in curves.keys()]
        fig.legend(handles=[i for i in handles],ncol=2,bbox_to_anchor=(0.95,1.15),framealpha=1)
        fig.tight_layout()

        if save_fig:
            fname = f'{self.save_path}Curve_TimeSeriesHourly_ManyStations.png'
            fig.savefig(fname,dpi=300,format='png',bbox_inches='tight')
            print(f'Figure saved into {fname}')
        
    def curve_timeseries_dailypattern_allstations(self,X:pd.DataFrame):
        X_ = pd.DataFrame()
        for c in X.columns:
            X_ = pd.concat((X_,X.loc[:,c]),axis=0)
        X_ = X_.loc[:,0]
        data = X_.groupby(X_.index.hour).median()
        q1,q3 = X_.groupby(X_.index.hour).quantile(q=0.25),X_.groupby(X_.index.hour).quantile(q=0.75)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(data)
        ax.fill_between(x=data.index,y1=q1,y2=q3,alpha=0.5)
        ax.set_xlabel('hour')
        yrange = np.arange(0,110,10)
        ax.set_yticks(yrange)
        ax.set_yticklabels([i for i in ax.get_yticks()])
        ax.set_ylabel('O$_3$ ($\mu$g/$m^3$)')
        ax.set_ylim(0,100)
        fig.tight_layout()

    def boxplot_measurements(self,X,save_fig):
        n = X.shape[1]
        yrange = np.arange(0.0,300,50)
        xrange = np.arange(1,n+1,1)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        bp = ax.boxplot(x=X,notch=False,vert=True,
                   whis=1.5,bootstrap = None,
                   positions=[i for i in range(len(xrange))],widths=0.5,labels=[str(i) for i in xrange],
                   flierprops={'marker':'.','markersize':1},
                   patch_artist=True)
        
        ax.set_yticks(yrange)
        ax.set_yticklabels([np.round(i,2) for i in ax.get_yticks()])
        ax.set_ylabel('O$_3$ ($\mu$g/$m^3$)')
        
        xrange = [i-1 for i in xrange if i%5==0]
        ax.set_xticks(xrange)
        ax.set_xticklabels([int(i+1) for i in xrange],rotation=0)
        ax.set_xlabel('Location index')
        fig.tight_layout()
        if save_fig:
            fname = self.save_path+'boxplot_concentration_allStations.png'
            fig.savefig(fname,dpi=300,format='png',bbox_inches='tight')
            print(f'Figure saved at {fname}')

    def geographical_network_visualization(self,map_path:str,df_coordinates:pd.DataFrame,locations_monitored:np.array=[],roi_idx:dict={},show_legend:bool=False,show_deployed_sensors:bool=True,save_fig:bool=False)->plt.figure:
        """
        Figure showing the geographical area where sensors are deployed along with coordinates of reference stations

        Args:
            map_path (str): path to map file
            df_coordinates (pd.DataFrame): dataframe containing coordiantes(Latitude,Longitude) of each reference station
            locations_monitored (np.array, optional): indices of monitored locations. Defaults to [].
            roi_idx (dict): dictionary indicating indices that belong to each region of interest (ROI) in case of heterogeneous design. The keys correspond to parameter used for separating ROIs.
            show_legend (bool, optional): Show legend indicating monitored and unmonitored locations. Defaults to False.
            save_fig (bool, optional): save generated figure. Defaults to False.

        Returns:
            plt.figure: Figure with map and stations 
        """
        
        if len(locations_monitored)!=0:
            df_coords_monitored = df_coordinates.iloc[locations_monitored]
            df_coords_unmonitored = df_coordinates.iloc[[i for i in range(df_coordinates.shape[0]) if i not in locations_monitored]]
            geometry_monitored = [Point(xy) for xy in zip(df_coords_monitored['Longitude'], df_coords_monitored['Latitude'])]
            geometry_unmonitored = [Point(xy) for xy in zip(df_coords_unmonitored['Longitude'], df_coords_unmonitored['Latitude'])]
            gdf_monitored = GeoDataFrame(df_coords_monitored, geometry=geometry_monitored)
            gdf_unmonitored = GeoDataFrame(df_coords_unmonitored, geometry=geometry_unmonitored)

        else:
            df_coords_monitored = df_coordinates.copy()
            geometry_monitored = [Point(xy) for xy in zip(df_coords_monitored['Longitude'], df_coords_monitored['Latitude'])]
            gdf_monitored = GeoDataFrame(df_coords_monitored, geometry=geometry_monitored)
        
        spain = gpd.read_file(f'{map_path}ll_autonomicas_inspire_peninbal_etrs89.shp')
        catalonia = spain.loc[spain.NAME_BOUND.str.contains('Catalunya')]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        geo_map = catalonia.plot(ax=ax,color='#117a65')
        
        try:
            if len(roi_idx)!=0:
                markers = ['^','o','s','P','D']
                colors = ['k','#943126']
                if show_deployed_sensors:
                    print('Map showing monitored and unmonitored locations for each ROI')
                    for i,idx,m in zip(range(len(roi_idx)),roi_idx.values(),markers):
                        #locations_monitored_roi = np.array(locations_monitored)[np.isin(locations_monitored,idx)]
                        locations_monitored_roi = np.array([i for i in locations_monitored if i in idx])
                        locations_unmonitored_roi = np.array([i for i in range(df_coordinates.shape[0]) if i not in locations_monitored and i in idx])
                        print(f'locations monitored for ROI {i}: {len(locations_monitored_roi)}\nlocations unmonitored for ROI {i}: {len(locations_unmonitored_roi)}')
                        # monitored locations in ROI
                        df_coords_monitored = df_coordinates.iloc[[i for i in range(df_coordinates.shape[0]) if i in locations_monitored_roi]]
                        geometry_monitored = [Point(xy) for xy in zip(df_coords_monitored['Longitude'], df_coords_monitored['Latitude'])]
                        gdf_monitored = GeoDataFrame(df_coords_monitored, geometry=geometry_monitored)
                        gdf_monitored.plot(ax=geo_map, marker=m, color=colors[1], markersize=6,label=f'$\mathcal{{R}}_{i+1}{{\cap}}\mathcal{{S}}$')
                        
                        # unmonitored locations in ROI
                        df_coords_unmonitored = df_coordinates.iloc[[i for i in range(df_coordinates.shape[0]) if i in locations_unmonitored_roi]]
                        print(f'Shape of unmonitored dataframe coordinates: {df_coords_unmonitored.shape}')
                        geometry_unmonitored = [Point(xy) for xy in zip(df_coords_unmonitored['Longitude'], df_coords_unmonitored['Latitude'])]
                        gdf_unmonitored = GeoDataFrame(df_coords_unmonitored, geometry=geometry_unmonitored)
                        gdf_unmonitored.plot(ax=geo_map, marker=m, color=colors[0], markersize=6,label=f'$\mathcal{{R}}_{i+1}{{\cap}}\mathcal{{S}}^{{c}}$') 

                else: # show icons belonging to each ROI
                    for i,idx,m,c in zip(range(len(roi_idx)),roi_idx.values(),markers,colors):
                        
                        df_coords_idx = df_coordinates.iloc[[i for i in range(df_coordinates.shape[0]) if i in idx]]
                        geometry_idx = [Point(xy) for xy in zip(df_coords_idx['Longitude'], df_coords_idx['Latitude'])]
                        gdf_monitored = GeoDataFrame(df_coords_idx, geometry=geometry_idx)
                        gdf_monitored.plot(ax=geo_map, marker=m, color=c, markersize=6,label=f'$\mathcal{{R}}_{i+1}$')
                
            else:
                gdf_monitored.plot(ax=geo_map, marker='o', color='#943126', markersize=6,label=f'Monitoring node')
                gdf_unmonitored.plot(ax=geo_map, marker='o', color='k', markersize=6,label=f'Unmonitored locations')
        except:
            warnings.warn('No unmonitored locations or unexpected error in dataframe')
        ax.set_xlim(0.0,4.0)
        ax.set_ylim(40.5,43)
        
        ax.set_ylabel('Latitude (degrees)')
        ax.set_xlabel('Longitude (degrees)')

        # set legend location
        if show_legend:
            if show_deployed_sensors:
                if len(roi_idx) == 2:
                    ax.legend(loc='center',ncol=len(roi_idx),framealpha=0,
                              handletextpad=-0.8,columnspacing=5e-4,labelspacing=0.1,bbox_to_anchor=(0.73,0.1))
                elif len(roi_idx)==3:
                    ax.legend(loc='center',ncol=len(roi_idx),framealpha=0,
                              handletextpad=-0.8,columnspacing=1e-6,labelspacing=0.05,bbox_to_anchor=(0.6,0.1))
            else:
                ax.legend(loc='lower right',ncol=1,framealpha=0.1,handletextpad=-0.1,columnspacing=0.5)
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()
        
        # save generated figure
        if save_fig:
            if show_deployed_sensors:
                fname = self.save_path+f'Map_PotentialLocations_{len(roi_idx)}ROIs.png'
            else:
                if len(roi_idx)!=0:
                    fname = self.save_path+f'Map_PotentialLocations_{len(roi_idx)}ROIs.png'
                else:
                    fname = self.save_path+f'Map_PotentialLocations.png'
            fig.savefig(fname,dpi=600,format='png',bbox_inches='tight')
            print(f'Figure saved at {fname}')
        return fig
        

    # Low-rank plots
    def singular_values_cumulative_energy(self,sing_vals,n,synthetic_dataset=False,save_fig=False):
        """
        Plot sorted singular values ratio and cumulative energy

        Parameters
        ----------
        sing_vals : numpy array
            singular values
        n : int
            network size
        save_fig : bool, optional
            save generated figures. The default is False.

        Returns
        -------
        None.

        """
        cumulative_energy = np.cumsum(sing_vals)/np.sum(sing_vals)
        xrange = np.arange(0,sing_vals.shape[0],1)
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        ax.plot(xrange,cumulative_energy,color='#1f618d',marker='o')
        ax.set_xticks(np.concatenate(([0.0],np.arange(xrange[9],xrange[-1]+1,10))))
        ax.set_xticklabels([int(i+1) for i in ax.get_xticks()])
        ax.set_xlabel('$i$th singular value')
        
        #yrange = np.arange(0.5,1.05,0.05)
        yrange = np.arange(0.,1.2,0.2)
        ax.set_yticks(yrange)
        ax.set_yticklabels([np.round(i,2) for i in ax.get_yticks()])
        ax.set_ylabel('Cumulative energy')
        if synthetic_dataset:
            ax.set_yscale('log')
        fig1.tight_layout()
        
        fig2 = plt.figure()
        ax = fig2.add_subplot(111)
        ax.plot(xrange, sing_vals / np.max(sing_vals),color='#1f618d',marker='o')
        ax.set_xticks(np.concatenate(([0.0],np.arange(xrange[9],xrange[-1]+1,10))))
        ax.set_xticklabels([int(i+1) for i in ax.get_xticks()],rotation=0)
        ax.set_xlabel('$i$th singular value')

        yrange = np.logspace(-4,0,5)
        ax.set_yticks(yrange)
        ax.set_ylabel('Normalized singular values')
        ax.set_ylim(1e-2,1)
        ax.set_yscale('log')
        if synthetic_dataset:
            ax.set_yscale('log')
        fig2.tight_layout()
        
        if save_fig:
            fname = self.save_path+f'Curve_sparsity_cumulativeEnergy_N{n}.png'
            fig1.savefig(fname,dpi=300,format='png')
            print(f'Figure saved at: {fname}')

            fname = self.save_path+f'Curve_sparsity_singularValues_N{n}.png'
            fig2.savefig(fname,dpi=300,format='png')
            print(f'Figure saved at: {fname}')
    
    def singular_values_cumulative_energy_sameFigure(self,sing_vals,n,save_fig=False):
        """
        Plot sorted singular values ratio and cumulative energy in the same figure

        Parameters
        ----------
        sing_vals : numpy array
            singular values
        n : int
            network size
        save_fig : bool, optional
            save generated figures. The default is False.

        Returns
        -------
        None.

        """
        cumulative_energy = np.cumsum(sing_vals)/np.sum(sing_vals)
        xrange = np.arange(0,sing_vals.shape[0],1)
        fig = plt.figure(constrained_layout=True)
        ax = fig.add_subplot(111)

        l1 = ax.plot(xrange, sing_vals / np.max(sing_vals),color='#ba4a00',marker='o',label='Normalized singular values')
        ax.set_xticks(np.concatenate(([0.0],np.arange(xrange[9],xrange[-1]+1,10))))
        ax.set_xticklabels([int(i+1) for i in ax.get_xticks()],rotation=0)
        ax.set_xlabel('$i$th singular value')
        yrange = np.logspace(-4,0,5)
        ax.set_yticks(yrange)
        ax.set_ylabel('Normalized singular values')
        ax.set_ylim(1e-2,1)
        ax.set_yscale('log')

        ax2 = ax.twinx()
        l2 = ax2.plot(xrange,cumulative_energy,color='#1f618d',marker='o',label='Cumulative energy')
        ax2.set_xticks(np.concatenate(([0.0],np.arange(xrange[9],xrange[-1]+1,10))))
        ax2.set_xticklabels([int(i+1) for i in ax2.get_xticks()])
        
        yrange = np.arange(0.,1.2,0.2)
        ax2.set_yticks(yrange)
        ax2.set_yticklabels([np.round(i,2) for i in ax2.get_yticks()])
        #ax2.set_ylabel('Cumulative energy')
        ax2.set_ylim(0,1)
        
        lines = l1+l2
        labels = [l.get_label() for l in lines]
        ax.yaxis.set_label_coords(-0.2,0.4)
        #ax.legend(lines,labels,loc='center',ncol=1,framealpha=1.,bbox_to_anchor=(0.5,1.15),handlelength=0.5,handletextpad=0.1)
        #fig.tight_layout()
        
        if save_fig:
            fname = self.save_path+f'Curve_singVals_cumulativeEnergy_N{n}.png'
            fig.savefig(fname,dpi=600,format='png',bbox_inches='tight')
            print(f'Figure saved at: {fname}')


    def boxplot_validation_rmse_svd(self,rmse_sparsity,n,max_sparsity_show=10,synthetic_dataset=False,save_fig=False) -> plt.figure:
        yrange = np.arange(0.0,35,5)
        xrange = rmse_sparsity.columns[:max_sparsity_show]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        bp = ax.boxplot(x=rmse_sparsity.iloc[:,:max_sparsity_show],notch=False,vert=True,
                   whis=1.5,bootstrap = None,
                   positions=[i for i in range(len(xrange))],widths=0.5,labels=[str(i) for i in xrange],
                   flierprops={'marker':'.','markersize':1},
                   patch_artist=True)
        
        ax.set_yticks(yrange)
        ax.set_yticklabels([np.round(i,2) for i in ax.get_yticks()])
        if synthetic_dataset:
            ax.set_yscale('log')
            ax.set_ylim(1e-2,1e1)
        else:
            ax.set_ylim(0,30)
        ax.set_ylabel('RMSE ($\mu$g/$m^3$)')
        xrange = np.array([i-1 for i in xrange if i%5==0])
        ax.set_xticks(xrange)
        ax.set_xticklabels([int(i+1) for i in xrange],rotation=0)
        ax.set_xlabel('Sparsity level')
        fig.tight_layout()

        if save_fig:
            fname = self.save_path+f'boxplot_RMSE_SVDreconstruction_validationSet_Smin{xrange.min()}_Smax{xrange.max()}_N{n}.png'
            fig.savefig(fname,dpi=300,format='png')
            print(f'Figure saved in {fname}')
    
        return fig

    def curve_frobenius_svd(self,frobenius_norm_svd,save_fig=False):
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(frobenius_norm_svd,color='k')
        xrange = np.arange(0,frobenius_norm_svd.shape[0],1)
        ax.set_xticks([i for i in xrange if i==0 or (i+1)%10==0])
        ax.set_xticklabels([int(i+1) for i in ax.get_xticks()])
        ax.set_xlabel('$i$th singular value')
        ax.set_ylabel('Fractional SVD reconstruction error')
        fig.tight_layout()
        if save_fig:
            fname = f'{self.save_path}Curve_singVals_NormalizedFrobeniusSVD.png'
            fig.savefig(fname,dpi=300,format='png')
            print(f'Figure saved in {fname}')
    
    def boxplot_rmse_comparison(self,rmse_method1:pd.DataFrame,rmse_method2:pd.DataFrame,maxerror:bool=False,save_fig:bool=False)->plt.figure:
        """
        Boxplot comparing validation set RMSE using 2 different numbers of deployed senors.
        E.g: compare fully monitored vs reduced

        Args:
            rmse_method1 (pd.DataFrame): rmse for certain number of sensors
            rmse_method2 (pd.DataFrame): rmse for different number of sensors (for example fully monitored)
            maxerror (bool, optional): dataframes contain maximum reconstruction error instead of RMSE. Defaults to False.
            save_fig (bool, optional): Save generqated figure. Defaults to False.

        Returns:
            plt.figure: Figure
        """
        n_sensors_1 = rmse_method1.columns[0]
        n_sensors_2 = rmse_method2.columns[0]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        bp1 = ax.boxplot(x=rmse_method1,notch=False,vert=True,
                   whis=1.5,bootstrap = None,
                   positions=[0],widths=0.5,labels=[n_sensors_1],
                   flierprops={'marker':'.','markersize':1},
                   patch_artist=True)
        
        bp2 = ax.boxplot(x=rmse_method2,notch=False,vert=True,
                   whis=1.5,bootstrap = None,
                   positions=[1],widths=0.5,labels=[n_sensors_2],
                   flierprops={'marker':'.','markersize':1},
                   patch_artist=True)
        bp1['boxes'][0].set_facecolor('lightgreen')
        bp2['boxes'][0].set_facecolor('#1a5276')
        
        if maxerror:
            yrange = np.arange(0.,55.,5)
            ax.set_ylim(0,50)
        else:
            yrange = np.arange(0.,22.,2)
            ax.set_ylim(0,20)
        ax.set_yticks(yrange)
        ax.set_yticklabels([np.round(i,1) for i in ax.get_yticks()])

        if maxerror:
            ax.set_ylabel('Max error ($\mu$g/$m^3$)')        
        else:
            ax.set_ylabel('RMSE ($\mu$g/$m^3$)')        
        ax.set_xlabel('Number of deployed sensors')
        fig.tight_layout()

        if save_fig:
            if maxerror:
                fname = f'{self.save_path}Maxerrorcomparison_NsensorsTotal_N1{n_sensors_1}_N2{n_sensors_2}.png'
            else:
                fname = f'{self.save_path}RMSEcomparison_NsensorsTotal_N1{n_sensors_1}_N2{n_sensors_2}.png'
            fig.savefig(fname,dpi=300,format='png')
    
        return fig
    
    def boxplot_errorratio(self,df_error1:pd.DataFrame,df_error2:pd.DataFrame,save_fig:bool=False)->plt.figure:
        n_sensors1 = df_error1.columns[0]
        n_sensors2 = df_error2.columns[0]
        df_ratio = df_error1.to_numpy() / df_error2.to_numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        bp = ax.boxplot(x=df_ratio,notch=False,vert=True,
                   whis=1.5,bootstrap = None,
                   positions=[0],widths=0.5,labels=[f'{n_sensors1} sensors vs {n_sensors2} senors'],
                   flierprops={'marker':'.','markersize':1},
                   patch_artist=True)
        
        
        bp['boxes'][0].set_facecolor('#1a5276')
        
        yrange = np.arange(0.,3.5,0.5)
        ax.set_ylim(0,3)
        ax.set_yticks(yrange)
        ax.set_yticklabels([np.round(i,1) for i in ax.get_yticks()])

        ax.set_ylabel('Reconstruction errors ratio')        
        ax.set_xlabel('')
        fig.tight_layout()

        if save_fig:
            fname = f'{self.save_path}ErrorRatio_NsensorsTotal_N1{n_sensors1}_N2{n_sensors2}.png'
            fig.savefig(fname,dpi=300,format='png')
    
        return fig
    
    def hist_worsterror(self,errormax_fullymonitored,errormax_reconstruction,n_sensors,save_fig=False):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(x=errormax_fullymonitored,bins=np.arange(0.,5.1,0.1),density=True,cumulative=False,color='#1a5276',label='Fully monitored network')
        ax.vlines(x=errormax_fullymonitored.mean(),ymin=0.0,ymax=1.0,colors='#1a5276',linestyles='--')
        ax.hist(x=errormax_reconstruction,bins=np.arange(0.,5.1,0.1),density=True,cumulative=False,color='orange',label=f'Reconstruction with {n_sensors} sensors',alpha=0.5)
        ax.vlines(x=errormax_reconstruction.mean(),ymin=0.0,ymax=1.0,colors='orange',linestyles='--')
        ax.set_xlabel('Maximum reconstruction error')
        ax.set_ylabel('Probability density')
        ax.legend(loc='upper left',ncol=1,framealpha=0.5)
        ax.set_xlim(0,5)
        ax.set_ylim(0,1)
        fig.tight_layout()
        if save_fig:
            fname = f'{self.save_path}Histogram_error_fullymonitored_vs_reconstruction_Nsensors{n_sensors}.png'
            fig.savefig(fname,dpi=300,format='png')
            print(f'Figure saved at {fname}')

    def hist_errorratio(self,errormax_fullymonitored,errormax_reconstruction,n_sensors,save_fig=False):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(x=errormax_reconstruction.to_numpy()/errormax_fullymonitored.to_numpy(),bins=np.arange(0,3.1,0.1),density=True,cumulative=False,color='#1a5276')
        ax.set_xlabel('Maximum error ratio')
        ax.set_ylabel('Probability density')
        ax.set_xlim(0,3)
        fig.tight_layout()
        if save_fig:
            fname = f'{self.save_path}Histogram_errorRatio_Nsensors{n_sensors}.png'
            fig.savefig(fname,dpi=300,format='png')
            print(f'Figure saved at {fname}')
    
    def curve_errorvariance_comparison(self,errorvar_fullymonitored:list,errorvar_reconstruction:list,variance_threshold_ratio:float,worst_coordinate_variance_fullymonitored:float,n:int,n_sensors:int,errorvar_reconstruction_Dopt:list=[],roi_idx:dict={},n_sensors_Dopt:int=0,method:str='random_based',random_seed:int=0,save_fig:bool=False) -> plt.figure:
        """
        Show error variance over a testing set at each network location. 
        The error variance is obtained after reconstructing the signal from p measurements.
        The p measurement locations are obtained from network design algorithm or D-optimality criteria.
        It also shows the threshold line which the network design algorithm used.
        Another algorithm can be shown for comparison.

        Args:
            errorvar_fullymonitored (list): error variance at each network location obtained with a fully monitored network. This corresponds to the lowest error variance possible.
            errorvar_reconstruction (list): error variance at each network locations obtained with a network with a reduced number of deployed sensors.
            variance_threshold_ratio (float): variance threshold ratio used for design algorithm. It is a multiple of the worst_coordinate_variance_fullymonitored.
            worst_coordinate_variance_fullymonitored (float): fully-monitored network worst coordinate error variance
            n (int): total number of network points
            n_sensors (int): number of deployed sensors
            errorvar_reconstruction_Dopt (list): error variance at each network location obtained by D-optimality (or other) criteria. Defaults to [].
            roi_idx (dict): dictionary containing indices of locations that belong to each ROI. The keys indicate the threshold used to separate the network.
            save_fig (bool, optional): Save generated figure. Defaults to False.

        Returns:
            plt.figure: Figure with error variance curves
        """
        if type(variance_threshold_ratio) is float:
            variance_threshold = variance_threshold_ratio*worst_coordinate_variance_fullymonitored
        
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(errorvar_fullymonitored,color='#1d8348',label='Fully monitored network')
            if len(errorvar_reconstruction_Dopt) !=0:
                ax.plot(errorvar_reconstruction_Dopt,color='orange',label=f'Joshi-Boyd solution',alpha=0.8)
            ax.plot(errorvar_reconstruction,color='#1a5276',label=f'Network design solution')
            ax.hlines(y=variance_threshold,xmin=0,xmax=n+1,color='k',linestyles='--',label=rf'Design threshold $\rho$={variance_threshold_ratio:.2f}$\rho_n$')
            xrange = np.arange(-1,n,10)
            xrange[0] = 0
            ax.set_xticks(xrange)
            ax.set_xticklabels([i+1 for i in ax.get_xticks()])
            ax.set_xlim(0,n)
            ax.set_xlabel('Location index')
            yrange = np.arange(0,1.75,0.25)
            ax.set_yticks(yrange)
            ax.set_yticklabels([np.round(i,2) for i in ax.get_yticks()])
            ax.set_ylim(0,1.5)
            ax.set_ylabel('Error variance')
            ax.legend(loc='center',ncol=2,framealpha=0.5,bbox_to_anchor=(0.5,1.1))
            fig.tight_layout()
            if save_fig:
                fname = f'{self.save_path}Curve_errorVariance_Threshold{variance_threshold_ratio:.2f}_Nsensors{n_sensors}.png'
                fig.savefig(fname,dpi=300,format='png',bbox_inches='tight')
                print(f'Figure saved at {fname}')


        else: # heterogeneous thresholds over multiple ROIs
            variance_threshold = [t*w for t,w in zip(variance_threshold_ratio,worst_coordinate_variance_fullymonitored)]
            # sort coordinate error variance such that the ROIs are shown in order
            coordinate_error_variance_fully_monitored_sorted = np.concatenate([errorvar_fullymonitored[i] for i in roi_idx.values()])
            coordinate_error_variance_design_sorted = np.concatenate([errorvar_reconstruction[i] for i in roi_idx.values()])

            fig = plt.figure(constrained_layout=True)
            ax = fig.add_subplot(111)
            # coordinate error variance at each location
            ax.plot(coordinate_error_variance_fully_monitored_sorted,color='#943126',label='Fully monitored case')
            # horizontal lines showing threshold design
            n_roi = np.concatenate([[0],[len(i) for i in roi_idx.values()]])
            n_roi_cumsum = np.cumsum(n_roi)
            for v,l in zip(variance_threshold,range(len(n_roi_cumsum))):
                if l==0:
                    ax.hlines(y=v,xmin=n_roi_cumsum[l]-1,xmax=n_roi_cumsum[l+1]-1,color='k',linestyles='--',label='Design threshold')
                else:
                    ax.hlines(y=v,xmin=n_roi_cumsum[l],xmax=n_roi_cumsum[l+1]-1,color='k',linestyles='--')
            
            # Joshi Boyd and IRNet results
            if len(errorvar_reconstruction_Dopt) !=0:
                coordinate_error_variance_Dopt_sorted = np.concatenate([errorvar_reconstruction_Dopt[i] for i in roi_idx.values()])
                ax.plot(coordinate_error_variance_Dopt_sorted,color='orange',label=f'JB {n_sensors_Dopt} sensors',alpha=0.8)
            ax.plot(coordinate_error_variance_design_sorted,color='#1a5276',label=f'IRWNet {n_sensors} sensors')
            
            xrange = np.arange(-1,n,10)
            xrange[0] = 0
            ax.set_xticks(xrange)
            ax.set_xticklabels([i+1 for i in ax.get_xticks()])
            ax.set_xlim(-0.5,n)
            ax.set_xlabel('Location index')
            yrange = np.arange(0,3.5,0.5)
            ax.set_yticks(yrange)
            ax.set_yticklabels([np.round(i,2) for i in ax.get_yticks()])
            ax.set_ylim(0,3.0+0.1)
            ax.set_ylabel('Per-coordinate error variance')
            ax.legend(loc='center',ncol=2,framealpha=1,
                      handlelength=0.5,handletextpad=0.1,columnspacing=0.2,
                      bbox_to_anchor=(0.5,0.88))
            #fig.tight_layout()
            if save_fig:
                #fname = f'{self.save_path}Curve_errorVariance_Threshold{variance_threshold_ratio}_Nsensors{n_sensors}_NsensorsDopt{n_sensors_Dopt}_NsensorsROIDopt_{n_sensors_roi}.png'
                if method == 'random_based':
                    fname = f'{self.save_path}Curve_errorVariance_VarThreshold{variance_threshold_ratio}_Nsensors{n_sensors}_NsensorsDopt{n_sensors_Dopt}_randomSeed{random_seed}.png'
                else:
                    fname = f'{self.save_path}Curve_errorVariance_VarThreshold{variance_threshold_ratio}_Nsensors{n_sensors}_NsensorsDopt{n_sensors_Dopt}.png'
                fig.savefig(fname,dpi=300,format='png',bbox_inches='tight')
                print(f'Figure saved at {fname}')


    def curve_rmse_hourly(self,rmse_time,month=0,save_fig=False):
        hours = [i for i in rmse_time.keys()]
        median = [rmse_time[i].median().to_numpy()[0] for i in hours]
        q1,q3 = [rmse_time[i].quantile(q=0.25).to_numpy()[0] for i in hours], [rmse_time[i].quantile(q=0.75).to_numpy()[0] for i in hours]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(median,color='#1a5276')
        ax.fill_between(x=hours,y1=q1,y2=q3,color='#1a5276',alpha=0.5)
        ax.set_xticks(hours[::4])
        ax.set_xticklabels([i for i in ax.get_xticks()])
        ax.set_xlabel('Hour')
        yrange = np.arange(0,12.,2.)
        ax.set_yticks(yrange)
        ax.set_yticklabels([np.round(i,1) for i in ax.get_yticks()])
        ax.set_ylabel('RMSE ($\mu$g/$m^3$)')
        ax.set_ylim(yrange[0],yrange[-1])
        fig.tight_layout()
        if save_fig:
            fname = f'{self.save_path}deploy_sensors_hourly_month{month}.png'
            fig.savefig(fname,dpi=300,format='png')
        return fig

#%%

if __name__ == '__main__':
    """ load dataset to use """
    abs_path = os.path.dirname(os.path.realpath(__file__))
    files_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/files/catalonia/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/test/'
    # load dataset
    pollutant = 'O3'
    start_date = '2011-01-01'
    end_date = '2022-12-31'
    N=48
    dataset = Dataset(pollutant,N,start_date,end_date,files_path)
    dataset.load_dataset()
    dataset.check_dataset()
    dataset.sort_stations(order='type')
    
    # train/val/test split
    train_ratio = 0.75#0.75
    validation_ratio = 0.15#0.15
    test_ratio = 0.1#0.10
    shuffle = True
    if train_ratio + validation_ratio + test_ratio != 1.:
        raise ValueError('Ratios of train/validation/test set do not match.')
    X_train, X_test = train_test_split(dataset.ds, test_size= 1 - train_ratio,shuffle=shuffle,random_state=92)
    if validation_ratio and test_ratio != 0.0:
        X_val, X_test = train_test_split(X_test, test_size=test_ratio/(test_ratio + validation_ratio),shuffle=shuffle,random_state=92) 

    # Low-rank basis: SVD
    snapshots_matrix_train = X_train.to_numpy().T
    snapshots_matrix_val = X_val.to_numpy().T
    snapshots_matrix_test = X_test.to_numpy().T
    snapshots_matrix_train_centered = snapshots_matrix_train - snapshots_matrix_train.mean(axis=1)[:,None]
    snapshots_matrix_val_centered = snapshots_matrix_val - snapshots_matrix_train.mean(axis=1)[:,None]
    snapshots_matrix_test_centered = snapshots_matrix_test - snapshots_matrix_train.mean(axis=1)[:,None]
    U,sing_vals,Vt = np.linalg.svd(snapshots_matrix_train,full_matrices=False)
    print(f'Training snapshots matrix has dimensions {snapshots_matrix_train.shape}.\nLeft singular vectors matrix has dimensions {U.shape}\nRight singular vectors matrix has dimensions [{Vt.shape}]\nNumber of singular values: {sing_vals.shape}')

    # Regions of Interest  parameters
    
    rois_source = 'random'
    seed_random = 0
    rois_thresholds = {'urban':9,'rural':18}
    n_urban,n_rural = N,0 # sum == n
    n_regions = len(rois_thresholds)
    if rois_source == 'file':
        dataset.rois = dataset.stations_types.copy()        
        dataset.rois['thresholds'] = [rois_thresholds[i] for i in dataset.rois['Area']]

    elif rois_source == 'random':
        if n_urban+n_rural != N:
            raise ValueError(f'Number of urban ({n_urban}) plus rural ({n_rural}) mismatch total number of locations ({N})')
        area = np.concatenate((n_urban*['urban'],n_rural*['rural']),axis=0)
        rng = np.random.default_rng(seed=seed_random)
        rng.shuffle(area)
        dataset.rois = pd.DataFrame(area,index=dataset.ds.columns,columns=['Area'])
        dataset.rois['thresholds'] = [rois_thresholds[i] for i in dataset.rois['Area']]

    sigma_threshold = dataset.rois['thresholds'].to_numpy()
    
    # device parameters
    sigma_noise = 0.25*rois_thresholds['urban']
    seed_noise = 0
    

    # exploratory figures    
    """" 
    plots = Figures(save_path=results_path,
                    figx=3.5,figy=2.5,
                    marker_size=1,
                    fs_label=15,fs_ticks=10,fs_legend=6,fs_title=10,
                    show_plots=True)
    plots.boxplot_measurements(X_train,save_fig=False)
    plots.geographical_network_visualization(map_path=f'{files_path}ll_autonomicas_inspire_peninbal_etrs89/',df_coordinates=dataset.coordinates.reindex(dataset.distances.index),
                                                show_legend=True,show_deployed_sensors=False,save_fig=False)
    plt.show()
    """

    """
    The script executes one of the following:
        - determine_sparsity
        - sizing_and_placement
        - auto_validate
        - SPCREC_sensor_placement
        - reconstruct_signal
    """
    execute = 'auto_validate'
    
    # Get signal sparsity via SVD decomposition
    if execute == 'determine_sparsity':
        
        # signal sparsity from error at reconstruction train/val dataset
        print('\nDetermine signal sparsity from SVD decomposition.\nUse singular values ratios, cumulative energy, or reconstruction error for validation set.')
        s_range = np.arange(1,sing_vals.shape[0]+1,1)
        mse_svd,error_variance_svd, frobenius_norm_svd = signal_reconstruction_svd(U,snapshots_matrix_train,s_range)
        """O3 Envea device: sigma=0.1ppb=1*1.96 ug/m3"""
        ppb = 1.96#ug/m3
        mse_threshold = (1*ppb)**2 #~ sigma^2
        signal_sparsity_mse = np.min(np.where(mse_svd.mean(axis=0)<mse_threshold))
        print(f'Threshold based on mse < {mse_threshold:.2f}. Sparsity: {signal_sparsity_mse}')
        
        # signal sparsity from cumulative energy
        cumulative_energy = np.cumsum(sing_vals)/np.sum(sing_vals)
        energy_threshold = 0.9
        signal_sparsity_energy = np.where(cumulative_energy>=energy_threshold)[0][0]
        print(f'Energy threshold of {energy_threshold} reached at singular at singular value index: {signal_sparsity_energy}')
        
        # signal sparsity from hard thresholding
        ppb = 1.96#ug/m3
        sigma_noise = 0.1*ppb
        signal_sparsity_hard_threshold = singular_value_hard_threshold(snapshots_matrix_train,sing_vals,noise=-1)
        print(f'Hard-threshold singular value: {signal_sparsity_hard_threshold}')
        
        # dataset and sparsity figures
        plots = Figures(save_path=results_path,marker_size=1,
                        figx=3.5,figy=2.5,
                        fs_label=12,fs_ticks=12,fs_legend=10,fs_title=10,
                        show_plots=True)
        plots.singular_values_cumulative_energy_sameFigure(sing_vals,n=X_train.shape[1],save_fig=True)
        fig_rmse_sparsity_train = plots.boxplot_validation_rmse_svd(mse_svd,max_sparsity_show=sing_vals.shape[0],save_fig=False)
        plots = Figures(save_path=results_path,marker_size=1,
                        figx=3.5,figy=2.5,
                        fs_label=8,fs_ticks=8,fs_legend=10,fs_title=10,
                        show_plots=True)

        plots.curve_frobenius_svd(frobenius_norm_svd.values[0],save_fig=False)
        plt.show()
    
    elif execute=='sizing_and_placement':
        signal_sparsity = np.arange(34,N+1,1)# -1 for every possible sparsity value
        epsilon_range = np.array([0.2,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01])#np.array([0.2,0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01])
        sizing_and_placement(U=U,signal_sparsity=signal_sparsity,
                             sigma_noise=sigma_noise,sigma_threshold=sigma_threshold,
                             rois_thresholds=rois_thresholds,rois_source=rois_source,
                             n_urban=n_urban,n_rural=n_rural,
                             seed_random=seed_random,epsilon_range=epsilon_range)

    elif execute=='validate':
        """ Iterate over a scenario to find the optimal basis that solves the SP-CREC for a validation snapshots matrix"""
        n_locations_thresholded,n_locations_monitored = validate_spcrec(path_to_files=f'{results_path}AirPollution/',dataset=dataset,
                                                                        snapshots_matrix=snapshots_matrix_val,U=U,sigma_noise=sigma_noise,
                                                                        rois_thresholds=rois_thresholds,n_urban=n_urban,n_rural=n_rural,
                                                                        rois_source=rois_source,
                                                                        seed_random=0,seed_noise=seed_noise
                                                                        )
        df = n_locations_thresholded==N
        idx = df.where(df).stack().index.to_list()
        p_optimal,idx_optimal = np.min([n_locations_monitored.loc[i] for i in idx]),np.argmin([n_locations_monitored.loc[i] for i in idx])
        s_optimal,e_optimal = idx[idx_optimal]
        print(f'\nOptimal sparsity and epsilon hyperparameters:\ns:{s_optimal}\nepsilon:{e_optimal}\nNumber of monitored locations: {p_optimal}')

    elif execute == 'auto_validate':
        """ explore thresholded locations over a validation set """

        """ Iterate over a scenario to find the optimal basis that solves the SP-CREC for a validation snapshots matrix"""
        path_to_files = f'{results_path}AirPollution/'
        s_range = np.arange(1,U.shape[1]+1)
        n_locations_thresholded_epsilon,n_locations_monitored_epsilon = validate_spcrec(path_to_files=path_to_files,s_range=s_range,
                                                                                        snapshots_matrix=snapshots_matrix_val,U=U,
                                                                                        sigma_noise=sigma_noise,sigma_threshold=sigma_threshold,
                                                                                        rois_thresholds=rois_thresholds,n_urban=n_urban,n_rural=n_rural,
                                                                                        rois_source=rois_source,
                                                                                        seed_random=seed_random,seed_noise=seed_noise
                                                                                        )
        n_locations_monitored_epsilon.replace(0.0,np.nan,inplace=True)
        n_locations_thresholded_epsilon.replace(0.0,np.nan,inplace=True)

        """ 
        Get epsilon hyperparameter. 
        The number of monitored locations obtained by SP-CREC should increase with signal sparsity
        Create a new file without the epsilon label
        """        
        s_range = np.arange(1,N+1,1)
        epsilon = {el:0 for el in s_range}
        df = n_locations_monitored_epsilon.loc[1,:].copy()
        df.dropna(inplace=True)
        epsilon[1] = df[::-1].idxmin()# [::-1] gives priority to lower epsilon values
        for s in s_range[1:]:
            if n_locations_monitored_epsilon.loc[s,:].isna().sum()!=n_locations_monitored_epsilon.shape[1]:
                df = n_locations_monitored_epsilon.loc[s,:].copy()
                df_prev = n_locations_monitored_epsilon.loc[s-1,:].copy()
                n_prev = df_prev.loc[epsilon[s-1]]
                mask = df>=n_prev
                epsilon[s] = df.loc[mask][::-1].idxmin()

        n_locations_monitored = pd.Series([n_locations_monitored_epsilon.loc[s,epsilon[s]] for s in s_range],index=s_range)
        n_locations_thresholded = pd.Series([n_locations_thresholded_epsilon.loc[s,epsilon[s]] for s in s_range],index=s_range)
        
        # rename solutions: remove epsilon
        for s in s_range:
            for p in np.arange(s,N+1,1):
                fname = f'{path_to_files}SPCREC_SensorsLocations_N[{N}]_S[{s}]_epsilon[{epsilon[s]:.2f}]_nROIS[{n_regions}]_nUrban[{n_urban}]_nRural[{n_rural}]_RandomSeed[{seed_random}]_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}]_nSensors[{p}].pkl'
                try:
                    with open(fname,'rb') as f:
                        locations_monitored_spcrec = np.sort(pickle.load(f)[0])
                    
                    fname = f'{path_to_files}SPCREC_SensorsLocations_N[{N}]_S[{s}]_nROIS[{n_regions}]_nUrban[{n_urban}]_nRural[{n_rural}]_RandomSeed[{seed_random}]_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}]_nSensors[{p}].pkl'
                    with open(fname,'wb') as f:
                        pickle.dump(locations_monitored_spcrec,f,protocol=pickle.HIGHEST_PROTOCOL)
                except:
                    continue

        """ Continue SP-CREC for a basis (s) adding more locations to fulfill constraints on validation set """
        s_range = np.arange(1,N+1,1)
        for signal_sparsity in s_range:
            n_locations_monitored_init = int(n_locations_monitored.loc[signal_sparsity])
            n_locations_thresholded_init = int(n_locations_thresholded.loc[signal_sparsity])
            print(f'\nBasis sparsity: {signal_sparsity}\nepsilon: {epsilon[signal_sparsity]}\nNumber of thresholded locations: {n_locations_thresholded_init}\nNumber of monitored locations: {n_locations_monitored_init}')
            if n_locations_monitored_init !=0:
                sizing_and_placement_autovalidated(U=U,signal_sparsity=signal_sparsity,epsilon=0.09,
                                                   rois_thresholds=rois_thresholds,n_urban=n_urban,n_rural=n_rural,
                                                   sigma_noise=sigma_noise,seed_noise=seed_noise,
                                                   n_locations_monitored_init=n_locations_monitored_init,snapshots_matrix=snapshots_matrix_val,
                                                   sigma_threshold=sigma_threshold,rois_source=rois_source,seed_random=seed_random,
                                                   path_to_files=path_to_files,save_results=False)
                
    elif execute == 'hyperparameters_selection':
        
        path_to_files = f'{results_path}AirPollution/Validated/'
        s_range = np.arange(1,N+1,1)
        p = pd.Series(np.zeros(shape=s_range.shape),index=s_range)
        for s in s_range:
            print(f'Loading files for s: {s}')
            for n_locations_monitored in np.arange(s,N+1,1):
                fname = f'{path_to_files}SPCREC_SensorsLocations_Validated_N[{N}]_S[{s}]_nROIS[{n_regions}]_nUrban[{n_urban}]_nRural[{n_rural}]_RandomSeed[{seed_random}]_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}]_nSensors[{n_locations_monitored}].pkl'
                try:
                    with open(fname,'rb') as f:
                        locations_monitored_spcrec = np.sort(pickle.load(f)[0])
                    p.loc[s] = len(locations_monitored_spcrec)
                    print(f'Loaded files for s: {s}, p: {locations_monitored_spcrec}')
                except:
                    try:
                        with open(fname,'rb') as f:
                            locations_monitored_spcrec = np.sort(pickle.load(f))
                        p.loc[s] = len(locations_monitored_spcrec)
                    except:
                        pass
        if sigma_noise == 0.75*rois_thresholds['urban'] and (n_urban==24 and n_rural==24):
            s_load = [20,22,25,27,30,33,34,40,45,48]
            s_show = [20,22,25,27,30,33,35,40,45,48]
            xmin = 20
            xmax = N
            ymin = 42
            ymax = 48+1
        elif sigma_noise == 0.75*rois_thresholds['urban'] and (n_urban == 12 and n_rural == 36):
            s_load = [17,20,22,24,25,30,35,40,46,48]
            s_show = [17,20,22,24,25,30,35,40,45,48]
            xmin = 15
            xmax = N
            ymin = 40
            ymax = 48+1
        
        elif sigma_noise == 0.75*rois_thresholds['urban'] and (n_urban == 36 and n_rural == 12):
            s_load = [23,25,30,34,35,37,40,45,48]
            s_show = [23,25,30,34,35,37,40,45,48]
            xmin = 20
            xmax = N+1
            ymin = 46
            ymax = 48+0.5

        elif sigma_noise == 0.25*rois_thresholds['urban'] and (n_urban == 24 and n_rural == 24):
            s_load = [20,22,25,30,35,42,45,48]
            s_show = [20,22,25,30,35,40,45,48]
            xmin = 16
            xmax = N+1
            ymin = 45
            ymax = 48+0.5

        plots = Figures(save_path=results_path,marker_size=1,
                        figx=3.5,figy=2.5,
                        fs_label=8,fs_ticks=9,fs_legend=7,fs_title=10,
                        show_plots=True)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(s_show,[p[s] for s in s_load],color='k')
        ax.set_xticks([i for i in s_show if i%5==0])#ax.set_xticks(s_show)
        ax.set_xticklabels(int(i) for i in ax.get_xticks())
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        ax.set_xlabel('Number of retained SVD modes')
        ax.set_ylabel('$|\mathcal{S}|$')
        fig.tight_layout()
        plt.show()

    elif execute == 'TraceMin_sensor_placement':
        """ 
        Alternative sensor placement algorithm
        Solve Trace minimization for increasing number of sensors
        until the solution meets coordinate error variance threshold
        """
        print(f'\n\nSensor placement that minimize trace of covariance matrix for a fixed number of sensors')
                
        signal_sparsity = 22 
        p_break = 45 #-1
        trace_min_iterative(snapshots_matrix=snapshots_matrix_val,U=U,signal_sparsity=signal_sparsity,
                            sigma_threshold=sigma_threshold,rois_threshold=rois_thresholds,
                            n_urban=n_urban,n_rural=n_rural,seed_random=seed_random,
                            seed_noise=seed_noise,p_break=p_break,algorithm='JB_trace')
        
        iterate_over_rois = False
        if iterate_over_rois:
            """
            Alternative method for deploying sensors over the network.
            Minimizes the Trace while iterating over different ROIs
            """
            s = 30
            cumulative_energy = np.cumsum(sing_vals)/np.sum(sing_vals)
            trace_min_iterative_rois(snapshots_matrix=snapshots_matrix_train,snapshots_matrix_val=snapshots_matrix_val,
                                cumulative_energy_basis=cumulative_energy[s],
                                rois_source=rois_source,rois_thresholds=rois_thresholds,
                                n_urban=n_urban,n_rural=n_rural,seed_random=seed_random,
                                sigma_noise=sigma_noise,
                                start_larger_threshold=False,force_n=-1,algorithm='JB_trace')
            
    elif execute == 'compare_reconstructions':
        # Both SP-CREC and TraceMin select and optimal number of locations to be monitored and meeting the thresholds
        if sigma_noise == 0.75*rois_thresholds['urban'] and (n_urban == 24 and n_rural == 24):
            signal_sparsity = 30
            p_spcrec = 42
            p_tracemin = 48
        elif sigma_noise == 0.75*rois_thresholds['urban'] and (n_urban == 12 and n_rural == 36):
            signal_sparsity = 22
            p_spcrec = 40
            p_tracemin = 43
        elif sigma_noise == 0.75*rois_thresholds['urban'] and (n_urban == 36 and n_rural == 12):
            signal_sparsity = 35
            p_spcrec = 46
            p_tracemin = 48
        elif sigma_noise == 0.25*rois_thresholds['urban'] and (n_urban == 24 and n_rural == 24):
            signal_sparsity = 22
            p_spcrec = 45
            p_tracemin = 46
        """ 
        Add noise to signal
        """
        snapshots_matrix = snapshots_matrix_test.copy()
        rng = np.random.default_rng(seed=seed_noise)
        Z = rng.standard_normal(size=snapshots_matrix.shape)
        snapshots_matrix_noisy = snapshots_matrix + sigma_noise*Z

        """ 
        Load monitored locations obtained by SPCREC algorithm 
        """
        path_to_files = f'{results_path}AirPollution/Validated/'

        if rois_source == 'random':
            fname_spcrec = f'{path_to_files}SPCREC_SensorsLocations_Validated_N[{N}]_S[{signal_sparsity}]_nROIS[{n_regions}]_nUrban[{n_urban}]_nRural[{n_rural}]_RandomSeed[{seed_random}]_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}]_nSensors[{p_spcrec}].pkl'
            fname_tracemin_optimal = f'{path_to_files}TraceMin_SensorsLocations_N[{N}]_S[{signal_sparsity}]_nROIS[{n_regions}]_nUrban[{n_urban}]_nRural[{n_rural}]_RandomSeed[{seed_random}]_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}]_nSensors[{p_tracemin}].pkl'
            fname_tracemin_fail = f'{path_to_files}TraceMin_SensorsLocations_N[{N}]_S[{signal_sparsity}]_nROIS[{n_regions}]_nUrban[{n_urban}]_nRural[{n_rural}]_RandomSeed[{seed_random}]_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}]_nSensors[{p_spcrec}].pkl'
        else:
            fname_spcrec = f'{path_to_files}SPCREC_SensorsLocations_Validated_N[{N}]_S[{signal_sparsity}]_nROIS[{n_regions}]_nUrban[{n_urban}]_nRural[{n_rural}]_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}]_nSensors[{p_spcrec}].pkl'
            fname_tracemin_optimal = f'{path_to_files}TraceMin_SensorsLocations_N[{N}]_S[{signal_sparsity}]_nROIS[{n_regions}]_nUrban[{n_urban}]_nRural[{n_rural}]_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}]_nSensors[{p_tracemin}].pkl'
            fname_tracemin_optimal = f'{path_to_files}TraceMin_SensorsLocations_N[{N}]_S[{signal_sparsity}]_nROIS[{n_regions}]_nUrban[{n_urban}]_nRural[{n_rural}]_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}]_nSensors[{p_spcrec}].pkl'
        
        with open(fname_spcrec,'rb') as f:
            locations_monitored_spcrec = np.sort(pickle.load(f)[0])
        with open(fname_tracemin_optimal,'rb') as f:
            locations_monitored_tracemin_optimal = np.sort(pickle.load(f))
        with open(fname_tracemin_fail,'rb') as f:
            locations_monitored_tracemin_fail = np.sort(pickle.load(f))
        

        C_spcrec = np.identity(N)[locations_monitored_spcrec,:]
        C_tracemin_optimal = np.identity(N)[locations_monitored_tracemin_optimal,:]
        C_tracemin_fail = np.identity(N)[locations_monitored_tracemin_fail,:]

        """
        Measure noisy snapshots matrix
        """
        snapshots_matrix_noisy_measured_spcrec = C_spcrec@snapshots_matrix_noisy
        snapshots_matrix_noisy_measured_tracemin_optimal = C_tracemin_optimal@snapshots_matrix_noisy
        snapshots_matrix_noisy_measured_tracemin_fail = C_tracemin_fail@snapshots_matrix_noisy
        
        """ 
        Reconstruct signal from measurements
            - Measure snapshots matrix
            - Compute error variance on results
        """
        Psi = U[:,:signal_sparsity]
        # SP-CREC
        beta_hat = np.linalg.pinv(C_spcrec@Psi)@snapshots_matrix_noisy_measured_spcrec
        snapshots_matrix_predicted = Psi@beta_hat
        error = snapshots_matrix - snapshots_matrix_predicted
        error_var_spcrec = error.var(axis=1,ddof=0)
        #TraceMin optimal
        beta_hat = np.linalg.pinv(C_tracemin_optimal@Psi)@snapshots_matrix_noisy_measured_tracemin_optimal
        snapshots_matrix_predicted = Psi@beta_hat
        error = snapshots_matrix - snapshots_matrix_predicted
        error_var_tracemin_optimal = error.var(axis=1,ddof=0)
        # TraceMin fail
        beta_hat = np.linalg.pinv(C_tracemin_fail@Psi)@snapshots_matrix_noisy_measured_tracemin_fail
        snapshots_matrix_predicted = Psi@beta_hat
        error = snapshots_matrix - snapshots_matrix_predicted
        error_var_tracemin_fail = error.var(axis=1,ddof=0)

        n_locations_thresholded_spcrec = (error_var_spcrec<=sigma_threshold**2).sum()
        n_locations_thresholded_tracemin_optimal = (error_var_tracemin_optimal<=sigma_threshold**2).sum()
        n_locations_thresholded_tracemin_fail = (error_var_tracemin_fail<=sigma_threshold**2).sum()
        

        """ Show figure with comparison """
        
        # sort ROIs to show them aggregated together
        roi_idx = {el:[] for el in rois_thresholds.values()}
        for i in roi_idx:
            roi_idx[i] = np.array([np.array(j) for j in np.arange(0,N,1) if sigma_threshold[j] == i])
        error_var_spcrec_sorted = np.concatenate([error_var_spcrec[idx] for idx in roi_idx.values()])
        error_var_tracemin_optimal_sorted = np.concatenate([error_var_tracemin_optimal[idx] for idx in roi_idx.values()])
        error_var_tracemin_fail_sorted = np.concatenate([error_var_tracemin_fail[idx] for idx in roi_idx.values()])

        print(f'SP-CREC\n- Number of monitored locations: {p_spcrec}\n- Number of locations below threshold: {n_locations_thresholded_spcrec}')
        print(f'- Number of locations in ROI urban: {np.isin(locations_monitored_spcrec,roi_idx[rois_thresholds["urban"]]).sum()}')
        print(f'- Number of locations in ROI rural: {np.isin(locations_monitored_spcrec,roi_idx[rois_thresholds["rural"]]).sum()}')

        print(f'TraceMin\n- Number of monitored locations: {p_tracemin}\n- Number of locations below threshold: {n_locations_thresholded_tracemin_optimal} ')
        print(f'TraceMin with same number of locations as SP-CREC\n- Number of monitored locations: {p_spcrec}\n- Number of locations below threshold: {n_locations_thresholded_tracemin_fail}')

        plots = Figures(save_path=results_path,marker_size=1,
                        figx=3.5,figy=2.5,
                        fs_label=9,fs_ticks=9,fs_legend=8,fs_title=10,
                        show_plots=True)
        thresholds = [i for i in rois_thresholds.values()]
        
        show_traceMin_optimal = False
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        if show_traceMin_optimal:
            ax.plot(error_var_spcrec_sorted,color='#1a5276',label=rf'SP-CREC,   $|\mathcal{{S}}|={p_spcrec}$')
            ax.plot(error_var_tracemin_optimal_sorted,color='orange',label=rf'Trace min, $|\mathcal{{S}}|={p_tracemin}$')
            ax.plot(error_var_tracemin_fail_sorted,color='orange',linestyle='--',label=rf'Trace min, $|\mathcal{{S}}|={p_spcrec}$')
        else:
            ax.plot(error_var_spcrec_sorted,color='#1a5276',label=rf'SP-CREC')
            ax.plot(error_var_tracemin_fail_sorted,color='orange',label=rf'Trace min')

        for i in rois_thresholds:
            if i=='urban':
                ax.hlines(y=rois_thresholds[i]**2,xmin=0,xmax=n_urban-1,color='k',linestyles='--')
            elif i=='rural':
                ax.hlines(y=rois_thresholds[i]**2,xmin=n_urban,xmax=n_urban+n_rural-1,color='k',linestyles='--',label='Design threshold')
        if show_traceMin_optimal:
            ax.legend(loc='center',ncol=2,framealpha=1.,bbox_to_anchor=(0.5,1.2))
        else:
            ax.legend(loc='upper left',ncol=1,framealpha=1.)
        #ax.legend(loc='upper center',ncol=2,framealpha=1.)
        xrange = np.concatenate((np.arange(-1,N,10),[N-1]))
        xrange[0] = 0
        ax.set_xticks(xrange)
        ax.set_xticklabels([int(i+1) for i in ax.get_xticks()])
        ax.set_xlim(0,N-1)
        ax.set_xlabel('Location index')
        yrange = np.arange(0,450+100,100)
        ax.set_yticks(yrange)
        ax.set_yticklabels([int(i) for i in ax.get_yticks()])
        ax.set_ylim(0,np.max(yrange))
        ax.set_ylabel(rf'Error variance $(\mu g/m^3)^2$')
        fig.tight_layout()
        
        save_fig = True
        if save_fig:
            if rois_source == 'random':
                fname = f'{path_to_files}SPCREC_vs_TraceMin_N[{N}]_S[{signal_sparsity}]_nROIS[{n_regions}]_nUrban[{n_urban}]_nRural[{n_rural}]_RandomSeed[{seed_random}]_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}].png'
            else:
                fname = f'{path_to_files}SPCREC_vs_TraceMin_N[{N}]_S[{signal_sparsity}]_nROIS[{n_regions}]_nUrban[{n_urban}]_nRural[{n_rural}]_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}].png'
            fig.savefig(fname,dpi=300,format='png',bbox_inches='tight')
            print(f'Figure saved at {fname}')

    elif execute == 'compare_basis_size':
        rois_size = {'n_urban': [12,24,36], 'n_rural': [36,24,12]}
        colors = {12:'#1a5276',24:'orange',36:'#b03a2e'}

        path_to_files = f'{results_path}AirPollution/Validated/'
        s_range = np.arange(1,N+1,1)
        p = pd.Series(np.zeros(shape=s_range.shape),index=s_range)

        plots = Figures(save_path=results_path,marker_size=1,
                        figx=3.5,figy=2.5,
                        fs_label=8,fs_ticks=9,fs_legend=7,fs_title=10,
                        show_plots=True)
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for n_urban,n_rural in zip(rois_size['n_urban'],rois_size['n_rural']):
            for s in s_range:
                print(f'Loading files for s: {s}')
                for n_locations_monitored in np.arange(s,N+1,1):
                    fname = f'{path_to_files}SPCREC_SensorsLocations_Validated_N[{N}]_S[{s}]_nROIS[{n_regions}]_nUrban[{n_urban}]_nRural[{n_rural}]_RandomSeed[{seed_random}]_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}]_nSensors[{n_locations_monitored}].pkl'
                    try:
                        with open(fname,'rb') as f:
                            locations_monitored_spcrec = np.sort(pickle.load(f)[0])
                        p.loc[s] = len(locations_monitored_spcrec)
                        print(f'Loaded files for s: {s}, p: {locations_monitored_spcrec}')
                    except:
                        try:
                            with open(fname,'rb') as f:
                                locations_monitored_spcrec = np.sort(pickle.load(f))
                            p.loc[s] = len(locations_monitored_spcrec)
                        except:
                            pass
            if n_urban==24 and n_rural==24:
                s_load = [20,22,25,27,30,33,34,40,45,48]
                s_show = [20,22,25,27,30,33,35,40,45,48]
                xmin = 20
                xmax = N
                ymin = 42
                ymax = 48+1
            elif n_urban == 12 and n_rural == 36:
                s_load = [17,20,22,24,25,30,35,40,46,48]
                s_show = [17,20,22,24,25,30,35,40,45,48]
                xmin = 15
                xmax = N
                ymin = 40
                ymax = 48+1
            elif n_urban == 36 and n_rural == 12:
                s_load = [23,25,30,34,35,37,40,45,48]
                s_show = [23,25,30,34,35,37,40,45,48]
                xmin = 20
                xmax = N+1
                ymin = 46
                ymax = 48+0.5

        
            ax.plot(s_show,[p[s] for s in s_load],color=colors[n_urban],label=rf'$|\mathcal{{R}}_1|={n_urban}$, $|\mathcal{{R}}_2|={n_rural}$')
        xrange = np.concatenate((np.arange(15,45+5,5),[48]))
        ax.set_xticks([i for i in xrange])#ax.set_xticks(s_show)
        ax.set_xticklabels(int(i) for i in ax.get_xticks())
        ax.set_xlim(xrange[0],xrange[-1])
        yrange = np.arange(0,48+4,4)
        if yrange[-1]>N:
            yrange[-1]=N
        ax.set_yticks(yrange)
        ax.set_yticklabels([int(i) for i in ax.get_yticks()])
        ax.set_ylim(yrange[0],yrange[-1])
        ax.set_xlabel('Number of retained SVD modes')
        ax.set_ylabel('$|\mathcal{S}|$')
        #ax.legend(loc='center',ncol=2,framealpha=0.5,bbox_to_anchor=(0.5,1.1))
        ax.legend(loc='lower right',ncol=1,framealpha=1.)
        fig.tight_layout()

        save_fig = True
        if save_fig:
            if rois_source == 'random':
                fname = f'{path_to_files}SPCREC_nSensors_vs_nModes_nROIS[{n_regions}]_nUrban{[i for i in rois_size["n_urban"]]}_nRural{[i for i in rois_size["n_rural"]]}_RandomSeed[{seed_random}]_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}].png'
            else:
                fname = f'{path_to_files}SPCREC_nSensors_vs_nModes_nROIS[{n_regions}]_nUrban{[i for i in rois_size["n_urban"]]}_nRural{[i for i in rois_size["n_rural"]]}_UrbanThreshold[{rois_thresholds["urban"]}]_RuralThreshold[{rois_thresholds["rural"]}]_SigmaNoise[{sigma_noise}].png'
            fig.savefig(fname,dpi=300,format='png',bbox_inches='tight')
            print(f'Figure saved at {fname}')


        
#%%
    
    elif execute=='SPCREC_sensor_placement':
        print(f'\n\nAlgorithm for heterogeneous constraints. Different design threshold will be applied to different Regions of Interest (ROIs)')
                
        """ 
        Specify signal's sparsity
         Energy | sparsity
         0.9    | 33+1
         -----------------
         SVHT: 12 (no +1!)
        """
        signal_sparsity = 30 # add one because python cuts off last entry in range
        Psi = U[:,:signal_sparsity]
        Psi_bar = U[:,signal_sparsity:]
        n = Psi.shape[0]
        thresholds = (sigma_threshold**2).copy()
        print(f'Low-rank decomposition. Basis shape: {Psi.shape}')
        
        """ 
        Set Regions of Interest (ROIs) thresholds
            The rows of the snapshots matrix are previously sorted based on ROIs.
            The ROIs are obtained from a file loaded within the dataset class
        """
        path_to_files = f'{results_path}AirPollution/'
        variance_nsvd = np.diag((1/snapshots_matrix_val.shape[1])*Psi_bar@Psi_bar.T@snapshots_matrix_val@snapshots_matrix_val.T@Psi_bar@Psi_bar.T)
        sigma_th2 = thresholds - variance_nsvd

        sigma_noise_list = sigma_noise*np.asarray(len(sigma_threshold)*[1])
        variance_ratio = (sigma_th2)/(sigma_noise_list**2)#sigma_th^2 / sigma_noise^2
        
        algorithm = 'NetworkPlanning_iterative'
        sensor_placement = sp.SensorPlacement(algorithm, n, signal_sparsity,
                                              n_refst=n,n_lcs=0,n_unmonitored=0)
        
        """
        Solve SPCREC sensor placement problem
        Algorithm parameters
            Mind the parameters. epsilon=1e-1 can make the problem infeasible at next iteration as some locations are assigned to be unmonitored.
            As epsilon increases less stations are assigned to be monitored, or it is easier to assign an unmonitored locations.
        
            epsilon=1e-1 tends to fail at multiple thresholds as threshold <1.1
        """
        
        epsilon = 9e-2#[10e-2,...,1e-2,8e-3?]
        n_it = 20
        h_prev = np.zeros(n)
        weights = 1/(h_prev+epsilon)
        locations_monitored = []
        locations_unmonitored = []
        
        locations = networkPlanning_iterative(sensor_placement,n,Psi,variance_ratio,
                                              epsilon=epsilon,h_prev=h_prev,weights=weights,n_it=n_it,
                                              locations_monitored=locations_monitored,locations_unmonitored=locations_unmonitored)
        
        """ 
        Compute the solution obtained for the coordinate error variance
        check if the solution meets the thresholds
        """
        sensor_placement.locations = [[],np.sort(locations[0]),np.sort(locations[1])]
        sensor_placement.C_matrix()
        C = sensor_placement.C[1]
        
        n_locations_monitored = len(locations[0])
        n_locations_unmonitored = len(locations[1])
        print(f'Network design results:\n- Total number of potential locations: {n}\n- basis sparsity: {signal_sparsity}\n- Number of monitored locations: {n_locations_monitored}\n- Number of unmonitored locations: {n_locations_unmonitored}')
        
        # evaluate if algorithm was successful analitically
        coordinate_error_variance = (sigma_noise**2)*np.diag(Psi@np.linalg.inv(Psi.T@C.T@C@Psi)@Psi.T)
        
        if (coordinate_error_variance<sigma_threshold**2).sum() == n:
                print(f'Success at computing coordinate error variance at every location')
                success = 1
        else:
                warnings.warn('Some locations failed')

        # save results
        if success:        
            fname = f'{results_path}SPCREC_SensorsLocations_N{n}_S{signal_sparsity}_nROIS{n_regions}_SigmaThresholds{[i for i in np.unique(sigma_threshold)]}_SigmaNoise{[i for i in np.unique(sigma_noise)]}_nSensors{n_locations_monitored}.pkl'
            with open(fname,'wb') as f:
                pickle.dump(locations,f,protocol=pickle.HIGHEST_PROTOCOL)
            print(f'File saved in {fname}')            
        sys.exit()
    #%%
    # Reconstruct signal using measurements at certain locations and compare with actual values
    elif execute == 'reconstruct_signal':
        print('\nReconstructing signal from sparse measurements.\nTwo algorithms are used:\n- Network design\n- Joshi-Boyd (D-optimal) sensor selection.')
        """ Low-rank basis """
        snapshots_matrix_train = X_train.to_numpy().T
        snapshots_matrix_train_centered = snapshots_matrix_train - snapshots_matrix_train.mean(axis=1)[:,None]
        #snapshots_matrix_val = X_val.to_numpy().T
        #snapshots_matrix_val_centered = snapshots_matrix_val - snapshots_matrix_train.mean(axis=1)[:,None]
        snapshots_matrix_test = X_test.to_numpy().T
        snapshots_matrix_test_centered = snapshots_matrix_test - snapshots_matrix_train.mean(axis=1)[:,None]
        U,sing_vals,Vt = np.linalg.svd(snapshots_matrix_train,full_matrices=False)
        n = U.shape[0]
        print(f'Training snapshots matrix has dimensions {snapshots_matrix_train_centered.shape}.\nLeft singular vectors matrix has dimensions {U.shape}\nRight singular vectors matrix has dimensions [{Vt.shape}]\nNumber of singular values: {sing_vals.shape}')
        # specify signal sparsity and network parameters
        signal_sparsity = 34
        Psi = U[:,:signal_sparsity]
        Psi_ort = U[:,signal_sparsity:]

        """ ROIs used when solving SPCREC """
        dataset.rois = dataset.stations_types
        rois_thresholds = {'urban':9,'suburban':9,'rural':18}
        dataset.rois['thresholds'] = [rois_thresholds[i] for i in dataset.rois['Area']]
        sigma_threshold = dataset.rois['thresholds'].to_numpy()
        n_regions = len(np.unique(sigma_threshold))
        print(f'- Number of ROIs: {n_regions}\n- ROIs sigma thresholds: {[i for i in np.unique(sigma_threshold)]}')
        
        """ 
        Add noise to signal
            - Use signal_noise value used when solving SPCREC
        """
        sigma_noise = 0.75*9
        snapshots_matrix = Psi@Psi.T@snapshots_matrix_train
        seed = 0
        rng = np.random.default_rng(seed=seed)
        Z = rng.standard_normal(size=snapshots_matrix.shape)
        snapshots_matrix_noisy = snapshots_matrix + sigma_noise*Z

        """ Load monitored locations obtained by SPCREC algorithm """
        n_locations_monitored = 44
        fname = f'{results_path}SPCREC_SensorsLocations_N{n}_S{signal_sparsity}_nROIS{n_regions}_SigmaThresholds{[i for i in np.unique(sigma_threshold)]}_SigmaNoise{[i for i in np.unique(sigma_noise)]}_nSensors{n_locations_monitored}.pkl'
        with open(fname,'rb') as f:
            locations_monitored_spcrec = np.sort(pickle.load(f)[0])
        C_spcrec = np.identity(n)[locations_monitored_spcrec,:]

        """ 
        Load monitored locations obtained by alternative algorithm
            - Trace Minimization
        """       
        fname = f'{results_path}TraceMin_SensorsLocations_N{n}_S{signal_sparsity}_nSensors{n_locations_monitored}.pkl'
        with open(fname,'rb') as f:
            locations_monitored_tracemin = np.sort(pickle.load(f))
        C_tracemin = np.identity(n)[locations_monitored_tracemin,:]
        print(f'Trace minimization algorithm matches {len([i for i in locations_monitored_tracemin if i in locations_monitored_spcrec])} locations out of {n_locations_monitored} potential locations')

        """ 
        Reconstruct signal from measurements
            - Measure snapshots matrix
            - Compute error variance on results
        """
        #spcrec
        snapshots_matrix_noisy_measured = C_spcrec@snapshots_matrix_noisy
        beta_hat = np.linalg.pinv(C_spcrec@Psi)@snapshots_matrix_noisy_measured
        snapshots_matrix_predicted = Psi@beta_hat
        error = snapshots_matrix - snapshots_matrix_predicted
        coordinate_error_variance = (sigma_noise**2)*np.diag(Psi@np.linalg.inv(Psi.T@C_spcrec.T@C_spcrec@Psi)@Psi.T)
        error_var_spcrec = error.var(axis=1,ddof=0)
        
        # trace min        
        snapshots_matrix_noisy_measured = C_tracemin@snapshots_matrix_noisy
        beta_hat = np.linalg.pinv(C_tracemin@Psi)@snapshots_matrix_noisy_measured
        snapshots_matrix_predicted = Psi@beta_hat
        error = snapshots_matrix - snapshots_matrix_predicted
        error_var_tracemin = error.var(axis=1,ddof=0)

        """
        Draw figure comparing results
            - Show coordinate error variance
            - Show sigma_thresholds
        """
        plots = Figures(save_path=results_path,marker_size=1,
                figx=3.5,figy=2.5,
                fs_label=9,fs_ticks=9,fs_legend=8,fs_title=10,
                show_plots=True)
        thresholds = [i for i in np.unique(sigma_threshold)]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(error_var_spcrec,color='#1a5276',label='SP-CREC')
        ax.plot(error_var_tracemin,color='orange',label='Trace min')
        for i in range(len(thresholds)):
            ax.hlines(y=thresholds[i]**2,xmin=np.min(np.where(sigma_threshold==thresholds[i])),xmax=np.max(np.where(sigma_threshold==thresholds[i])),color='k',linestyles='--',label=rf'$\sigma_{{th}}^2$={thresholds[i]:.1f}$^2$')
        ax.legend(loc='center',ncol=2,framealpha=0.5,bbox_to_anchor=(0.5,1.1))
        xrange = np.arange(-1,n,10)
        xrange[0] = 0
        ax.set_xticks(xrange)
        ax.set_xticklabels([i+1 for i in ax.get_xticks()])
        ax.set_xlim(0,n)
        ax.set_ylabel(rf'Coordinate error variance $(\mu g/m^3)^2$')
        ax.set_xlabel('Location index')
        fig.tight_layout()
        save_fig = True
        if save_fig:
            fname = f'{results_path}SPCREC_vs_TraceMin_N{n}_S{signal_sparsity}_nROIS{n_regions}_SigmaThresholds{thresholds}_SigmaNoise{sigma_noise}_nSensors{n_locations_monitored}.png'
            fig.savefig(fname,dpi=300,format='png',bbox_inches='tight')
            print(f'Figure saved at {fname}')



    # %%
        snapshtots_matrix = snapshots_matrix_train
        U,sing_vals,Vt = np.linalg.svd(snapshtots_matrix,full_matrices=False)
        E = np.zeros(shape=len(sing_vals))
        for i in np.arange(0,len(sing_vals)):
            Ur = U[:,:i+1]
            snapshots_matrix_proj = Ur@Ur.T@snapshots_matrix
            error = snapshots_matrix_proj - snapshots_matrix
            E[i] = np.linalg.norm(error,ord='fro')/np.linalg.norm(snapshots_matrix,ord='fro')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(E,color='k')
        ax.set_ylabel(rf'Fractional error $E_{{F}}$')
        ax.set_xlabel('Sparsity')
        ax.set_xticklabels([int(i+1) for i in ax.get_xticks()])
        fig.tight_layout()
        plt.show()