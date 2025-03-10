#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 17:21:04 2023

@author: jparedes
"""
import os
import sys
import time
import argparse
import warnings
from abc import ABC,abstractmethod
import netCDF4
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import pickle
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import dask
import dask.array as da
import sensor_placement as sp
from cvxopt import matrix, solvers, spmatrix, sparse, spdiag

#%% Script parameters
parser = argparse.ArgumentParser(prog='IRNet-sensorPlacement',
                                 description='Iterative Reweighted Network design.',
                                 epilog='---')
# action to perform-
parser.add_argument('--design_large_networks',help='IRNet algorithm for monitoring network of large size',action='store_true',default=True)
parser.add_argument('-n','--n_locations',help='Number of potential locations. Network size.',type=int,default=691150,required=False)
parser.add_argument('-s','--signal_sparsity',help='Signal sparsity for basis cutoff',type=int,default=150,required=False)
# 
parser.add_argument('-br','--batch_rows',help='Number of rows per batch',type=int,default=10,required=False)
parser.add_argument('-bc','--batch_cols',help='Number of columns per batch',type=int,default=10,required=False)
parser.add_argument('--pre_computed_batches',help='Load results of pre-computed batches',action='store_true',default=False)
parser.add_argument('-b_init','--b_init',help='Initial batch for large scale algorithm',type=int,default=1,required=False)

# network design parameters
parser.add_argument('-e','--epsilon',help='Reweighting epsilon parameter',type=float,default=5e-2,required=False)
parser.add_argument('-n_it','--num_it',help='Number of iterations for updating monitored set',type=int,default=20,required=False)
#parser.add_argument('-vtr','--variance_threshold_ratio',help='Maximum error variance threshold for network design',type=float,default=1.1,required=False)
args = parser.parse_args()

#%% Classes
# dataset class
class LoadDataSet(ABC):
    @abstractmethod
    def load(self,file_path,fname,**kwargs):
        raise NotImplementedError
    
class LoadGRP(LoadDataSet):
    def load(self,file_path:str,fname:str='sst.mon.mean.nc'):
        rootgrp =  netCDF4.Dataset(file_path+fname,'r')
        print(f'List of variables: {rootgrp.variables.keys()}')
        # total number of pixels and valid pixels with measurements
        return rootgrp
class LoadDataFrame(LoadDataSet):
    def load(self,file_path:str,fname:str):
        df = np.load(f'{self.files_path}{self.fname}').T
        return df
    
# image preprocessing class
class ImagePreprocessing(ABC):
    @abstractmethod
    def filter(self,img,**kwargs):
        raise NotImplementedError
    def get_coordinates_range(self,**kwargs):
        raise NotImplementedError
    
class Windowing(ImagePreprocessing):
    """Image windowing. Maintain values within specified coordinates"""
    def filter(self,image,**kwargs,)->np.ndarray:
        lat_range,lon_range = kwargs['lat_range'],kwargs['lon_range']
        lon_min,lon_max = kwargs['lon_min'],kwargs['lon_max']
        lat_min,lat_max = kwargs['lat_min'],kwargs['lat_max']
        idx_window_lon = np.where(np.logical_and(lon_range>=lon_min,lon_range<=lon_max))[0]
        idx_window_lat = np.where(np.logical_and(lat_range>=lat_min,lat_range<=lat_max))[0]
        image_new = image[idx_window_lat[0]:idx_window_lat[-1]+1,idx_window_lon[0]:idx_window_lon[-1]+1]
        mask_new = np.isnan(image_new)
        return image_new,mask_new
    def get_coordinates_range(self,**kwargs)->np.array:
        """ return Coordinates range in the windowing interval """
        lat_range,lon_range = kwargs['lat_range'],kwargs['lon_range']
        lon_min,lon_max = kwargs['lon_min'],kwargs['lon_max']
        lat_min,lat_max = kwargs['lat_min'],kwargs['lat_max']
        idx_window_lon = np.where(np.logical_and(lon_range>=lon_min,lon_range<=lon_max))[0]
        idx_window_lat = np.where(np.logical_and(lat_range>=lat_min,lat_range<=lat_max))[0]
        lat_range_window = lat_range[idx_window_lat]
        lon_range_window = lon_range[idx_window_lon]
        return lat_range_window,lon_range_window

class Downsampling(ImagePreprocessing):
    """"Image downsampling to new resolution."""
    def filter(self,image:np.ndarray,scale_factor_x:float,scale_factor_y:float)->np.ndarray:
        img_new = cv2.resize(image,(0,0),fx=scale_factor_x,fy=scale_factor_y,interpolation=cv2.INTER_NEAREST)
        mask_new = np.isnan(img_new)
        return img_new,mask_new
class NoPreprocessing(ImagePreprocessing):
    """ Simple No preprocessing class"""
    def filter(self,image:np.ndarray,mask:np.ndarray):
        return image,mask
# sampling class: used for defining how many points to load from image
class SubSampling(ABC):
    @abstractmethod
    def sample(self,arr,**kwargs):
        raise NotImplementedError
class RandomSubSampling(SubSampling):
    """Not using all points in image but sampling random points from image"""
    def sample(self,arr,**kwargs)->np.array:
        n_points_sample,random_seed_subsampling = kwargs['n_points_sample'],kwargs['random_seed_subsampling']
        n = len(arr)
        indices = np.arange(0,n,1)
        rng = np.random.default_rng(seed=random_seed_subsampling)    
        indices_perm = rng.permutation(indices)
        indices_sampled = np.sort(indices_perm[:n_points_sample])
        arr_subsampled = arr[indices_sampled]
        return arr_subsampled, indices_sampled

class NoSubSampling(SubSampling):
    """Select all points in image. Return all indices"""
    def sample(self,arr,**kwargs):
        return arr,np.arange(0,len(arr),1)

# dataset class 
class Dataset():
    def __init__(self,files_path:str,loader,preprocessor,sampler,fname:str='SST_month.parquet',n_locations:int=6624):
        self.files_path = files_path
        self._loader = loader
        self._preprocessor = preprocessor
        self._sampler = sampler
        self.fname = fname
        self.n_locations = n_locations

    def get_coordinates(self,rootgrp):
        lat_range = np.array([rootgrp.variables['lat'][i].data.item() for i in range(rootgrp.variables['lat'].shape[0])])
        lon_range = np.array([rootgrp.variables['lon'][i].data.item() for i in range(rootgrp.variables['lon'].shape[0])])
        return lat_range,lon_range

    def get_mask(self,rootgrp):
        mask = rootgrp.variables['sst'][0].mask
        return mask
    
    def get_pixels(self,rootgrp):
        n_samples = rootgrp.variables['sst'].shape[0]
        n_pixels = rootgrp.variables['sst'].shape[1]*rootgrp.variables['sst'].shape[2]   
        n_land = rootgrp.variables['sst'][0].mask.sum()
        return n_samples,n_pixels,n_land
    
    def create_dataframe(self,rootgrp,**kwargs)->pd.DataFrame:
        """
        Create dataframe from rootgrp dataset

        Args:
            rootgrp (_type_): netCDF4 files
            **kwargs must include corresponding image preprocessing parameters
        Returns:
            pd.DataFrame: _description_
        """
        
        # land_vals = -9.96921e+36
        # preprocess first sample
        data = rootgrp.variables['sst'][0].data
        mask = self.get_mask(rootgrp)
        data[mask] = np.nan
        data_new,mask_new = self._preprocessor.filter(data,**kwargs)
        print(f'Preprocessed images have dimensions {data_new.shape}. Mask of land values has dimensions: {mask_new.shape}')
        # get number of samples and potential locations in preprocessed dataset
        n_samples = rootgrp.variables['sst'].shape[0]
        n_pixels = data_new.shape[0]*data_new.shape[1]
        # reshape array of measurements and subsample some locations
        data_reshaped = np.reshape(data_new,newshape=n_pixels,order='F')
        data_reshaped = data_reshaped[~np.isnan(data_reshaped)]
        data_subsampled,indices_subsampled = self._sampler.sample(data_reshaped,**kwargs)
        n_locations = len(data_subsampled)
        # create snapshots matrix and store first sample
        print(f'Creating snapshots matrix for {n_locations} locations and {n_samples} samples')
        snapshots_matrix = np.zeros(shape=(n_locations,n_samples),dtype=np.float32)
        snapshots_matrix[:,0] = data_subsampled
        # store other samples in snapshots matrix
        for i in np.arange(1,n_samples,1):
            data = rootgrp.variables['sst'][i].data
            data[mask] = np.nan
            data_new,mask_new = self._preprocessor.filter(data,**kwargs)
            # drop pixels where there is land
            data_reshaped = np.reshape(data_new,newshape=n_pixels,order='F')
            data_reshaped = data_reshaped[~np.isnan(data_reshaped)]
            data_subsampled,_ = self._sampler.sample(data_reshaped,**kwargs)
            snapshots_matrix[:,i] = data_subsampled
        
        date_range = pd.date_range(start='1981-09-01 00:00:00',freq='M',periods=n_samples)
        idx_measurement = np.where(~np.reshape(mask_new,newshape=n_pixels,order='F'))[0]
        idx_land = np.where(np.reshape(mask_new,newshape=n_pixels,order='F'))[0]
        print(f'New dataset has {snapshots_matrix.shape[1]} measurements for {snapshots_matrix.shape[0]} locations.')
        df = pd.DataFrame(snapshots_matrix.T,dtype=np.float32,columns=indices_subsampled,index=date_range)
        df.columns = df.columns.astype(str)
        return df,idx_measurement,idx_land,data_new

# ROI classes: used for defining different design thresholds
class roi_generator(ABC):
    @abstractmethod
    def generate_rois(self,**kwargs):
        raise NotImplementedError

class CoordinateRoi1D(roi_generator):
    """ Regions of interest based on latitude and longitude coordinates """
    def generate_rois(self, **kwargs)->dict:
        # Original and preprocessed image. Using the same preprocessing method used for creating the dataset
        preprocessor = kwargs['preprocessor']
        image_preprocessed,_ = preprocessor.filter(**kwargs)
        # Coordinates range of preprocessed image
        lat_range_window,lon_range_window = preprocessor.get_coordinates_range(**kwargs)
        # thresholds for defining ROIs. Indicate min and max in pairs
        lat_thresholds_min, lat_thresholds_max = kwargs['lat_thresholds_min'],kwargs['lat_thresholds_max']
        lon_thresholds_min, lon_thresholds_max = kwargs['lon_thresholds_min'],kwargs['lon_thresholds_max']
        # iterate over coordiante thresholds
        n_regions = kwargs['n_regions']
        n_pixels = image_preprocessed.shape[0]*image_preprocessed.shape[1]
        roi_idx = {el:[] for el in np.arange(n_regions)}
        for i,lat_min,lat_max,lon_min,lon_max in zip(range(n_regions),lat_thresholds_min,lat_thresholds_max,lon_thresholds_min,lon_thresholds_max):
            idx_window_lon = np.where(np.logical_and(lon_range_window>=lon_min,lon_range_window<=lon_max))[0]
            idx_window_lat = np.where(np.logical_and(lat_range_window>=lat_min,lat_range_window<=lat_max))[0]
            region_mask = np.zeros_like(image_preprocessed,dtype=bool)
            region_mask[idx_window_lat[0]:idx_window_lat[-1]+1,idx_window_lon[0]:idx_window_lon[-1]+1] = True
            roi_idx[i] = np.where(np.reshape(region_mask,newshape=n_pixels,order='F'))[0]
        return roi_idx

class CoordinateRoi2D(roi_generator):
    """ Create regions of interests based on lat/lon coordinates on 2D image """
    def generate_rois(self,lat_range_img,lon_range_img,lat_thresholds_min,lat_thresholds_max,lon_thresholds_min,lon_thresholds_max,n_regions)->np.ndarray:
        if n_regions != len(lat_thresholds_min):
            raise ValueError(f'Total number of ROIs ({n_regions}) mismatches the specified thresholds for splitting the image')
        else:
            roi_matrix = np.zeros(shape=(len(lat_range_img),len(lon_range_img)),dtype=int)
            R = range(n_regions)
            for r,lat_min,lat_max,lon_min,lon_max in zip(R,lat_thresholds_min,lat_thresholds_max,lon_thresholds_min,lon_thresholds_max):
                idx_window_lat = np.where(np.logical_and(lat_range_img>=lat_min,lat_range_img<=lat_max))[0]
                idx_window_lon = np.where(np.logical_and(lon_range_img>=lon_min,lon_range_img<=lon_max))[0]
                roi_matrix[np.ix_(idx_window_lat,idx_window_lon)] = int(r)
            return roi_matrix
        
    
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

class ROI():
    """
    Region of interest (ROI) class. Select a generator from different roigenerator classes.
    Use as:
        roi = ROI(generator())
        roi.deine_ROIs(**kwargs)
    """
    def __init__(self,generator,design_ratio):
        self._generator = generator
        self.design_ratio = design_ratio
    def define_rois(self,**kwargs)->dict:
        self.roi_idx = self._generator.generate_rois(**kwargs)
        self.regions = {el:r for el,r in zip(np.sort(np.unique(self.roi_idx.astype(int))),self.design_ratio)}

# Reading and Writing file classes
class FileWriter(ABC):
    @abstractmethod
    def save(self,**kwargs):
        raise NotImplementedError
class WriteRandomROIRandomSubSampling(FileWriter):
    """ save locations file when the signal corresponds to random subsampling and was randomly splitted into differents ROIs"""
    def save(self,results_path,n,signal_sparsity,variance_threshold_ratio,n_locations_monitored,**kwargs):
        random_seed_roi = kwargs['random_seed_roi']
        random_seed_subsampling = kwargs['random_seed_subsampling']
        fname = f'{results_path}SensorsLocations_N{n}_S{signal_sparsity}_VarThreshold{variance_threshold_ratio}_nSensors{n_locations_monitored}_randomSeed{random_seed_roi}_randomSeedSubsampling{random_seed_subsampling}.pkl'
        with open(fname,'wb') as f:
            pickle.dump(locations[0],f,protocol=pickle.HIGHEST_PROTOCOL)
        print(f'File saved in {fname}')

class WriteLocationsMiniBatches(FileWriter):
    """ saves to file IRNet solutions """
    def save(self,results_path,locations,n,signal_sparsity,variance_threshold_ratio,n_locations_monitored,completed=True,iteration=0):
        if completed:
            fname = f'{results_path}SensorsLocations_N{n}_S{signal_sparsity}_VarThreshold{variance_threshold_ratio}_nSensors{n_locations_monitored}.pkl'
        else:
            fname = f'{results_path}SensorsLocations_N{n}_S{signal_sparsity}_VarThreshold{variance_threshold_ratio}_nSensors{n_locations_monitored}_iteration{iteration}.pkl'
        with open(fname,'wb') as f:
            pickle.dump(locations,f,protocol=pickle.HIGHEST_PROTOCOL)
        print(f'File saved in {fname}')

class WriteExecutionTime(FileWriter):
    """ saves execution times to a file"""
    def save(self,results_path,time_exec,n,signal_sparsity,variance_threshold_ratio,n_locations_monitored,completed=True,iteration=0):
        if completed:
            fname = f'{results_path}ExecutionTime_N{n}_S{signal_sparsity}_VarThreshold{variance_threshold_ratio}_nSensors{n_locations_monitored}.pkl'
        else:
            fname = f'{results_path}ExecutionTime_N{n}_S{signal_sparsity}_VarThreshold{variance_threshold_ratio}_nSensors{n_locations_monitored}_iteration{iteration}.pkl'
        with open(fname,'wb') as f:
            pickle.dump(time_exec,f,protocol=pickle.HIGHEST_PROTOCOL)
        print(f'File saved in {fname}')

class WriteExecutionTimeMiniBatches(FileWriter):
    """ saves execution times to a file"""
    def save(self,results_path,time_exec,n,signal_sparsity,variance_threshold_ratio,n_locations_monitored,completed=True,iteration=0):
        if completed:
            fname = f'{results_path}ExecutionTime_N{n}_S{signal_sparsity}_VarThreshold{variance_threshold_ratio}_nSensors{n_locations_monitored}.pkl'
        else:
            fname = f'{results_path}ExecutionTime_N{n}_S{signal_sparsity}_VarThreshold{variance_threshold_ratio}_nSensors{n_locations_monitored}_iteration{iteration}.pkl'
        with open(fname,'wb') as f:
            pickle.dump(time_exec,f,protocol=pickle.HIGHEST_PROTOCOL)
        print(f'File saved in {fname}')

class WriteCustomFile(FileWriter):
    def save(self,results_path,file,fname,**kwargs):
        with open(results_path+fname,'wb') as f:
            pickle.dump(file,f,protocol=pickle.HIGHEST_PROTOCOL)
        print(f'File saved in {fname}')

class FileSaver():
    """ Save locations file generated by the network design algorithm"""
    def __init__(self,writer:FileWriter):
        self._writer = writer
    def save(self,results_path,**kwargs):
        self._writer.save(results_path,**kwargs)

class FileReader(ABC):
    @abstractmethod
    def read(self,**kwrags):
        raise NotImplementedError
class ReadRandomRoiRandomSubSampling(FileReader):
    """ Load locations file when the signal corresponds to random subsampling and was randomly splitted into different ROIs"""
    def read(self,results_path,n,signal_sparsity,variance_threshold_ratio,n_locations_monitored,**kwargs)->list:
        random_seed_roi = kwargs['random_seed_roi']
        random_seed_subsampling = kwargs['random_seed_subsampling']
        fname = f'{results_path}SensorsLocations_N{n}_S{signal_sparsity}_VarThreshold{variance_threshold_ratio}_nSensors{n_locations_monitored}_randomSeed{random_seed_roi}_randomSeedSubsampling{random_seed_subsampling}.pkl'
        with open(fname,'rb') as f:
            locations_monitored = np.sort(pickle.load(f))
        return locations_monitored
class ReadRandomRoiRandomSubSampling_Boyd(FileReader):
    """ Load locations file when the signal corresponds to random subsampling and was randomly splitted into different ROIs"""
    def read(self,results_path,n,signal_sparsity,variance_threshold_ratio,n_locations_monitored,**kwargs)->list:
        random_seed_roi = kwargs['random_seed_roi']
        random_seed_subsampling = kwargs['random_seed_subsampling']
        fname = f'{results_path}SensorsLocations_Boyd_N{n}_S{signal_sparsity}_VarThreshold{variance_threshold_ratio}_nSensors{n_locations_monitored}_randomSeed{random_seed_roi}_randomSeedSubsampling{random_seed_subsampling}.pkl'
        with open(fname,'rb') as f:
            locations_monitored = np.sort(pickle.load(f))
        return locations_monitored

class FileLoader():
    def __init__(self,reader):
        self._reader = reader
    def load(self,results_path,n,signal_sparsity,variance_threshold_ratio,n_locations_monitored,**kwargs)->list:
        locations_monitored = self._reader.read(results_path,n,signal_sparsity,variance_threshold_ratio,n_locations_monitored,**kwargs)
        return locations_monitored

# sensor placement class
class OptimizationProblem(ABC):
    @abstractmethod
    def create_matrices_sdp(self, Psi, phi_matrix, constant_matrix, design_threshold,n,signal_sparsity, sensor_variance,**kwargs):
        raise NotImplementedError
    def solve(self,**kwargs):
        raise NotImplementedError

class OptimizationProblemLargeScale(OptimizationProblem):
    def create_matrices_ineq(self,n:int,Psi:np.ndarray,signal_sparsity:int,epsilon:float,locations_monitored:list,locations_unmonitored:list,n_monitored_previous_batches:int):
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
        if n_monitored_previous_batches == 0:        
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
        
        elif n_monitored_previous_batches !=0 and signal_sparsity - n_monitored_previous_batches >=0:
            # a minimum number of sensors in batch b is needed to complete the global restriction to at least sample signal's sparsity
            matrix_ineq = sparse([spdiag(matrix(-1,(n,1))),
                              spdiag(matrix(1,(n,1))),
                              matrix(-1,(1,n)),
                              sparse(-1*C_monitored),
                              sparse(C_unmonitored)])
        
            vector_ineq = matrix([matrix(np.tile(0,n)),
                                  matrix(np.tile(1,n)),
                                  -np.double(signal_sparsity - n_monitored_previous_batches),
                                  matrix(np.tile(-(1-epsilon),len(locations_monitored))),
                                  matrix(np.tile(epsilon,len(locations_unmonitored)))],tc='d')
        else:
            # a minimum number of active sensors is no longer needed in batch b
            matrix_ineq = sparse([spdiag(matrix(-1,(n,1))),
                              spdiag(matrix(1,(n,1))),
                              sparse(-1*C_monitored),
                              sparse(C_unmonitored)])
        
            vector_ineq = matrix([matrix(np.tile(0,n)),
                                  matrix(np.tile(1,n)),
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
        matrix_ineq, vector_ineq = self.create_matrices_ineq(n=n,Psi=Psi,signal_sparsity=signal_sparsity,epsilon=epsilon,
                                                             locations_monitored=locations_monitored,locations_unmonitored=locations_unmonitored,
                                                             n_monitored_previous_batches=n_monitored_previous_batches)

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
            solvers.options['show_progress'] = False
            self.problem = solvers.sdp(c,Gl=matrix_ineq,hl=vector_ineq,Gs=matrix_sdp,hs=vector_sdp,verbose=False,solver='dsdp',primalstart=primal_start)
            print('dsdp solver found')
        except:    
            print('Solving using non-specialized solver')
            solvers.options['show_progress'] = False
            self.problem = solvers.sdp(c,Gl=matrix_ineq,hl=vector_ineq,Gs=matrix_sdp,hs=vector_sdp,verbose=False)
        self.h = np.array(self.problem['x'])

# MiniBatches Classes
class Batches(ABC):
    @abstractmethod
    def create_batch(self,**kwargs):
        raise NotImplementedError
class MiniBatches2D(Batches):
    """ Creates minibatches based on 2D indices of matrix. The large matrix is scanned from left to right and up to down"""
    def create_batch(self,n_rows,n_cols,batch_rows,batch_cols):
        if n_rows%batch_rows:
            raise ValueError(f'Number of rows per minibatch ({batch_rows}) is not multiple of total number of rows ({n_rows}).')
        elif n_cols%batch_cols:
            raise ValueError(f'Number of columns per minibatch ({batch_cols}) is not multiple of total number of rows ({n_cols}).')
        elif batch_rows > n_rows:
            raise ValueError(f'Number of rows per minibatch ({batch_rows}) is larger than total number of rows in original image ({n_rows}).')
        elif batch_cols > n_cols:
            raise ValueError(f'Number of rows per minibatch ({batch_cols}) is larger than total number of rows in original image ({n_cols}).')
        else:
            """
            minibatches = {}
            b = 1
            for i in range(0,n_rows,batch_rows):
                for j in range(0,n_cols,batch_cols):
                    row_idx = range(i,min(i+batch_rows,n_rows))
                    col_idx = range(j,min(j+batch_cols,n_cols))
                    minibatches[b] = [(r,c) for r in row_idx for c in col_idx]
                    b+=1
            """
            minibatches = np.zeros(shape=(n_rows,n_cols),dtype=int)
            b=1
            for i in range(0,n_rows,batch_rows):
                for j in range(0,n_cols,batch_cols):
                    row_idx = slice(i,min(i+batch_rows,n_rows))
                    col_idx = slice(j,min(j+batch_cols,n_cols))
                    minibatches[row_idx,col_idx] = int(b)
                    b+=1
            return minibatches

class MiniBatches1D(Batches):
    """ Create minibatches based on 1D vector containing all indices """
    def create_batch(self,batch_size,n_total):
        if n_total%batch_size:
            raise ValueError(f'Number of elements for subsplit partition {batch_size} is not multiple of total number of locations {n_total}')
        elif batch_size > n_total:
            raise ValueError(f'Number of elements for subsplit partition {batch_size} larger than total number of potential locaitons in original image {n_total}')
        minibatches = dict(zip(np.arange(int(n_total/batch_size)),np.split(np.arange(n_total),n_total/batch_size)))
        return minibatches
    
# Network design algorithm classes
class NetworkDesign(ABC):
    @abstractmethod
    def design(self,**kwargs):
        raise NotImplementedError
class NetworkDesignCVXPY(NetworkDesign):
    def design(self,sensor_placement:sp.SensorPlacement,Psi:np.ndarray,deployed_network_variance_threshold:float,epsilon:float,h_prev:np.ndarray,weights:np.ndarray,n_it:int,locations_monitored:list=[],locations_unmonitored:list=[])->list:
        """
        IRL1 network planning iteration implemented using cvxpy library
        Args:
            sensor_placement (sp.SensorPlacement): sensor placement object containing network information
            N (int): total number of network locations
            deployed_network_variance_threshold (float): error variance threshold for network design
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
        while len(locations_monitored) + len(locations_unmonitored) != sensor_placement.n:
            # solve sensor placement with constraints
            
            sensor_placement.initialize_problem(Psi,rho=deployed_network_variance_threshold,
                                                w=weights,locations_monitored=locations_monitored,locations_unmonitored=locations_unmonitored)
            sensor_placement.solve()
            print(f'Problem status: {sensor_placement.problem.status}')
            if sensor_placement.problem.status == 'optimal':
                # update sets with new monitored locations
                new_monitored = [i[0] for i in np.argwhere(sensor_placement.h.value >= 1-epsilon) if i[0] not in locations_monitored]
                new_unmonitored = [i[0] for i in np.argwhere(sensor_placement.h.value <= epsilon) if i[0] not in locations_unmonitored]

                locations_monitored += new_monitored
                locations_unmonitored += new_unmonitored
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
            sys.stdout.flush()
        time_end = time.time()
        locations = [locations_monitored,locations_unmonitored]
        print(f'IRL1 algorithm finished in {time_end-time_init:.2f}s.')
        return locations
    
class RusuWrapperCVXOPT(NetworkDesign):
    def __init__(self):
        pass
    def design(self,n:int,signal_sparsity:int,Psi:np.ndarray,constant_matrix:np.ndarray,design_threshold:list,sensor_variance:float,epsilon:float,h_prev:np.ndarray,weights:np.ndarray,n_it:int,n_monitored_previous_batches:int,locations_monitored:list=[],locations_unmonitored:list=[])->list:
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
        optimization_problem = OptimizationProblemLargeScale()
        print(f'Length locations: {len(locations_monitored) + len(locations_unmonitored)}')
        new_monitored = []
        new_unmonitored = []
        while len(locations_monitored) + len(locations_unmonitored) != n:
            # solve sensor placement with constraints
            optimization_problem.solve(n=n,Psi=Psi,signal_sparsity=signal_sparsity,weights=weights,
                                       design_threshold=design_threshold,sensor_variance=sensor_variance,
                                       constant_matrix=constant_matrix,n_monitored_previous_batches=n_monitored_previous_batches)
            sys.stdout.flush()
            if optimization_problem.problem['status'] in ['optimal','unknown']:
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
                if (np.linalg.norm(optimization_problem.h - h_prev)<=epsilon or it==n_it) and (len(locations_monitored) + len(locations_unmonitored) !=n):
                    locations_monitored += [[int(i[0]) for i in np.argsort(optimization_problem.h,axis=0)[::-1] if i not in locations_monitored][0]]
                    it = 0        
                h_prev = optimization_problem.h
                weights = 1/(h_prev + epsilon)
                it +=1
                print(f'Iteration results\n ->Primal objective: {optimization_problem.problem["primal objective"]:.6f}\n ->{len(locations_monitored) + len(locations_unmonitored)} locations assigned\n ->{len(locations_monitored)} monitored locations\n ->{len(locations_unmonitored)} unmonitored locations\n')
                sys.stdout.flush()
            else:

                print(f'Optimization problem status: {optimization_problem.problem["status"]}')
                # solver fails at iteration
                if len(new_unmonitored) !=0:
                    #locations_monitored = locations_monitored[:-len(new_monitored)]
                    locations_unmonitored = locations_unmonitored[:-len(new_unmonitored)]
                if it==n_it and (len(locations_monitored) + len(locations_unmonitored) !=n):
                    locations_monitored += [[int(i[0]) for i in np.argsort(optimization_problem.h,axis=0)[::-1] if i not in locations_monitored][0]]
                    it=0
                it+=1


        time_end = time.time()
        locations = [locations_monitored,locations_unmonitored]
        print('-'*50)
        print(f'IRL1 algorithm finished in {time_end-time_init:.2f}s.')
        print('-'*50+'\n')
        sys.stdout.flush()
        return locations,time_end - time_init

class BatchesUnion(NetworkDesign):
    def design(self,**kwargs)->list:
        """ Iterate over different minibatches from the basis. The number of potential locations increases at each step.
        The minibatches dictionary is defined in order to solve the IRNet algorithm on a smaller basis that increases until it covers the whole size of the original basis.
        The ROI dictionary only contains indices for different design thresholds """

        # variables for large scale algorithm
        batch_size = kwargs['batch_size']
        Psi_full = kwargs['Psi_full']
        snapshots_matrix_full = kwargs['snapshots_matrix_full']
        n_total = Psi_full.shape[0]
        # IRNet parameters
        epsilon = kwargs['epsilon']
        n_it = kwargs['n_it_irnet']
        # ROIs parameters
        roi_idx = kwargs['roi_idx']
        variance_threshold_ratio = kwargs['variance_threshold_ratio']
        
        if n_total%batch_size:
            raise ValueError(f'Number of elements for subsplit partition {batch_size} is not multiple of total number of locations {n_total}')
        elif batch_size > n_total:
            raise ValueError(f'Number of elements for subsplit partition {batch_size} larger than total number of potential locaitons in original image {n_total}')
        algorithm = 'IRL1ND'
        energy_threshold = 0.9
        # tracking monitored locations at each sub-split
        locations_monitored = []
        locations_unmonitored = []
        locations_monitored_subsplit = []
        locations_unmonitored_subsplit = []       
        
        # Iterate over different sub-areas sizes sizes and solve the IRNet algorithm
        minibatches_1D = MiniBatches1D()
        minibatches = minibatches_1D.create_batch(batch_size=batch_size,n_total=n_total)
        
        iterator = range(len(minibatches))
        subsplit_indices = [i for i in minibatches.values()]
        subsplit_keys = [i for i in minibatches.keys()]
        time_iterations = np.zeros(shape=int(n_total/batch_size))
        
        for i in iterator:
            indices = np.sort(np.concatenate([minibatches[j] for j in subsplit_keys[:i+1]]))
            snapshots_matrix_subsplit = snapshots_matrix_full[indices,:]
            U_subsplit,sing_vals_subsplit,Vt_subsplit = np.linalg.svd(snapshots_matrix_subsplit,full_matrices=False)
            energy_subsplit = np.cumsum(sing_vals_subsplit)/np.sum(sing_vals_subsplit)
            signal_sparsity_subsplit = np.where(energy_subsplit>=0.9)[0][0]
            Psi_subsplit = U_subsplit[:,:signal_sparsity_subsplit]
            n_subsplit = Psi_subsplit.shape[0]
            print(f'Current region has {n_subsplit} potential locations')
            sys.stdout.flush()
            # get indices of each ROI that are selected by the subsplit as indicated in indices
            roi_idx_subsplit = {el:[] for el in roi_idx.keys()}
            for k in roi_idx:
                roi_idx_subsplit[k] = np.array([j for j in roi_idx[k] if j in indices])
            coordinate_error_variance_fullymonitored_subsplit = np.diag(Psi_subsplit@Psi_subsplit.T)
            maxvariance_fullymonitored_ROI_subsplit = [np.max(coordinate_error_variance_fullymonitored_subsplit[i]) for i in roi_idx_subsplit.values() if len(i)!=0]
            # define deployed network variance threshold for every location in the subsplit depending on the ROI it falls on
            deployed_network_variance_threshold_subsplit = np.zeros(n_subsplit)
            for f,v,idx in zip(variance_threshold_ratio.values(),maxvariance_fullymonitored_ROI_subsplit,roi_idx_subsplit.values()):
                if len(idx)!=0:
                    np.put(deployed_network_variance_threshold_subsplit,idx,f*v)

            # carry out monitored locations from previous subsplit step
            if len(locations_monitored)!=0:
                locations_monitored_subsplit = np.where(np.isin(indices,locations_monitored))[0]
                locations_monitored_subsplit = locations_monitored_subsplit.tolist()
            else:
                locations_monitored_subsplit = []

            sensor_placement_subsplit = sp.SensorPlacement(algorithm, n_subsplit, signal_sparsity_subsplit,
                                                            n_refst=n_subsplit,n_lcs=0,n_unmonitored=0)
            h_prev = np.zeros(n_subsplit)
            weights = 1/(h_prev+epsilon)
            # call IRNet algorithm for subsplit
            network_design_subsplit = IRNET(NetworkDesignCVXOPT())
            locations_subsplit,time_subsplit = network_design_subsplit.design_network(sensor_placement=sensor_placement_subsplit,Psi=Psi_subsplit,deployed_network_variance_threshold=deployed_network_variance_threshold_subsplit,
                                                                                      epsilon=epsilon,h_prev=h_prev,weights=weights,n_it=n_it,
                                                                                      locations_monitored=locations_monitored_subsplit,locations_unmonitored=[])
            # worst coordinate variance of selected locations
            sensor_placement_subsplit.locations = [[],np.sort(locations_subsplit[0]),np.sort(locations_subsplit[1])]
            sensor_placement_subsplit.C_matrix()
            coordinate_variance_subsplit = np.diag(Psi_subsplit@np.linalg.inv(Psi_subsplit.T@sensor_placement_subsplit.C[1].T@sensor_placement_subsplit.C[1]@Psi_subsplit)@Psi_subsplit.T)
            worst_coordinate_variance_subsplit = coordinate_variance_subsplit.max()
            
            
            # evaluate if algorithm was successful
            """
            success = [ [np.max(coordinate_variance_subsplit[j]) for j in roi_idx_subsplit.values()][i]<variance_threshold_ratio[i]*maxvariance_fullymonitored_ROI_subsplit[i] for i in range(n_regions)]
            for i in range(n_regions):
                if success[i]:
                    print(f'Success in deploying sensors for ROI {i}')
                else:
                    warnings.warn(f'Region of interest {i} failed to fulfill design threshold')
            """
            # get monitored locations from original set of indices
            locations_monitored_subsplit = sensor_placement_subsplit.locations[1]
            locations_monitored = indices[locations_monitored_subsplit]
            locations_monitored_subsplit = []
            locations_unmonitored_subsplit = []
            time_iterations[i] = time_subsplit
            print('-'*30)
            print(f'Times up to iteration {i}: {[np.round(k,3) for k in time_iterations if k!=0]}')
            print('-'*30)
            if i%2 ==0:
                saver = FileSaver(WriteExecutionTime())
                saver.save(results_path,time_exec=time_iterations,n=n_total,signal_sparsity=signal_sparsity_subsplit,
                           variance_threshold_ratio=variance_threshold_ratio,n_locations_monitored=len(locations_monitored),completed=False,iteration=i)
            sys.stdout.flush()
        locations = locations_subsplit
        return locations, time_iterations

class AggregatedBatches(NetworkDesign):
    def design(self,image,snapshots_matrix_full,roi_idx,roi_regions,energy_threshold=0.9,batch_rows=10,batch_cols=5,epsilon=5e-2,n_it=20,sensor_variance=1.,results_path='',order='forward',pre_computed_batches=False,**kwargs)->list:
        """ 
        Solve the IRNet algorithm by splitting the image into minibatches.
        On each iteration the snapshots matrix is sampled on the indices belonging to the k-th minibatch and the corresponding basis is computed(from SVD)

        The minibatch is used to solve the IRNet at each iteration.
        Previous solutions are used to store the corresponding basis vectors.
        The total number of potential locaitons does NOT increase with each iteration.
        """
        
        # split whole image into minibatches
        n_rows,n_cols = image.shape
        n_tot = n_rows*n_cols
        minibatches_2D = MiniBatches2D()
        minibatches = minibatches_2D.create_batch(n_rows=n_rows,n_cols=n_cols,
                                                  batch_rows=batch_rows,batch_cols=batch_cols)
        B = np.unique(minibatches.astype(int))
        n_minibatches = len(B)
        minibatches_flat = np.reshape(minibatches,newshape=minibatches.shape[0]*minibatches.shape[1],order='F')
        _,batch_sizes = np.unique(minibatches,return_counts=True)
        batch_size = np.unique(batch_sizes)
        if len(batch_size) != 1:
            warnings.warn(f'Batch sizes ({len(batch_size)}) is not the same across all minibatches.')
        else:
            batch_size = batch_size[0]
        
        # Regions of Interest
        roi_flat = np.reshape(roi_idx,newshape=roi_idx.shape[0]*roi_idx.shape[1],order='F')
        threshold_flat = np.array([roi_regions[i] for i in roi_flat])
        roi_dict = {el:[] for el in np.unique(roi_flat) }
        for i in roi_dict:
            roi_dict[i] = np.where(roi_flat==i)[0]

        # Iterate over different minibatches and solve the IRNet algorithm
        minibatches_indices = {el:[] for el in B}
        minibatches_indices_monitored = {el:[] for el in B}
        minibatches_indices_unmonitored = {el:[] for el in B}
        for b in B:
            minibatches_indices[b] = np.where(minibatches_flat==b)[0]

        # set minibatch iteration order and pre-computed batches
        n_locations_monitored = 0
        time_batches = {el:0 for el in B}
        if 'b_init' in kwargs:
            b_specified = kwargs['b_init']
            print(f'Initial batch specified: {b_specified}')
            b_init = np.where(B==kwargs['b_init'])[0][0]
            if order == 'forward':
                iterator = B[b_init:]
            elif order == 'backwards':
                iterator = B[b_init::-1]
        else:
            if order == 'forward':
                iterator = B
            elif order == 'backwards':
                iterator = B[::-1]
        if pre_computed_batches:
            print(f'pre-computed batches specified. Loading previous files\n')
            fname = f'{results_path}minibatches_indices_monitored_N{n_tot}_batchSize{batch_size}_batch{iterator[0]}of{B[-1]}.pkl'
            with open(fname,'rb') as f:
                minibatches_indices_monitored_loaded = pickle.load(f)
            for i in minibatches_indices_monitored_loaded:
                minibatches_indices_monitored[i] = minibatches_indices_monitored_loaded[i]
            
            fname = f'{results_path}minibatches_indices_unmonitored_N{n_tot}_batchSize{batch_size}_batch{iterator[0]}of{B[-1]}.pkl'
            with open(fname,'rb') as f:
                minibatches_indices_unmonitored_loaded = pickle.load(f)
            for i in minibatches_indices_unmonitored_loaded:
                minibatches_indices_unmonitored[i] = minibatches_indices_unmonitored_loaded[i]
            
            fname = f'{results_path}time_batches_N{n_tot}_batchSize{batch_size}_batch{iterator[0]}of{B[-1]}.pkl'
            with open(fname,'rb') as f:
                time_batches_loaded = pickle.load(f)
            for i in time_batches_loaded:
                time_batches[i] = time_batches_loaded[i]
            
            iterator = iterator[1:]

        
        # batch iteration
        for b in iterator:
            print(f'Starting minibatch {b}')
            # obtain indices of union of minibatches up to batch b
            # idx_union regulates the order inside the Union of multiple batches
            idx_union = []
            for k in np.arange(B[0],b+1,1):
                idx_batch = np.where(minibatches_flat==k)[0]
                idx_union.extend(idx_batch)
            print(f'New batch has {len(idx_batch)} new locations')
            

            # Sort snapshots matrix accordding to sorted indices that belong to the union set and compute low-rank basis
            snapshots_matrix_union = snapshots_matrix_full[idx_union,:]
            U_union,sing_vals_union,_ = np.linalg.svd(snapshots_matrix_union,full_matrices=False)
            energy_union = np.cumsum(sing_vals_union)/np.sum(sing_vals_union)
            signal_sparsity_union = np.where(energy_union>=energy_threshold)[0][0]
            if signal_sparsity_union == 0:
                signal_sparsity_union +=1
            Psi_union = U_union[:,:signal_sparsity_union]
            print(f'Signal from union of minibatches has sparsity: {signal_sparsity_union}')

            

            # obtain ROIs inside minibatch
            roi_union = roi_flat[idx_union].astype(int)
            roi_batch = roi_flat[idx_batch].astype(int)
            threshold_union = threshold_flat[idx_union]
            threshold_batch = threshold_flat[idx_batch]

            # get worst coordiante error variance for fully-monitored network per ROI
            cev_fm_union = np.diag(Psi_union@Psi_union.T)
            wcev_fm_roi_union = {el:[] for el in np.unique(roi_union)}
            for i in wcev_fm_roi_union:
                idx = np.where(roi_union == i)[0]
                if len(idx) !=0:
                    wcev_fm_roi_union[i] = cev_fm_union[idx].max()
            
            # compute design threshold at each location in batch
            design_threshold_batch = np.array([t*wcev_fm_roi_union[r] for t,r in zip(threshold_batch,roi_batch)])
            """ 
            for i in range(len(design_threshold_batch)):
                if design_threshold_batch[i]<sensor_variance:
                    warnings.warn(f'Design threshold ({design_threshold_batch[i]}) at location {i} is lower than sensor variance ({sensor_variance:.2f}). Changing to {sensor_variance:.2f}')
                    design_threshold_batch[i] = sensor_variance
            """
            # obtain monitored locations in union set
            monitored_locations_union = []
            for k in np.arange(B[0],b+1,1):
                monitored_locations_batch = minibatches_indices[k][minibatches_indices_monitored[k]]
                if len(monitored_locations_batch) !=0:
                    monitored_locations_union.extend([np.where(idx_union==i)[0][0] for i in monitored_locations_batch])
            n_monitored_previous_batches = len(monitored_locations_union)
            print(f'Number of monitored locations from previous batches: {n_monitored_previous_batches}')
            sys.stdout.flush()

            # compute constant matrix contribution from monitored locations
            constant_matrix = np.zeros(shape=(signal_sparsity_union,signal_sparsity_union))
            for idx in monitored_locations_union:
                row_basis_vector = Psi_union[idx,:][None,:]
                constant_matrix += row_basis_vector.T@row_basis_vector
            
            # get indices in union that strictly belong to batch b: those are the locations the algorithm will determine
            indices = []
            for idx in idx_batch:
                indices.extend(np.where(idx_union==idx)[0])
            Psi = Psi_union[indices,:]
            n = Psi.shape[0]
            
            # IRNet algorithm for batch b
            h_prev = np.zeros(n)
            weights = 1/(h_prev+epsilon)

            irnet_batch = RusuWrapperCVXOPT()
            locations_batch,time_batch = irnet_batch.design(n=n,signal_sparsity=signal_sparsity_union,Psi=Psi,constant_matrix=constant_matrix,
                                                            design_threshold=design_threshold_batch,sensor_variance=sensor_variance,
                                                            epsilon=epsilon,n_it=n_it,
                                                            h_prev=h_prev,weights=weights,
                                                            locations_monitored=[],locations_unmonitored=[],n_monitored_previous_batches=n_monitored_previous_batches)
            
            minibatches_indices_monitored[b] = locations_batch[0]
            minibatches_indices_unmonitored[b] = locations_batch[1]
            time_batches[b] = time_batch
            n_locations_monitored += len(minibatches_indices_monitored[b])
            print(f'Total number of monitored locations up to batch {b}: {n_locations_monitored}\n')
            # save intermediate steps
            if b%2==0:
                print('Saving partial results')
                saver = FileSaver(WriteCustomFile())
                saver.save(results_path=results_path,file=minibatches,fname=f'minibatches_N{n_tot}_batchSize{batch_size}.pkl')
                saver.save(results_path=results_path,file=minibatches_indices,fname=f'minibatches_indices_N{n_tot}_batchSize{batch_size}.pkl')
                saver.save(results_path=results_path,file=minibatches_indices_monitored,fname=f'minibatches_indices_monitored_N{n_tot}_batchSize{batch_size}_batch{b}of{B[-1]}.pkl')
                saver.save(results_path=results_path,file=minibatches_indices_unmonitored,fname=f'minibatches_indices_unmonitored_N{n_tot}_batchSize{batch_size}_batch{b}of{B[-1]}.pkl')
                saver.save(results_path=results_path,file=time_batches,fname=f'time_batches_N{n_tot}_batchSize{batch_size}_batch{b}of{B[-1]}.pkl')

        print('Saving final results')
        saver = FileSaver(WriteCustomFile())
        saver.save(results_path=results_path,file=minibatches,fname=f'minibatches_N{n_tot}_batchSize{batch_size}.pkl')
        saver.save(results_path=results_path,file=minibatches_indices,fname=f'minibatches_indices_N{n_tot}_batchSize{batch_size}.pkl')
        saver.save(results_path=results_path,file=minibatches_indices_monitored,fname=f'minibatches_indices_monitored_N{n_tot}_batchSize{batch_size}.pkl')
        saver.save(results_path=results_path,file=minibatches_indices_unmonitored,fname=f'minibatches_indices_unmonitored_N{n_tot}_batchSize{batch_size}.pkl')
        saver.save(results_path=results_path,file=time_batches,fname=f'time_batches_N{n_tot}_batchSize{batch_size}.pkl')


        return  minibatches_indices,minibatches_indices_monitored,minibatches_indices_unmonitored,time_batches
            

class NetworkDesignIterativeCVXOPT(NetworkDesign):
    def design(self,dataset,Psi:np.ndarray,variance_threshold_ratio:float,epsilon:float,n_it:int,locaitons_forbidden:list=[])->list:
        """
        IRNet network planning algorithm for large monitoring network using Regions of Interest (ROIs) and forbidden locations
        Args:
            dataset : loaded dataset containing coordinates and snapshots matrix
            Psi (np.ndarray): low-rank basis with (n_rows) locations and (n_cols) vectors/dimension
            variance_threshold_ratio (float): percentage of worsening of coordinate error variance with respect to fully monitored network
            epsilon (float): IRL1 weights update constant
            n_it (int): IRL1 max iterations
            locations_forbidden (list,optional): set of indices of locations where a sensor cannot be deployed

        Returns:
            locations (list): indices of monitored and unmonitored locations [S,Sc]
        """
        algorithm = 'IRNet_ROI'#['IRNet_ROI','NetworkPlanning_iterative_LMI','IRL1ND']
        # Define ROIs size
        lat_min,lat_max = -90,90
        lon_min,lon_max = 0,360
        delta_lat = 5
        delta_lon = 5
        
        locations_monitored = []
        locations_unmonitored = []
        locations_monitored_roi = []
        locations_unmonitored_roi = []
        
        time_init = time.time()
        idx_roi = np.array([],dtype=int)
        for latitude in np.arange(lat_min,lat_max,delta_lat):
            for longitude in np.arange(lon_min,lon_max,delta_lon):
                # get entries of original (large) basis matrix that belong to ROI
                idx_roi_new = define_ROI(dataset,lat_min=latitude,lat_max=latitude+delta_lat,lon_min=longitude,lon_max=longitude+delta_lon)
                print(f'Longitude: {longitude} - Latitude: {latitude}')
                print(f'Number of new elements in ROI: {len(idx_roi_new)}')
                if len(idx_roi_new) == 0:
                    continue
                # joint new indices to previous ROI indices
                idx_roi = np.sort(np.unique(np.concatenate((idx_roi,idx_roi_new),axis=0,dtype=np.int64)))
                Psi_roi = Psi[idx_roi,:]
                fully_monitored_network_max_variance_roi = da.diagonal(da.matmul(Psi_roi,Psi_roi.T)).max()
                fully_monitored_network_max_variance_roi = fully_monitored_network_max_variance_roi.compute()
                deployed_network_variance_threshold_roi = variance_threshold_ratio*fully_monitored_network_max_variance_roi
                n_roi = Psi_roi.shape[0]
                print(f'Size of ROI: {n_roi}')
                sys.stdout.flush()
                Psi_roi = Psi_roi.compute()

                # IRNet method parameters
                sensor_placement = sp.SensorPlacement(algorithm,n_roi,args.signal_sparsity,n_refst=n_roi,n_lcs=0,n_unmonitored=0)
                epsilon_zero = epsilon/10
                primal_start_roi = {'x':[],'sl':[],'ss':[]}
                it_roi = 0
                h_prev_roi = np.zeros(n_roi)
                weights_roi = 1/(h_prev_roi+args.epsilon)
                # carry monitored locations from previous ROI step
                if len(locations_monitored)!=0:
                    locations_monitored_roi = np.where(np.isin(idx_roi,locations_monitored))[0]
                else:
                    locations_monitored_roi = []
                if len(locations_unmonitored)!=0:
                    locations_unmonitored_roi = np.where(np.isin(idx_roi,locations_unmonitored_roi))[0]
                else:
                    locations_unmonitored_roi = []
                
                new_monitored_roi = []
                new_unmonitored_roi = []

                # begin IRNet iterations
                while len(locations_monitored_roi) + len(locations_unmonitored_roi) != n_roi:
                    # solve sensor placement with constraints
                    sensor_placement.initialize_problem(Psi_roi,rho=deployed_network_variance_threshold_roi,w=weights_roi,epsilon=epsilon,
                                                    locations_monitored=locations_monitored_roi,locations_unmonitored = locations_unmonitored_roi,
                                                    primal_start=primal_start_roi,include_sparsity_constraint=False)
                    
                    if sensor_placement.problem['status'] == 'optimal':
                        # get solution dictionary
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
                            locations_monitored_roi += [[int(i[0]) for i in np.argsort(sensor_placement.h,axis=0)[::-1] if i not in locations_monitored_roi][0]]
                            it_roi = 0        
                        h_prev_roi = sensor_placement.h
                        weights_roi = 1/(h_prev_roi + epsilon)
                        it_roi +=1
                        print(f'Iteration results\n ->Primal objective: {sensor_placement.problem["primal objective"]:.6f}\n ->{len(locations_monitored_roi) + len(locations_unmonitored_roi)} locations assigned\n ->{len(locations_monitored_roi)} monitored locations\n ->{len(locations_unmonitored_roi)} unmonitored locations\n')
                        sys.stdout.flush()
                    
                    else:
                        # solver fails at iteration
                        #locations_monitored = locations_monitored[:-len(new_monitored)]
                        locations_unmonitored_roi = locations_unmonitored_roi[:-len(new_unmonitored_roi)]
                        it_roi+=1
                
                # add monitored locations from ROI to list of overall monitored locations of original basis entries
                locations_monitored += list(idx_roi[locations_monitored_roi])
                locations_unmonitored += list(idx_roi[locations_unmonitored_roi])
        time_end = time.time()

        locations = [locations_monitored,locations_unmonitored]
        print(f'IRNet algorithm finished in {time_end-time_init:.2f}s.')
        return locations

class IRNET():
    def __init__(self,algorithm):
        self._algorithm = algorithm
    def design_network(self,**kwargs)->list:
        locations = self._algorithm.design(**kwargs)
        return locations
    

def networkPlanning_iterative(sensor_placement:sp.SensorPlacement,Psi:np.ndarray,deployed_network_variance_threshold:float,epsilon:float,h_prev:np.ndarray,weights:np.ndarray,n_it:int,locations_monitored:list=[],locations_unmonitored:list=[])->list:
    """
    IRNet network planning algorithm
    Args:
        sensor_placement (sp.SensorPlacement): sensor placement object containing network information
        Psi (np.ndarray): low-rank basis with (n_rows) locations and (n_cols) vectors/dimension
        deployed_network_variance_threshold (float): error variance threshold for network design
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
    epsilon_zero = epsilon/10
    primal_start = {'x':[],'sl':[],'ss':[]}
    it = 0
    time_init = time.time()
    
    while len(locations_monitored) + len(locations_unmonitored) != sensor_placement.n:
        # solve sensor placement with constraints
        sensor_placement.initialize_problem(Psi,rho=deployed_network_variance_threshold,w=weights,epsilon=epsilon,
                                        locations_monitored=locations_monitored,locations_unmonitored = locations_unmonitored,
                                        primal_start=primal_start)
        
        if sensor_placement.problem['status'] == 'optimal':
            # get solution
            primal_start['x'] = sensor_placement.problem['x']
            primal_start['sl'] = sensor_placement.problem['sl']
            primal_start['ss'] = sensor_placement.problem['ss']
            # update sets
            new_monitored = [int(i[0]) for i in np.argwhere(sensor_placement.h >= 1-epsilon) if i[0] not in locations_monitored]
            new_unmonitored = [int(i[0]) for i in np.argwhere(sensor_placement.h <= epsilon_zero) if i[0] not in locations_unmonitored]
            locations_monitored += new_monitored
            locations_unmonitored += new_unmonitored
            # check convergence
            if np.linalg.norm(sensor_placement.h - h_prev)<=epsilon or it==n_it:
                locations_monitored += [[int(i[0]) for i in np.argsort(sensor_placement.h,axis=0)[::-1] if i not in locations_monitored][0]]
                it = 0        
            h_prev = sensor_placement.h
            weights = 1/(h_prev + epsilon)
            it +=1
            print(f'Iteration results\n ->Primal objective: {sensor_placement.problem["primal objective"]:.6f}\n ->{len(locations_monitored) + len(locations_unmonitored)} locations assigned\n ->{len(locations_monitored)} monitored locations\n ->{len(locations_unmonitored)} unmonitored locations\n')
        
        else:
            # solver fails at iteration
            locations_monitored = locations_monitored[:-len(new_monitored)]
            locations_unmonitored = locations_unmonitored[:-len(new_unmonitored)]
            it+=1


    time_end = time.time()
    locations = [locations_monitored,locations_unmonitored]
    print(f'IRL1 algorithm finished in {time_end-time_init:.2f}s.')
    return locations


def networkPlanning_iterative_ROIs(dataset,Psi:np.ndarray,variance_threshold_ratio:float,epsilon:float,n_it:int,locaitons_forbidden:list=[])->list:
    """
    IRNet network planning algorithm for large monitoring network using Regions of Interest (ROIs) and forbidden locations
    Args:
        dataset : loaded dataset containing coordinates and snapshots matrix
        Psi (np.ndarray): low-rank basis with (n_rows) locations and (n_cols) vectors/dimension
        variance_threshold_ratio (float): percentage of worsening of coordinate error variance with respect to fully monitored network
        epsilon (float): IRL1 weights update constant
        n_it (int): IRL1 max iterations
        locations_forbidden (list,optional): set of indices of locations where a sensor cannot be deployed

    Returns:
        locations (list): indices of monitored and unmonitored locations [S,Sc]
    """
    algorithm = 'IRNet_ROI'#['IRNet_ROI','NetworkPlanning_iterative_LMI','IRL1ND']
    # Define ROIs size
    lat_min,lat_max = -90,90
    lon_min,lon_max = 0,360
    delta_lat = 5
    delta_lon = 5
    
    locations_monitored = []
    locations_unmonitored = []
    locations_monitored_roi = []
    locations_unmonitored_roi = []
    
    time_init = time.time()
    idx_roi = np.array([],dtype=int)
    for latitude in np.arange(lat_min,lat_max,delta_lat):
        for longitude in np.arange(lon_min,lon_max,delta_lon):
            # get entries of original (large) basis matrix that belong to ROI
            idx_roi_new = define_ROI(dataset,lat_min=latitude,lat_max=latitude+delta_lat,lon_min=longitude,lon_max=longitude+delta_lon)
            print(f'Longitude: {longitude} - Latitude: {latitude}')
            print(f'Number of new elements in ROI: {len(idx_roi_new)}')
            if len(idx_roi_new) == 0:
                continue
            # joint new indices to previous ROI indices
            idx_roi = np.sort(np.unique(np.concatenate((idx_roi,idx_roi_new),axis=0,dtype=np.int64)))
            Psi_roi = Psi[idx_roi,:]
            fully_monitored_network_max_variance_roi = da.diagonal(da.matmul(Psi_roi,Psi_roi.T)).max()
            fully_monitored_network_max_variance_roi = fully_monitored_network_max_variance_roi.compute()
            deployed_network_variance_threshold_roi = variance_threshold_ratio*fully_monitored_network_max_variance_roi
            n_roi = Psi_roi.shape[0]
            print(f'Size of ROI: {n_roi}')
            sys.stdout.flush()
            Psi_roi = Psi_roi.compute()

            # IRNet method parameters
            sensor_placement = sp.SensorPlacement(algorithm,n_roi,args.signal_sparsity,n_refst=n_roi,n_lcs=0,n_unmonitored=0)
            epsilon_zero = epsilon/10
            primal_start_roi = {'x':[],'sl':[],'ss':[]}
            it_roi = 0
            h_prev_roi = np.zeros(n_roi)
            weights_roi = 1/(h_prev_roi+args.epsilon)
            # carry monitored locations from previous ROI step
            if len(locations_monitored)!=0:
                locations_monitored_roi = np.where(np.isin(idx_roi,locations_monitored))[0]
            else:
                locations_monitored_roi = []
            if len(locations_unmonitored)!=0:
                locations_unmonitored_roi = np.where(np.isin(idx_roi,locations_unmonitored_roi))[0]
            else:
                locations_unmonitored_roi = []
            
            new_monitored_roi = []
            new_unmonitored_roi = []

            # begin IRNet iterations
            while len(locations_monitored_roi) + len(locations_unmonitored_roi) != n_roi:
                # solve sensor placement with constraints
                sensor_placement.initialize_problem(Psi_roi,rho=deployed_network_variance_threshold_roi,w=weights_roi,epsilon=epsilon,
                                                locations_monitored=locations_monitored_roi,locations_unmonitored = locations_unmonitored_roi,
                                                primal_start=primal_start_roi,include_sparsity_constraint=False)
                
                if sensor_placement.problem['status'] == 'optimal':
                    # get solution dictionary
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
                        locations_monitored_roi += [[int(i[0]) for i in np.argsort(sensor_placement.h,axis=0)[::-1] if i not in locations_monitored_roi][0]]
                        it_roi = 0        
                    h_prev_roi = sensor_placement.h
                    weights_roi = 1/(h_prev_roi + epsilon)
                    it_roi +=1
                    print(f'Iteration results\n ->Primal objective: {sensor_placement.problem["primal objective"]:.6f}\n ->{len(locations_monitored_roi) + len(locations_unmonitored_roi)} locations assigned\n ->{len(locations_monitored_roi)} monitored locations\n ->{len(locations_unmonitored_roi)} unmonitored locations\n')
                    sys.stdout.flush()
                
                else:
                    # solver fails at iteration
                    #locations_monitored = locations_monitored[:-len(new_monitored)]
                    locations_unmonitored_roi = locations_unmonitored_roi[:-len(new_unmonitored_roi)]
                    it_roi+=1
            
            # add monitored locations from ROI to list of overall monitored locations of original basis entries
            locations_monitored += list(idx_roi[locations_monitored_roi])
            locations_unmonitored += list(idx_roi[locations_unmonitored_roi])
    time_end = time.time()

    locations = [locations_monitored,locations_unmonitored]
    print(f'IRNet algorithm finished in {time_end-time_init:.2f}s.')
    return locations

#%%
# figures
class Figures():
    def __init__(self,save_path,figx=3.5,figy=2.5,fs_title=10,fs_label=10,fs_ticks=10,fs_legend=10,marker_size=3,dpi=300,use_grid=False,show_plots=False):
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

    def SST_map(self,data,vmin=0,vmax=30,show_coords=False,show_cbar=True,save_fig=False,save_fig_fname='SST_map.png'):
        gmap = Basemap(lon_0=180, projection="kav7", resolution='c')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cmap = mpl.colormaps['Spectral'].resampled(64).reversed()
        cmap.set_bad('w')
        cmap_trunc = mpl.colors.LinearSegmentedColormap.from_list('trunc_cmap',cmap(np.linspace(0.,30.,100)),N=64)
        im = gmap.imshow(data,cmap=cmap,origin='lower',vmin=vmin,vmax=vmax)
        
        xrange = np.arange(0,1600,200)
        if show_coords:
            gmap.drawmeridians(np.arange(0, 360.01, 90), ax=ax,linewidth=0.5,labels=[1,0,1,0])
            gmap.drawparallels(np.arange(-90,90.1,45), ax=ax,linewidth=0.5,labels=[1,0,0,1])
        else:
            gmap.drawmeridians(np.arange(0, 360.01, 90), ax=ax,linewidth=0.5)
            gmap.drawparallels(np.arange(-90,90.1,45), ax=ax,linewidth=0.5)        
        
        # color bar in same figure
        """ 
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im,cax=cax,orientation='vertical')
        cbar.set_ticks(np.arange(0,34,5))
        cbar.set_ticklabels([np.round(i,1) for i in cbar.get_ticks()])
        cbar.set_label('Temperature (ºC)')
        """
        if show_cbar:
            fig2, ax2 = plt.subplots(figsize=(6, 1), layout='constrained')
            cbar = fig2.colorbar(mpl.cm.ScalarMappable(cmap=cmap,norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)),cax=ax2,
                                orientation='horizontal', label=f'Temperature (ºC)',
                                location='top',extend='both')

        fig.tight_layout()
        if save_fig:
            fname = f'{self.save_path}{save_fig_fname}'
            fig.savefig(fname,dpi=300,format='png',bbox_inches='tight')
            print(f'Figure saved at {fname}')
            if show_cbar:
                fname= f'{self.save_path}SST_map_colorbar.png'
                fig2.savefig(fname,dpi=300,format='png')

        return fig
    

    def curve_IQR_measurements(self,snapshots_matrix,save_fig):
        n = snapshots_matrix.shape[0]
        yrange = np.arange(-5,40,5)
        xrange = np.arange(0,n,1)
        median = np.median(snapshots_matrix,axis=1)
        q1,q3 = np.percentile(snapshots_matrix,axis=1,q=[25,75])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xrange,median,color='#1a5276')
        ax.fill_between(x=xrange,y1=q1,y2=q3,color='#1a5276',alpha=0.5)
        ax.set_yticks(yrange)
        ax.set_yticklabels([np.round(i,1) for i in ax.get_yticks()])
        ax.set_ylabel('Temp (ºC)')
        
        xrange = np.where(xrange%10000==0)[0]
        ax.set_xticks(xrange)
        xrange[0]=1
        ax.set_xticklabels([int(i+1) for i in xrange],rotation=0)
        ax.set_xlabel('Location index')
        fig.tight_layout()
        if save_fig:
            fname = self.save_path+'curve_Temp_allLocations.png'
            fig.savefig(fname,dpi=300,format='png',bbox_inches='tight')
            print(f'Figure saved at {fname}')

    # Low-rank plots
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
        ax.set_xticks(np.concatenate(([0.0],np.arange(xrange[99],xrange[-1]+1,100))))
        ax.set_xticklabels([int(i+1) for i in ax.get_xticks()],rotation=0)
        yrange = np.logspace(-4,0,5)
        ax.set_yticks(yrange)
        #ax.set_ylabel('Normalized singular values')
        ax.set_ylim(sing_vals[-2]/np.max(sing_vals),1)
        ax.set_yscale('log')

        ax2 = ax.twinx()
        l2 = ax2.plot(xrange,cumulative_energy,color='#1f618d',marker='o',label='Cumulative energy')
        ax2.set_xticklabels([int(i+1) for i in ax2.get_xticks()])
        ax2.set_xlabel('$i$th singular value')
        yrange = np.arange(0.,1.2,0.2)
        ax2.set_yticks(yrange)
        ax2.set_yticklabels([np.round(i,2) for i in ax2.get_yticks()])
        ax2.set_ylabel('Cumulative energy')
        ax2.set_ylim(0,1)
        
        lines = l1+l2
        labels = [l.get_label() for l in lines]
        #ax.legend(lines,labels,loc='center',ncol=1,framealpha=1.,bbox_to_anchor=(0.5,1.15),handlelength=0.5,handletextpad=0.1)
        #fig.tight_layout()

        
        if save_fig:
            fname = self.save_path+f'Curve_singVals_cumulativeEnergy_N{n}.png'
            fig.savefig(fname,dpi=600,format='png',bbox_inches='tight')
            print(f'Figure saved at: {fname}')


    def singular_values_cumulative_energy(self,sing_vals,save_fig=False):
        """
        Plot sorted singular values ratio and cumulative energy

        Parameters
        ----------
        sing_vals : numpy array
            singular values
        save_fig : bool, optional
            save generated figures. The default is False.

        Returns
        -------
        None.

        """
        cumulative_energy = np.cumsum(sing_vals)/np.sum(sing_vals)
        sing_vals_normalized = sing_vals / np.max(sing_vals)
        xrange = np.arange(0,sing_vals.shape[0],1)
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        ax.plot(xrange,cumulative_energy,color='#1f618d',marker='o')
        ax.set_xticks(np.concatenate(([0.0],np.arange(xrange[49],xrange[-1]+1,50))))
        ax.set_xticklabels([int(i+1) for i in ax.get_xticks()])
        ax.set_xlabel('$i$th singular value')
        
        #yrange = np.arange(0.5,1.05,0.05)
        yrange = np.arange(0.,1.2,0.2)
        ax.set_yticks(yrange)
        ax.set_yticklabels([np.round(i,2) for i in ax.get_yticks()])
        ax.set_ylabel('Cumulative energy')
        fig1.tight_layout()
        
        fig2 = plt.figure()
        ax = fig2.add_subplot(111)
        ax.plot(xrange, sing_vals_normalized,color='#1f618d',marker='o')
        ax.set_xticks(np.concatenate(([0.0],np.arange(xrange[49],xrange[-1]+1,50))))
        ax.set_xticklabels([int(i+1) for i in ax.get_xticks()],rotation=0)
        ax.set_xlabel('$i$th singular value')

        yrange = np.logspace(-4,0,5)
        ax.set_yticks(yrange)
        ax.set_ylabel('Normalized singular values')
        ax.set_yscale('log')
        ax.set_ylim(sing_vals_normalized[-2],1e0)
        fig2.tight_layout()
        
        if save_fig:
            fname = self.save_path+f'Curve_sparsity_cumulativeEnergy.png'
            fig1.savefig(fname,dpi=300,format='png')
            print(f'Figure saved at: {fname}')

            fname = self.save_path+f'Curve_sparsity_singularValues.png'
            fig2.savefig(fname,dpi=300,format='png')
            print(f'Figure saved at: {fname}')
         
    def boxplot_validation_rmse_svd(self,rmse_sparsity_val:pd.DataFrame,rmse_sparsity_train:pd.DataFrame=pd.DataFrame(),max_sparsity_show:int=10,save_fig:bool=False) -> plt.figure:
        yrange = np.arange(0.0,1.4,0.2)
        xrange = rmse_sparsity_val.columns[:max_sparsity_show].astype(int)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        bp_val = ax.boxplot(x=rmse_sparsity_val.iloc[:,:max_sparsity_show],notch=False,vert=True,
                   whis=1.5,bootstrap = None,
                   positions=[i for i in range(len(xrange))],widths=0.5,labels=[str(i) for i in xrange],
                   flierprops={'marker':'.','markersize':1},
                   patch_artist=True)
        for i in range(len(xrange)):
            bp_val['boxes'][i].set_facecolor('#1a5276')


        if rmse_sparsity_train.shape[0] !=0:
            bp_train = ax.boxplot(x=rmse_sparsity_train.iloc[:,:max_sparsity_show],notch=False,vert=True,
                    whis=1.5,bootstrap = None,
                    positions=[i for i in range(len(xrange))],widths=0.5,labels=[str(i) for i in xrange],
                    flierprops={'marker':'.','markersize':1},
                    patch_artist=True)
            for i in range(len(xrange)):
                bp_train['boxes'][i].set_facecolor('lightgreen')

        
        ax.set_yticks(yrange)
        ax.set_yticklabels([np.round(i,2) for i in ax.get_yticks()])
        ax.set_ylim(0,1.2)
        ax.set_ylabel('RMSE (ºC)')
        ax.set_xlabel('Sparsity level')
        fig.tight_layout()

        if save_fig:
            fname = self.save_path+f'boxplot_RMSE_SVDreconstruction_validationSet_Smin{xrange.min()}_Smax{xrange.max()}.png'
            fig.savefig(fname,dpi=300,format='png')
            print(f'Figure saved in {fname}')
    
        return fig
    
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
    
    def curve_errorvariance_comparison(self,errorvar_fullymonitored:list,errorvar_reconstruction:list,variance_threshold_ratio:float,worst_coordinate_variance_fullymonitored:float,n:int,n_sensors:int,errorvar_reconstruction_Dopt:list=[],roi_idx:dict={},n_sensors_Dopt:int=0,N=0,signal_sparsity=0,save_fig:bool=False) -> plt.figure:
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

        # Plot for a single design threshold over the whole network
        if type(variance_threshold_ratio) is float or len(variance_threshold_ratio) == 1:
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
                fname = f'{self.save_path}Curve_errorVariance_N{n}_S{signal_sparsity}_Threshold{variance_threshold_ratio:.2f}_Nsensors{n_sensors}.png'
                fig.savefig(fname,dpi=300,format='png')
                print(f'Figure saved at {fname}')

        # Plot for different design thresholds for different ROIs
        else:
            variance_threshold = [t*w for t,w in zip(variance_threshold_ratio,worst_coordinate_variance_fullymonitored)]
            # sort coordinate error variance such that the ROIs are shown in order
            coordinate_error_variance_fully_monitored_sorted = np.concatenate([errorvar_fullymonitored[i] for i in roi_idx.values()])
            coordinate_error_variance_design_sorted = np.concatenate([errorvar_reconstruction[i] for i in roi_idx.values()])

            fig = plt.figure()
            ax = fig.add_subplot(111)
            # coordinate error variance at each location
            ax.plot(coordinate_error_variance_fully_monitored_sorted,color='#943126',label='Fully monitored network')
            if len(errorvar_reconstruction_Dopt) !=0:
                coordinate_error_variance_Dopt_sorted = np.concatenate([errorvar_reconstruction_Dopt[i] for i in roi_idx.values()])
                ax.plot(coordinate_error_variance_Dopt_sorted,color='orange',label=f'Joshi-Boyd {n_sensors_Dopt} sensors',alpha=0.8)
            ax.plot(coordinate_error_variance_design_sorted,color='#1a5276',label=f'Network design {n_sensors} sensors')
            
            # horizontal lines showing threshold design
            n_roi = np.concatenate([[0],[len(i) for i in roi_idx.values()]])
            n_roi_cumsum = np.cumsum(n_roi)
            for v,l in zip(variance_threshold,range(len(n_roi_cumsum))):
                if l==0:
                    ax.hlines(y=v,xmin=n_roi_cumsum[l]-1,xmax=n_roi_cumsum[l+1]-1,color='k',linestyles='--',label='Design threshold')
                else:
                    ax.hlines(y=v,xmin=n_roi_cumsum[l],xmax=n_roi_cumsum[l+1]-1,color='k',linestyles='--')
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
            ax.set_ylabel('Coordinate error variance')
            ax.legend(loc='center',ncol=2,framealpha=0.5,bbox_to_anchor=(0.5,1.1))
            fig.tight_layout()
            if save_fig:
                fname = f'{self.save_path}Curve_SST_errorVariance_N{n}_S{signal_sparsity}_VarThreshold{variance_threshold_ratio}_Nsensors{n_sensors}_NsensorsDopt{n_sensors_Dopt}.png'
                fig.savefig(fname,dpi=300,format='png')
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


#%% Functions
# recover map
def recover_map(y:np.array,idx_nan:np.array,n_rows:int,n_cols:int)->np.ndarray:
    """
    Recovers a snapshot image from vector of measurements and array with nan indices (indicating earth)
    
    Args:
        y (np.array): array of measurements
        idx_nan (np.array): array with indices where entry is nan
        n_rows (int): number of rows of snapshot figure
        n_cols (int): number of columns of snapshot figure

    Returns:
        X_mat (np.ndarray): recovered snapshot figure
    """
    
    # initalize array with nan values
    X_vect = np.full(len(y) + len(idx_nan),np.nan,dtype=np.float32)
    #print(f'Length of non-NaNs array: {len(y)}. Length of array of NaN indices: {len(idx_nan)}. Length of recovered array: {len(X_vect)}.\nFirst NaN idx: {idx_nan.values[0][0]}. Last NaN idx: {idx_nan.values[-1][0]}')
    # recover value at ith location
    idx = 0
    for i in range(len(X_vect)):
        #if i not in idx_nan.values:
        if i not in idx_nan:
            X_vect[i] = y[idx]
            idx += 1
    # reshape to matrix form
    X_mat = np.reshape(X_vect,newshape=(n_rows,n_cols),order='F')
    return X_mat

        

def grid_mask_map(image_height:int=720, image_width:int = 1440, block_size:int = 15):
    assert image_height % block_size == 0, "Image height must be a multiple of the block size."
    assert image_width % block_size == 0, "Image width must be a multiple of the block size."
    mask_grid = np.zeros((image_height, image_width), dtype=np.float32)
    # Iterate over each block
    for i in range(0, image_height, block_size):
        for j in range(0, image_width, block_size):
            # Calculate the center of each block
            center_i = i + block_size // 2
            center_j = j + block_size // 2
            # Mark the center pixel
            mask_grid[center_i, center_j] = 1 
    return mask_grid

def window_image(data:np.ndarray,lat_range:np.array,lon_range:np.array,lon_min:float,lon_max:float,lat_min:float,lat_max:float)->np.ndarray:
    idx_window_lon = np.where(np.logical_and(lon_range>=lon_min,lon_range<=lon_max))[0]
    idx_window_lat = np.where(np.logical_and(lat_range>=lat_min,lat_range<=lat_max))[0]
    data_new = data[idx_window_lat[0]:idx_window_lat[-1]+1,idx_window_lon[0]:idx_window_lon[-1]+1]
    mask_new = np.isnan(data_new)
    return data_new,mask_new, idx_window_lat,idx_window_lon

def define_ROI(dataset,lon_min:float=0.0,lon_max:float=180.,lat_min:float=-90.,lat_max:float=0.):
    """
    Defines a ROI from the dataset indices of pixels with measurements and specified coordinates for windowing.
    Latitude in interval [-90,90]
    Longitude in interval [0,360].
    The indices can be used on the dataset.df for obtaining the respective snapshots matrix at those locations (dataset.df[:,idx_ROI_measurements])

    Args:
        dataset (_type_): dataset class containing dataset indices and coordinates
        lon_min (float, optional): Minimum longitude of ROI. Defaults to 0.0.
        lon_max (float, optional): Maximum longitude of ROI. Defaults to 180..
        lat_min (float, optional): Minimum latitude of ROI. Defaults to -90..
        lat_max (float, optional): Maximum latitude of ROI. Defaults to 0.
    """
    # check
    if lat_min < -90:
        raise ValueError(f'Wrong value for minimum latitude of ROI (-90): {lat_min}')
    if lat_max > 90:
        raise ValueError(f'Wrong value for maximum latitude of ROI (90): {lat_max}')
    if lon_min < 0:
        raise ValueError(f'Wrong value for minimum longitude of ROI (0): {lon_min}')
    if lon_max > 360.:
        raise ValueError(f'Wrong value for maximum longitude of ROI (360): {lon_max}')
    if lat_min > lat_max :
        raise ValueError(f'Wrong order of maximum ({lat_max}) and minimum ({lat_min}) latitudes')
    if lon_min > lon_max :
        raise ValueError(f'Wrong order of maximum ({lon_max}) and minimum ({lon_min}) longitudes')

    idx_measurements = np.array([i[0] for i in dataset.idx_measurements.values],dtype=int)
    mesh = np.meshgrid(dataset.lon_range,dataset.lat_range)
    mesh_lon_array = np.reshape(mesh[0],newshape=mesh[0].shape[0]*mesh[0].shape[1],order='F')
    mesh_lat_array = np.reshape(mesh[1],newshape=mesh[1].shape[0]*mesh[1].shape[1],order='F')
    # window latitude and longitude to be within ROI borders
    lon_array_ROI = np.where(np.logical_and(mesh_lon_array>=lon_min,mesh_lon_array<=lon_max))[0]
    lat_array_ROI = np.where(np.logical_and(mesh_lat_array>=lat_min,mesh_lat_array<=lat_max))[0]
    # get indices that are within the window on both coordinates
    idx_ROI = lon_array_ROI[np.isin(lon_array_ROI,lat_array_ROI)]
    # retrieve the indices in basis that correspond to ROI indices in map
    #idx_ROI_measurements = np.sort(idx_measurements[np.isin(idx_measurements,idx_ROI)])
    idx_ROI_measurements = np.sort(np.where(np.isin(idx_measurements,idx_ROI))[0])
    return idx_ROI_measurements

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


# signal reconstruction functions
def signal_reconstruction_svd(U:np.ndarray,mean_values:np.ndarray,snapshots_matrix_val:np.ndarray,s_range:np.ndarray,centered_data:bool=True) -> pd.DataFrame:
    """
    Decompose signal keeping s-first singular vectors using training set data
    and reconstruct validation set.

    Args:
        U (numpy array): left singular vectors matrix
        mean_values (numpy array): average value for each location obtained from training set snapshots matrix
        snapshots_matrix_val (numpy array): non-centererd snapshots matrix of validation set data
        s_range (numpy array): list of sparsity values to test
        centered_data (bool): True for PCA computed on centered data.

    Returns:
        rmse_sparsity: dataframe containing reconstruction errors at different times for each sparsity threshold in the range
    """
    print(f'Determining signal sparsity by decomposing training set and reconstructing validation set.\nRange of sparsity levels: {s_range}')
    rmse_sparsity = []
    for s in s_range:
        if centered_data:
            snapshots_matrix_val_pred_svd = da.matmul(U[:,:s],da.matmul(U[:,:s].T,snapshots_matrix_val - mean_values)) + mean_values
        else:
            snapshots_matrix_val_pred_svd = da.matmul(U[:,:s],da.matmul(U[:,:s].T,snapshots_matrix_val))

        error = snapshots_matrix_val - snapshots_matrix_val_pred_svd
        rmse = da.sqrt((error**2).mean(axis=0))
        rmse_sparsity.append(rmse)
    rmse_sparsity = np.array(dask.compute(*rmse_sparsity)).T
    rmse_sparsity = pd.DataFrame(rmse_sparsity,columns = s_range)

    return rmse_sparsity

def signal_reconstruction_regression(Psi:np.ndarray,locations_measured:np.ndarray,X_test:pd.DataFrame,X_test_measurements:pd.DataFrame=[],snapshots_matrix_train:np.ndarray=[],snapshots_matrix_test_centered:np.ndarray=[],projected_signal:bool=False)->pd.DataFrame:
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
    error_variance = error.var()
    """
    error_max = pd.DataFrame(np.abs(error).max(axis=1),columns=[n_sensors_reconstruction],index=X_test.index)
    error_var = np.zeros(shape = error.shape)
    for i in range(error.shape[0]):
        error_var[i,:] = np.diag(error.iloc[i,:].to_numpy()[:,None]@error.iloc[i,:].to_numpy()[:,None].T)
    error_var = pd.DataFrame(error_var,index=X_test.index,columns=X_test.columns)
    """
    return rmse, error_variance
#%%
if __name__ == '__main__':
    """ load dataset to use """
    abs_path = os.path.dirname(os.path.realpath(__file__))
    files_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/files/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/test/'    
    dataset = Dataset(files_path,LoadGRP(),Windowing(),NoSubSampling())
    rootgrp = dataset._loader.load(files_path,fname='sst.mon.mean.nc')
    lat_range,lon_range = dataset.get_coordinates(rootgrp)
    """ windowing to certain ocean area: 
        - Test coordinates (96 points): lat_min,lat_max = - -5,-3
                                        lon_min,lon_max = 190,193
        - El Nino coordinates:  lat_min,lat_max = -5,5
                                lon_min,lon_max = 190,240 (170-120 ºW)
        - Half El Nino coordinates: lat_min,lat_max = -5,5
                                    lon_min,lon_max = 215,240 (or 190,215)
        
    """
    # windowed dataset coordinates range
    lat_min,lat_max = -5,5 # the difference (*4) determine number of rows
    lon_min,lon_max = 215,240 # the difference (*4) determine number of columns
    lat_range_img,lon_range_img = dataset._preprocessor.get_coordinates_range(lat_range=lat_range,lat_min=lat_min,lat_max=lat_max,
                                                                              lon_range=lon_range,lon_min=lon_min,lon_max=lon_max)
    # dataframe
    df,idx_measurements,idx_land,img_ = dataset.create_dataframe(rootgrp,lat_range=lat_range,lon_range=lon_range,
                                                            lon_min=lon_min,lon_max=lon_max,lat_min=lat_min,lat_max=lat_max)
    print(df.head())
    # map_recovered = recover_map(df.iloc[0,:],idx_land,n_rows=int(4*np.abs(lat_min - lat_max)),n_cols=int(4*np.abs(lon_max - lon_min)))

    
    """
        Network planning algorithm
            - deploy sensors susch that the reconstructed signal variance is minimized
            - deploy single class of senors (called reference stations)
            - the number of deployed sensors is unknown a priori
        
        IRNet method adapted for large monitoring networks (SST dataset) where the whole dataset does not fit into memory
    """
    
    if args.design_large_networks:
        print(f'Iterative network planning algorithm.\n Design parameters:\n -epsilon: {args.epsilon:.1e}\n -number of convergence iterations: {args.num_it}')
        sys.stdout.flush()
        # dataset split
        train_ratio = 0.75
        validation_ratio = 0.15
        test_ratio = 0.10
        X_train, X_test = train_test_split(df, test_size= 1 - train_ratio,shuffle=False,random_state=92)
        X_val, X_test = train_test_split(X_test, test_size=test_ratio/(test_ratio + validation_ratio),shuffle=False,random_state=92)         
        # snapshots matrix 
        snapshots_matrix_train = X_train.to_numpy().T
        snapshots_matrix_val = X_val.to_numpy().T
        snapshots_matrix_test = X_test.to_numpy().T
        snapshots_matrix_train_centered = snapshots_matrix_train - snapshots_matrix_train.mean(axis=1)[:,None]
        snapshots_matrix_val_centered = snapshots_matrix_val - snapshots_matrix_train.mean(axis=1)[:,None]
        snapshots_matrix_test_centered = snapshots_matrix_test - snapshots_matrix_train.mean(axis=1)[:,None]
        energy_threshold = 0.9
        """ low-rank decomposition
        U,sing_vals,_ = np.linalg.svd(snapshots_matrix_train_centered,full_matrices=False)
        energy = np.cumsum(sing_vals)/np.sum(sing_vals)
        signal_sparsity = np.where(energy>=energy_threshold)[0][0]
        if signal_sparsity == 0:
            signal_sparsity +=1
        Psi = U[:,:signal_sparsity]
        print(f'Signal from union of minibatches has sparsity: {signal_sparsity}')
 
        plots = Figures(save_path=results_path,marker_size=1,
                        figx=3.5,figy=2.5,
                        fs_label=12,fs_ticks=12,fs_legend=10,fs_title=10,
                        show_plots=True)
        plots.singular_values_cumulative_energy_sameFigure(sing_vals,n=snapshots_matrix_train_centered.shape[1],save_fig=True)
        plt.show()
        """
                
        
        # define regions of interest (ROIs) as a subgroup of the original image. Each ROI will have a different design ratio
        # roi.roi_idx contains ROI for each index
        # roi.regions contains ROI number and design ratio
        n_rois = 1
        design_ratio = [5.0]
        if len(design_ratio) != n_rois:
            raise ValueError(f'Number of user-defined thresholds ({len(design_ratio)}) mismatch number of ROIs ({n_rois})')
        
        lat_thresholds_min,lat_thresholds_max = [-5],[5]
        lon_thresholds_min,lon_thresholds_max = [190],[240]
        roi = ROI(CoordinateRoi2D(),design_ratio)
        roi.define_rois(lat_range_img=lat_range_img,lon_range_img=lon_range_img,
                        lat_thresholds_min=lat_thresholds_min,lat_thresholds_max=lat_thresholds_max,
                        lon_thresholds_min=lon_thresholds_min,lon_thresholds_max=lon_thresholds_max,
                        n_regions=n_rois
                        )
        if len(roi.regions) != n_rois:
            raise ValueError(f'Number of created ROIs ({len(roi.regions)}) mismatch number of input ROIs ({n_rois})')
        else:
            print(f'Regions Of Interest (ROIs) created: {roi.regions}')
        # IRNet algorithm for large scale deployments
        
        irnet = AggregatedBatches()
        minibatches_indices,minibatches_indices_monitored,minibatches_indices_unmonitored,time_batches = irnet.design(image=img_,snapshots_matrix_full=snapshots_matrix_train_centered,
                                                                                                                      roi_idx=roi.roi_idx,roi_regions=roi.regions,
                                                                                                                      energy_threshold=energy_threshold,
                                                                                                                      sensor_variance=1.,results_path=results_path,
                                                                                                                      **vars(args))
        
         
        sys.exit()
        
        """ 
        # worst coordinate variance computed measuring at selected locations
        sensor_placement_minibatch.locations = [[],np.sort(locations_minibatch[0]),np.sort(locations_minibatch[1])]
        sensor_placement_minibatch.C_matrix()
        coordinate_variance_minibatch = np.diag(Psi_minibatch@np.linalg.inv(Psi_minibatch.T@sensor_placement_minibatch.C[1].T@sensor_placement_minibatch.C[1]@Psi_minibatch)@Psi_minibatch.T)
        # evaluate if algorithm was successful
        worst_coordinate_variance_minibatch = coordinate_variance_minibatch.max()
        worst_coordiante_variance_ROI_minibatch = {el:[] for el in variance_threshold_ratio.keys()}
        for i in worst_coordiante_variance_ROI_minibatch:
            idx = roi_idx_minibatch[i]
            if len(idx)!=0:
                worst_coordiante_variance_ROI_minibatch[i] = np.max(coordinate_variance_minibatch[idx])
                if worst_coordiante_variance_ROI_minibatch[i] > variance_threshold_ratio[i]*maxvariance_fullymonitored_ROI_minibatch[i]:
                    warnings.warn(f'Region of interest {i} failed to fullfill design threshold')
        
        # get monitored locations in original set of indices
        locations_monitored_minibatch = sensor_placement_minibatch.locations[1]
        locations_monitored = [int(i) for i in np.concatenate([locations_monitored,indices_minibatch[locations_monitored_minibatch]])]
        time_batches[b] = time_minibatch
        n_locations_monitored += len(locations_monitored_minibatch)
        print('-'*30)
        print(f'Times up to batch {b}: {[np.round(k,3) for k in time_batches if k!=0]}')
        print(f'Locations monitored up to batch {b}: {n_locations_monitored}')
        print('-'*30)
        # save progress after some iterations
        if b%10 ==0:
            saver = FileSaver(WriteExecutionTimeMiniBatches())
            saver.save(results_path,time_exec=time_batches,
                        n=n_total,signal_sparsity=signal_sparsity_full,
                        variance_threshold_ratio=[i for i in variance_threshold_ratio.values()],n_locations_monitored=n_locations_monitored,
                        completed=False,iteration=b)
            
            saver = FileSaver(WriteLocationsMiniBatches())
            saver.save(results_path,locations=locations_monitored,
                        n=n_total,signal_sparsity=signal_sparsity_full,
                        variance_threshold_ratio=[i for i in variance_threshold_ratio.values()],n_locations_monitored=n_locations_monitored,
                        completed=False,iteration=b)
        sys.stdout.flush()
        print(f'Network design using MiniBatches strategy finished with {n_locations_monitored} locations monitored')
        locations_unmonitored = [i for i in np.arange(n_total) if i not in locations_monitored]
        locations = [locations_monitored,locations_unmonitored]
        
        
        # coordiante error variance and design thresholds
        coordinate_error_variance_fullymonitored = np.diag(Psi@Psi.T)
        maxvariance_fullymonitored = coordinate_error_variance_fullymonitored.max()
        maxvariance_fullymonitored_ROI = [np.max(coordinate_error_variance_fullymonitored[i]) for i in roi_idx.values()]
        
        
        deployed_network_variance_threshold = np.zeros(shape=n)
        design_threshold = np.zeros(shape=n)
        for f,v,idx in zip(design_ratio.values(),maxvariance_fullymonitored_ROI,roi_idx.values()):
            np.put(deployed_network_variance_threshold,idx,f*v)
            np.put(design_threshold,idx,f)
            
        # deploy sensors at selected locations and compute variance-covariance matrix
        sensor_placement.locations = [[],np.sort(locations[0]),np.sort(locations[1])]
        sensor_placement.C_matrix()
        # compute coordinate error variance at each ROI
        coordinate_error_variance = np.diag(Psi@np.linalg.inv(Psi.T@sensor_placement.C[1].T@sensor_placement.C[1]@Psi)@Psi.T)
        worst_coordinate_variance = coordinate_error_variance.max()
        worst_coordinate_variance_ROIs = [np.max(coordinate_error_variance[idx]) for idx in roi_idx.values()]
        design_threshold_ROIs = [f*v for f,v in zip(variance_threshold_ratio.values(),maxvariance_fullymonitored_ROI)]
        print(f'Network design coordiante error variance threshold by ROI: {[np.round(i,3) for i in design_threshold_ROIs]}\nAchieved worst coordinate error variance by ROI: {[np.round(i,3) for i in worst_coordinate_variance_ROIs]}')
        n_locations_monitored = len(locations[0])
        n_locations_unmonitored = len(locations[1])
        print(f'Network planning results:\n- Total number of potential locations: {n}\n- basis sparsity: {signal_sparsity}\n- Number of monitored locations: {n_locations_monitored}\n- Number of unmonitored locations: {n_locations_unmonitored}\n')
        
        # save results
        saver = FileSaver(WriteLocationsMiniBatches())
        saver.save(results_path,locations=locations,n=n,signal_sparsity=signal_sparsity,
                    variance_threshold_ratio=[i for i in variance_threshold_ratio.values()],n_locations_monitored=n_locations_monitored,
                    completed=True)

        saver = FileSaver(WriteExecutionTimeMiniBatches())
        saver.save(results_path,time_exec=time_exec,n=n,signal_sparsity=signal_sparsity,
                    variance_threshold_ratio=[i for i in variance_threshold_ratio.values()],n_locations_monitored=n_locations_monitored,
                    completed=True)
        """
        

    """ Reconstruct signal using measurements at certain locations and compare with actual values """
    reconstruct_signal = False
    if reconstruct_signal:
        # dataset split
        train_ratio = 0.75
        validation_ratio = 0.15
        test_ratio = 0.10
        X_train, X_test = train_test_split(df, test_size= 1 - train_ratio,shuffle=False,random_state=92)
        X_val, X_test = train_test_split(X_test, test_size=test_ratio/(test_ratio + validation_ratio),shuffle=False,random_state=92) 
        
        # low-rank decomposition
        snapshots_matrix_train = X_train.to_numpy().T
        snapshots_matrix_val = X_val.to_numpy().T
        snapshots_matrix_test = X_test.to_numpy().T
        snapshots_matrix_train_centered = snapshots_matrix_train - snapshots_matrix_train.mean(axis=1)[:,None]
        snapshots_matrix_val_centered = snapshots_matrix_val - snapshots_matrix_train.mean(axis=1)[:,None]
        snapshots_matrix_test_centered = snapshots_matrix_test - snapshots_matrix_train.mean(axis=1)[:,None]
        U,sing_vals,Vt = np.linalg.svd(snapshots_matrix_train_centered,full_matrices=False)
        print(f'Training snapshots matrix has dimensions {snapshots_matrix_train_centered.shape}.\nLeft singular vectors matrix has dimensions {U.shape}\nRight singular vectors matrix has dimensions {Vt.shape}\nNumber of singular values: {sing_vals.shape}')
        cumulative_energy = np.cumsum(sing_vals)/np.sum(sing_vals)
        energy_threshold = 0.9
        signal_sparsity = np.where(cumulative_energy>=energy_threshold)[0][0]
        print(f'Energy threshold of {energy_threshold} reached at singular at singular value index: {signal_sparsity}')
        Psi = U[:,:signal_sparsity]
        n = Psi.shape[0]
        print(f'Basis shape: {Psi.shape}')
        sys.stdout.flush()
        
        # define regions with different threshold desings
        random_seed = 0
        roi = ROI(RandomRoi())
        roi.define_rois(seed=random_seed,n=n,n_regions=2)
        roi_idx = roi.roi_idx
        roi_threshold,n_regions = [i for i in roi_idx.keys()],len(roi_idx)

        # Load locations obtained by the network design algorithm
        variance_threshold_ratio = [1.5,2.0]
        n_locations_monitored = 68
        file_loader = FileLoader(ReadRandomRoiRandomSubSampling())
        locations_monitored = file_loader.load(f'{results_path}SST/',n,signal_sparsity,variance_threshold_ratio,n_locations_monitored,
                                               random_seed_roi=random_seed,random_seed_subsampling=random_seed_subsampling)
        locations_unmonitored = [i for i in np.arange(n) if i not in locations_monitored]
        n_locations_unmonitored = len(locations_unmonitored)
        print(f'- Total number of potential locations: {n}\n- Number of monitored locations: {len(locations_monitored)}\n- Number of unmonitoreed locations: {len(locations_unmonitored)}')
                

        # compute coordinate error variance
        coordinate_error_variance_fullymonitored = np.diag(Psi@Psi.T)
        maxvariance_fullymonitored = coordinate_error_variance_fullymonitored.max()
        maxvariance_fullymonitored_ROI = [np.max(coordinate_error_variance_fullymonitored[i]) for i in roi_idx.values()]

        In = np.identity(n)
        C = In[locations_monitored,:]
        coordinate_error_variance_deployed = np.diag(Psi@np.linalg.inv(Psi.T@C.T@C@Psi)@Psi.T)
        maxvariance_deployed = coordinate_error_variance_deployed.max()
        maxvariance_deployed_ROI = [np.max(coordinate_error_variance_deployed[i]) for i in roi_idx.values()]
        
        # Load locations obtained by alternative algorithm
        try:
            n_locations_monitored_Boyd = 68
            file_loader = FileLoader(ReadRandomRoiRandomSubSampling_Boyd())
            locations_monitored_Boyd = file_loader.load(f'{results_path}SST/',n,signal_sparsity,variance_threshold_ratio,n_locations_monitored_Boyd,
                                                        random_seed_roi=random_seed,random_seed_subsampling=random_seed_subsampling)
            C = In[locations_monitored_Boyd,:]
            coordinate_error_variance_Boyd = np.diag(Psi@np.linalg.inv(Psi.T@C.T@C@Psi)@Psi.T)
            maxvariance_deployed_Boyd = coordinate_error_variance_Boyd.max()
            maxvariance_Boyd_ROI = [np.max(coordinate_error_variance_Boyd[i]) for i in roi_idx.values()]
        except:
            print('No locations file for alternative method')
            locations_monitored_Boyd = []
            coordinate_error_variance_Boyd = []
        
        # visualize coordinate error variance
        plots = Figures(save_path=results_path,marker_size=1,
                        fs_label=8,fs_ticks=8,fs_legend=4.5,fs_title=10,
                        show_plots=True)
        plots.curve_errorvariance_comparison(coordinate_error_variance_fullymonitored,coordinate_error_variance_deployed,
                                             variance_threshold_ratio,maxvariance_fullymonitored_ROI,
                                             n,n_locations_monitored,
                                             coordinate_error_variance_Boyd,roi_idx,n_locations_monitored_Boyd,
                                             N=n,signal_sparsity=signal_sparsity,
                                             save_fig=False)

        plt.show()
        sys.exit()
# %%
