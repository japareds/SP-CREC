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
import sys
import warnings
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from geopandas import GeoDataFrame

import sensor_placement as sp
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

#%% ROI classes
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
#%% file writer classes
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
        print(f'Reading file {fname}')
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
        print(f'Reading file {fname}')
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

#%% functions
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

#%% figures
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

    def curve_errorvariance_comparison(self,errorvar_fullymonitored:list,errorvar_reconstruction:list,variance_threshold_ratio:float,worst_coordinate_variance_fullymonitored:float,n:int,n_sensors:int,errorvar_reconstruction_Dopt:list=[],errorvar_reconstruction_Dopt_fails:list=[],roi_idx:dict={},n_sensors_Dopt:int=0,n_sensors_Dopt_fails:int=0,method:str='random_based',random_seed:int=0,alternative_method_label='Joshi-Boyd method',save_fig:bool=False) -> plt.figure:
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
                coordinate_error_variance_Dopt_fails_sorted = np.concatenate([errorvar_reconstruction_Dopt_fails[i] for i in roi_idx.values()])

                ax.plot(coordinate_error_variance_Dopt_sorted,color='orange',label=f'{alternative_method_label}',alpha=0.8)
                ax.plot(coordinate_error_variance_Dopt_fails_sorted,color='orange',label='',alpha=0.8,linestyle='--')
            ax.plot(coordinate_error_variance_design_sorted,color='#1a5276',label=f'SP-CREC')
            
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
                      handlelength=0.7,handletextpad=0.1,columnspacing=0.2,
                      bbox_to_anchor=(0.5,0.88))
            #fig.tight_layout()
            if save_fig:
                #fname = f'{self.save_path}Curve_errorVariance_Threshold{variance_threshold_ratio}_Nsensors{n_sensors}_NsensorsDopt{n_sensors_Dopt}_NsensorsROIDopt_{n_sensors_roi}.png'
                if method == 'random_based':
                    fname = f'{self.save_path}Curve_errorVariance_VarThreshold{variance_threshold_ratio}_Nsensors{n_sensors}_NsensorsDopt{n_sensors_Dopt}_Nsensors{alternative_method_label}Fails{n_sensors_Dopt_fails}_randomSeed{random_seed}.png'
                else:
                    fname = f'{self.save_path}Curve_errorVariance_VarThreshold{variance_threshold_ratio}_Nsensors{n_sensors}_Nsensors{alternative_method_label}{n_sensors_Dopt}.png'
                fig.savefig(fname,dpi=300,format='png',bbox_inches='tight')
                print(f'Figure saved at {fname}')


#%% main
if __name__ == '__main__':
    """ load dataset to use """
    abs_path = os.path.dirname(os.path.realpath(__file__))
    files_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/files/catalonia/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/test/'

    synthetic_dataset = False
    pollutant = 'O3'
    start_date = '2011-01-01'
    end_date = '2022-12-31'
    N=48
    if synthetic_dataset:
        files_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/files/synthetic/'
    else:
        dataset = Dataset(pollutant,N,start_date,end_date,files_path,synthetic_dataset)
        dataset.load_dataset()
        dataset.check_dataset()
    """ sort stations based on distance"""
    dataset.distances = dataset.coordinates_distances.loc['Ciutadella']
    dataset.distances.sort_values(ascending=True,inplace=True)
    dataset.ds = dataset.ds.loc[:,[f'O3_{i}' for i in dataset.distances.index if f'O3_{i}' in dataset.ds.columns]]
    print(f'Order of dataset locations: {dataset.ds.columns}')

    """ snapshots matrix and low-rank decomposition """
    # train/val/test split
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10
    X_train, X_test = train_test_split(dataset.ds, test_size= 1 - train_ratio,shuffle=False,random_state=92)
    X_val, X_test = train_test_split(X_test, test_size=test_ratio/(test_ratio + validation_ratio),shuffle=False,random_state=92) 
    print(f'Dataset matrix summary:\n {train_ratio} of dataset for training set with {X_train.shape[0]} measurements from {X_train.index[0]} until {X_train.index[-1]}\n {validation_ratio} of dataset for validation set with {X_val.shape[0]} measurements from {X_val.index[0]} until {X_val.index[-1]}\n {test_ratio} of measuerements for testing set with {X_test.shape[0]} measurements from {X_test.index[0]} until {X_test.index[-1]}')
    
    print('\nReconstructing signal from sparse measurements.\nTwo algorithms are used:\n- Network design\n- Joshi-Boyd (D-optimal) sensor selection.')
    # low-rank decomposition
    snapshots_matrix_train = X_train.to_numpy().T
    snapshots_matrix_val = X_val.to_numpy().T
    snapshots_matrix_test = X_test.to_numpy().T
    snapshots_matrix_train_centered = snapshots_matrix_train - snapshots_matrix_train.mean(axis=1)[:,None]
    snapshots_matrix_val_centered = snapshots_matrix_val - snapshots_matrix_train.mean(axis=1)[:,None]
    snapshots_matrix_test_centered = snapshots_matrix_test - snapshots_matrix_train.mean(axis=1)[:,None]
    U,sing_vals,Vt = np.linalg.svd(snapshots_matrix_train_centered,full_matrices=False)
    n = U.shape[0]
    print(f'Training snapshots matrix has dimensions {snapshots_matrix_train_centered.shape}.\nLeft singular vectors matrix has dimensions {U.shape}\nRight singular vectors matrix has dimensions [{Vt.shape}]\nNumber of singular values: {sing_vals.shape}')
    # specify signal sparsity and network parameters
    if n==44:
        signal_sparsity = 28#[28,34] # 28(30) for RMSE < 5 // 34 for energy>0.9
    elif n==48:
        signal_sparsity =36#[30,36] # 30 for RMSE < 0.5 // 36 for energy>0.9
    else:
        raise ValueError(f'No network with potential number of locations n:{n}')

    Psi = U[:,:signal_sparsity]
    print(f'Basis shape: {Psi.shape}.')

    """ Set homogeneous/heterogeneous threshold applied for designing network
        Change corresponding Random class for reading and splitting ROIs
        paper results with 2 ROIs random split seed = 0 VarRatio=[1.5,2.0]
        paper results with 3 ROIs random subsplit seed = 0 subsplit seed = 0 varRatio = [1.1,1.5,2.0]
    """
    homogeneous_threshold = False
    num_rois = 2
    # random ROIs parameters
    if num_rois == 2:
        random_seed = 0
        roi = ROI(RandomRoi()) #[RandomRoi(),SubSplitRandomRoi()]
        roi.define_rois(seed=random_seed,n=n,n_regions=num_rois)
        variance_threshold_ratio = [1.5,2.0]
    elif num_rois == 3:
        random_seed = 0
        seed_subsplit=0
        rois_split=[0]
        roi = ROI(SubSplitRandomRoi()) #[RandomRoi(),SubSplitRandomRoi()]
        roi.define_rois(seed=random_seed,n=n,n_regions_original=2,rois_split=rois_split,n_regions_subsplit=2,seed_subsplit=seed_subsplit)
        variance_threshold_ratio = [1.1,1.5,2.0]
    
    roi_idx = roi.roi_idx
    roi_threshold,n_regions = [i for i in roi_idx.keys()],len(roi_idx)
    
    # number of deployed sensors by both algorithms
    if num_rois == 2:
        n_sensors = 38
        n_sensors_Dopt = 38
        n_sensors_Dopt_fails = 38
    elif num_rois == 3:
        n_sensors = 38
        n_sensors_Dopt = 40
        n_sensors_Dopt_fails = 38
        
    # coordinate error variance for fully monitored network
    error_variance_fullymonitored = Psi@np.linalg.inv(Psi.T@Psi)@Psi.T
    error_variance_fullymonitored_roi = [(Psi@np.linalg.inv(Psi.T@Psi)@Psi.T)[i] for i in roi_idx.values()]
    worst_variance_fullymonitored_roi = [np.max(np.diag(Psi@np.linalg.inv(Psi.T@Psi)@Psi.T)[i]) for i in roi_idx.values()]
        
    # load monitored locations indices
    if num_rois == 2:
        reader = ReadLocations(ReadRandomFile())
        locations_monitored = reader.load_locations(file_path = f'{results_path}NetworkDesign/heterogeneous/random_based/',
                                                    n=n,signal_sparsity=signal_sparsity,
                                                    variance_threshold_ratio = variance_threshold_ratio,n_sensors = n_sensors,
                                                    random_seed=random_seed,
                                                    signal_threshold_ratio = variance_threshold_ratio)

    elif num_rois == 3:
        reader = ReadLocations(ReadSplitRandomFile())#[ReadRandomFile(),ReadSplitRandomFile()]
        locations_monitored = reader.load_locations(file_path = f'{results_path}NetworkDesign/heterogeneous/random_based/',n=n,signal_sparsity=signal_sparsity,
                                                    variance_threshold_ratio = variance_threshold_ratio,n_sensors = n_sensors,
                                                    random_seed=random_seed,seed_subsplit=seed_subsplit,rois_split=rois_split,
                                                    signal_threshold_ratio = variance_threshold_ratio)

    locations_unmonitored = [i for i in np.arange(n) if i not in locations_monitored]
    n_locations_monitored = len(locations_monitored)
    n_locations_unmonitored = len(locations_unmonitored)
    print(f'- Total number of potential locations: {n}\n- Number of monitored locations: {len(locations_monitored)}\n- Number of unmonitored locations: {len(locations_unmonitored)}')
    for idx in roi_idx.values():
        print(f'ROI with {len(idx)} potential locations. Network design algorithm has {np.isin(idx,locations_monitored).sum()} sensors in ROI.')
    # get worst coordinate error variance analytically (from formula)
    In = np.identity(n)
    C = In[locations_monitored,:]
    error_variance_design = Psi@np.linalg.inv(Psi.T@C.T@C@Psi)@Psi.T
    worst_variance_design = np.diag(error_variance_design).max()
    rmse_design = np.sqrt(np.trace(Psi@np.linalg.inv(Psi.T@C.T@C@Psi)@Psi.T)/n)
    
    # get worst coordinate error variance from empirical signal reconstruction. The signal is projected onto low-dimensional subspace
       
    print('Reconstructed signal results.')
    X_test_proj = (Psi@Psi.T@X_test.T).T
    X_test_proj.columns = X_test.columns
    X_test_proj.index = X_test.index
    X_test_proj_noisy = add_noise_signal(X_test_proj,seed=42,var=1.0)
    rmse_design_reconstruction,errorvar_design_reconstruction = signal_reconstruction_regression(Psi,locations_monitored,X_test=X_test_proj,X_test_measurements=X_test_proj_noisy,projected_signal=True)
    rmse_fullymonitored_reconstruction,errorvar_fullymonitored_reconstruction = signal_reconstruction_regression(Psi,np.arange(n),X_test=X_test_proj,X_test_measurements=X_test_proj_noisy,projected_signal=True)            
    
    # reconstruction using alternative method: Joshi-Boyd
    try:
        print(f'Loading alternative sensor placement locations obtained with Dopt method.')
        if homogeneous_threshold:
            fname = f'{results_path}Dopt/homogeneous/SensorsLocations_N{n}_S{signal_sparsity}_nSensors{n_locations_monitored}.pkl'
            with open(fname,'rb') as f:
                locations_monitored_Dopt = np.sort(pickle.load(f))
        else:
            if num_rois == 2:
                reader = ReadLocations(ReadRandomFileBoyd())
                locations_monitored_Dopt = reader.load_locations(file_path = f'{results_path}FeasibilityOpt/heterogeneous/random_based/',n=n,signal_sparsity=signal_sparsity,
                                                                variance_threshold_ratio = variance_threshold_ratio,n_sensors_Dopt = n_sensors_Dopt,
                                                                random_seed=random_seed)

                locations_monitored_Dopt_fails = reader.load_locations(file_path = f'{results_path}FeasibilityOpt/heterogeneous/random_based/',n=n,signal_sparsity=signal_sparsity,
                                                                        variance_threshold_ratio = variance_threshold_ratio,n_sensors_Dopt = n_sensors_Dopt_fails,
                                                                        random_seed=random_seed)

            elif num_rois == 3:
                reader = ReadLocations(ReadSplitRandomFileBoyd())#ReadRandomFileBoyd,ReadSplitRandomFileBoyd
                locations_monitored_Dopt = reader.load_locations(file_path = f'{results_path}FeasibilityOpt/heterogeneous/random_based/',n=n,signal_sparsity=signal_sparsity,
                                                                variance_threshold_ratio = variance_threshold_ratio,n_sensors_Dopt = n_sensors_Dopt,
                                                                random_seed=random_seed,seed_subsplit=seed_subsplit,rois_split=rois_split)

                locations_monitored_Dopt_fails = reader.load_locations(file_path = f'{results_path}FeasibilityOpt/heterogeneous/random_based/',n=n,signal_sparsity=signal_sparsity,
                                                                        variance_threshold_ratio = variance_threshold_ratio,n_sensors_Dopt = n_sensors_Dopt_fails,
                                                                        random_seed=random_seed,seed_subsplit=seed_subsplit,rois_split=rois_split)

        # JB
        locations_unmonitored_Dopt = [i for i in np.arange(n) if i not in locations_monitored_Dopt]
        C_Dopt = In[locations_monitored_Dopt,:]
        error_variance_Dopt = Psi@np.linalg.inv(Psi.T@C_Dopt.T@C_Dopt@Psi)@Psi.T
        worst_variance_Dopt = np.diag(error_variance_Dopt).max()
        rmse_Dopt = np.sqrt(np.trace(error_variance_Dopt)/n)
        rmse_Dopt_reconstruction,errorvar_Dopt_reconstruction = signal_reconstruction_regression(Psi,locations_monitored_Dopt,X_test=X_test_proj,X_test_measurements=X_test_proj_noisy,projected_signal=True)
        # JB fails
        locations_unmonitored_Dopt_fails = [i for i in np.arange(n) if i not in locations_monitored_Dopt_fails]
        C_Dopt_fails = In[locations_monitored_Dopt_fails,:]
        error_variance_Dopt_fails = Psi@np.linalg.inv(Psi.T@C_Dopt_fails.T@C_Dopt_fails@Psi)@Psi.T
        worst_variance_Dopt_fails = np.diag(error_variance_Dopt_fails).max()
        rmse_Dopt_fails = np.sqrt(np.trace(error_variance_Dopt_fails)/n)
        rmse_Dopt_reconstruction_fails,errorvar_Dopt_reconstruction_fails = signal_reconstruction_regression(Psi,locations_monitored_Dopt_fails,X_test=X_test_proj,X_test_measurements=X_test_proj_noisy,projected_signal=True)

    except:
            print(f'No Dopt sensor placement file in for worse error variance threshold {variance_threshold_ratio} and num sensors {n_locations_monitored}')
            errorvar_reconstruction_Dopt = []

    print('Results of variance-covariance matrix comparison\nROIs with different threshold design')
    print(f'Number of ROIs: {n_regions}.\nROIs thresholds: {roi_threshold}\nNumber of elements in each ROI: {[len(i) for i in roi_idx.values()]}\nNetwork designed with threshold ratios: {variance_threshold_ratio}')
    print(f'Fully monitored network:\n- worst coordinate error variance per ROI: {[np.round(i,3) for i in worst_variance_fullymonitored_roi]}')
    print(f'Maximum variance tolerated in each ROI: {[np.round(w*t,3) for w,t in zip(worst_variance_fullymonitored_roi,variance_threshold_ratio)]}')
    print(f'Designed network\n- worst coordinate error variance: {[np.round(np.max(np.diag(error_variance_design)[i]),3) for i in roi_idx.values()]}')
    print(f'D-optimality\n- worst coordinate error variance: {[np.round(np.max(np.diag(error_variance_Dopt)[i]),3) for i in roi_idx.values()]}')
    print(f'Network design and D-opt methods share {np.isin(locations_monitored,locations_monitored_Dopt).sum()} locations out of {len(locations_monitored)} monitored locations')

    print(f'Trace of network design locations: {np.trace(error_variance_design):.3f}')
    print(f'Trace of Boyd algorithm: {np.trace(error_variance_Dopt):.3f}')
    print(f'Trace of Failed Boyd algorithm: {np.trace(error_variance_Dopt_fails):.3f}')
    print(f'Trace of network design locations on regressor: {np.trace(np.linalg.inv(Psi.T@C.T@C@Psi)):.3f}')
    print(f'Trace of Boyd algorithm on regressor: {np.trace(np.linalg.inv(Psi.T@C_Dopt.T@C_Dopt@Psi)):.3f}')
    print(f'Trace of Failed Boyd algorithm on regressor: {np.trace(np.linalg.inv(Psi.T@C_Dopt_fails.T@C_Dopt_fails@Psi)):.3f}')

    # visualize                
    plots = Figures(save_path=results_path,marker_size=1,
        figx=3.5,figy=2.5,
        fs_label=11,fs_ticks=11,fs_legend=11,fs_title=10,
        show_plots=True)

    plots.geographical_network_visualization(map_path=f'{files_path}ll_autonomicas_inspire_peninbal_etrs89/',df_coordinates=dataset.coordinates.reindex(dataset.distances.index),
                                                locations_monitored=locations_monitored,roi_idx=roi_idx,show_legend=True,
                                                show_deployed_sensors=True,save_fig=False)

    plots = Figures(save_path=results_path,marker_size=1,
                    figx=3.5,figy=2.5,
                    fs_label=11,fs_ticks=10,fs_legend=9,fs_title=10,
                    show_plots=True)
    """ 
    plots.curve_errorvariance_comparison(np.diag(error_variance_fullymonitored),np.diag(error_variance_design),
                                        variance_threshold_ratio,worst_variance_fullymonitored_roi,
                                        n,n_locations_monitored,
                                        np.diag(error_variance_Dopt),roi_idx,n_sensors_Dopt,
                                        method='',random_seed=random_seed,save_fig=False)
    """
    plots.curve_errorvariance_comparison(errorvar_fullymonitored=np.diag(error_variance_fullymonitored),
                                         errorvar_reconstruction=np.diag(error_variance_design),
                                         variance_threshold_ratio=variance_threshold_ratio,worst_coordinate_variance_fullymonitored=worst_variance_fullymonitored_roi,
                                         n=n,n_sensors=n_locations_monitored,
                                         errorvar_reconstruction_Dopt=np.diag(error_variance_Dopt),errorvar_reconstruction_Dopt_fails=np.diag(error_variance_Dopt_fails),
                                         roi_idx=roi_idx,n_sensors_Dopt=n_sensors_Dopt,n_sensors_Dopt_fails=n_sensors_Dopt_fails,
                                         method='',random_seed=random_seed,
                                         alternative_method_label='Margin method',
                                         save_fig=True)
    plt.show()
    sys.exit()
