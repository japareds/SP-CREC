#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:00:53 2023


@author: jparedes
"""
import os
import sys
import netCDF4
import pandas as pd
from dask import dataframe as dd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes ,mark_inset

from abc import ABC,abstractmethod

#%% functions
""" 
from pyspark.sql import SparkSession
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix

spark = SparkSession.builder.appName('noasst').config("spark.driver.memory", "10g").getOrCreate()

def create_dataframe(rootgrp,n_snapshots,n_locations,land_vals):
    data_zero = rootgrp.variables['sst'][0].data
    data_zero[data_zero==land_vals] = np.nan       
    data_reshaped = np.reshape(data_zero,newshape=n_locations,order='F')
    df_pd = pd.DataFrame(data_reshaped).dropna()
    df_pd.reset_index(inplace=True,drop=False)
    df = spark.createDataFrame(df_pd)
    
    for i in range(1,n_snapshots):
        data = rootgrp.variables['sst'][i].data
        data[data==land_vals] = np.nan
        data_reshaped = np.reshape(data,newshape=n_locations,order='F')
        df_pd = pd.DataFrame(data_reshaped,columns=[i]).dropna()
        df_pd.reset_index(inplace=True,drop=False)
        df_new = spark.createDataFrame(df_pd)
        df = df.join(df_new,on=['index'])        
    return df
def read_dataframe(fname):
    df = spark.read.parquet(fname)
    return df
def compute_SVD(df):
    df = df.drop('index')
    mat = RowMatrix(df.rdd.map(lambda v: Vectors.dense(v.rawFeatures)))
    mat = RowMatrix(df.rdd.map(lambda v: Vectors.fromML(v.rawFeatures)))

    svd = mat.computeSVD(k=10,computeU=True)

df = create_dataframe(rootgrp,10,n_locations,land_vals)
os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory 2g"
fname = f'{file_path}NOAA_SST.parquet'
df.write.format('parquet').mode('overwrite').option('header','true').save(fname)
spark.stop()

"""

def downsample_images(data:np.ndarray,scale_factor_x:float,scale_factor_y:float)->np.ndarray:
    """
    Doensample img to new resolution. Use either INTER_AREA or INTER_NEAREST

    Args:
        data (np.ndarray): 2D image
        scale_factor_x (float): shrinkage along horizontal axis
        scale_factor_y (float): shrinkage along vertical axis

    Returns:
        np.ndarray: new image and new mask
    """
    data_new = cv2.resize(data,(0,0),fx=scale_factor_x,fy=scale_factor_y,interpolation=cv2.INTER_NEAREST)
    mask_new = np.isnan(data_new)
    return data_new,mask_new
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

    
def create_dataframe(rootgrp,preprocessing:str='None',downsample:bool=False,scale_factor_x:float=0.1,scale_factor_y:float=0.1,windowing:bool=False,lon_min:float=130.,lon_max:float=230,lat_min:float=40,lat_max:float=140,lat_range:list=[],lon_range:list=[],save_dataset:bool=False)->pd.DataFrame:
    data = rootgrp.variables['sst'][0].data
    mask = rootgrp.variables['sst'][0].mask
    data[mask] = np.nan
    plots = Figures(save_path=file_path,marker_size=1,
                    fs_label=12,fs_ticks=7,fs_legend=6,fs_title=10,
                    show_plots=True)


    # transform first snapshot
    if preprocessing not in ['downsample,windowing,None']:
        raise ValueError(f'Specify image preprocessing technique.')
    
    elif preprocessing == 'downsample':
        print(f'Creating snapshots matrix from downsampled images. Scale factor: ({scale_factor_x:.2f},{scale_factor_y:.2f})')
        data_scaled,mask_scaled = downsample_images(data,scale_factor_x,scale_factor_y)
        plots.SST_map(data_scaled,save_fig=False,show_coords=False)
        plt.show()
        
    elif preprocessing== 'windowing':
        print(f'Creating snapshots matrix from windowed images.\n -Latitude range:({lat_min:.2f},{lat_max:.2f})\n -Longitude range:({lon_min:.2f},{lon_max:.2f})')
        data_scaled,mask_scaled = window_image(data,lat_range,lon_range,lon_min,lon_max,lat_min,lat_max)
        plots.dataset_map(data_scaled,save_fig=False)
        plt.show()

   
    # fill snapshots matrix    
    n_pixels = data_scaled.shape[0]*data_scaled.shape[1]
    n_locations = n_pixels - mask_scaled.sum()
    snapshots_matrix = np.zeros(shape=(n_locations,n_samples),dtype=np.float32)
    data_reshaped = np.reshape(data_scaled,newshape=n_pixels,order='F')
    data_reshaped = data_reshaped[~np.isnan(data_reshaped)]
    snapshots_matrix[:,0] = data_reshaped

    for i in np.arange(1,n_samples,1):
        data = rootgrp.variables['sst'][i].data
        data[mask] = np.nan
        if preprocessing=='downsample':
            data_scaled,mask_scaled = downsample_images(data,scale_factor_x,scale_factor_y)
        elif preprocessing == 'windowing':
            data_scaled,mask_scaled = window_image(data,lat_range,lon_range,lon_min,lon_max,lat_min,lat_max)
        elif preprocessing == 'None':
            data_scaled = data
            mask_scaled = np.isnan(data_scaled)

        data_reshaped = np.reshape(data_scaled,newshape=n_pixels,order='F')
        data_reshaped = data_reshaped[~np.isnan(data_reshaped)]
        snapshots_matrix[:,i] = data_reshaped
    
    date_range = pd.date_range(start='1981-09-01 00:00:00',freq='M',periods=n_samples)
    idx_measurement = np.where(~np.reshape(mask_scaled,newshape=n_pixels,order='F'))[0]
    idx_land = np.where(np.reshape(mask_scaled,newshape=n_pixels,order='F'))[0]

    df = pd.DataFrame(snapshots_matrix.T,dtype=np.float32,columns=idx_measurement,index=date_range)
    df.columns = df.columns.astype(str)
    if save_dataset:
        fname = f'{file_path}SST_month.parquet'
        df.to_parquet(fname)
        np.savetxt(f'{file_path}idx_measurements.csv',idx_measurement,delimiter=',')
        np.savetxt(f'{file_path}idx_land.csv',idx_land,delimiter=',')
        
        print(f'Files saved in dataset directory: {file_path}')
    return df

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
            self.backend = 'QT5Agg'#['QT5Agg','WebAgg']
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

    def SST_map(self,data,show_coords=False,save_fig=False):
        gmap = Basemap(lon_0=180, projection="kav7", resolution='c')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cmap = mpl.colormaps['Spectral'].resampled(64).reversed()
        cmap.set_bad('w')
        cmap_trunc = mpl.colors.LinearSegmentedColormap.from_list('trunc_cmap',cmap(np.linspace(0.,30.,100)),N=64)
        im = gmap.imshow(data,cmap=cmap,origin='lower',vmin=0,vmax=30)
        
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
        # new figure for cbar
        fig2, ax2 = plt.subplots(figsize=(6, 1), layout='constrained')
        cbar = fig2.colorbar(mpl.cm.ScalarMappable(cmap=cmap,norm=mpl.colors.Normalize(vmin=0, vmax=30)),cax=ax2,
                             orientation='horizontal', label=f'Temperature (ºC)',
                             location='top',extend='both')

        fig.tight_layout()
        if save_fig:
            fname = f'{self.save_path}SST_map.png'
            fig.savefig(fname,dpi=300,format='png',bbox_inches='tight')
            print(f'Figure saved at {fname}')
            fname= f'{self.save_path}SST_map_colorbar.png'
            fig2.savefig(fname,dpi=300,format='png')


        return fig
    def image_non_projected(self,image,save_fig=False):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cmap = mpl.colormaps['Spectral'].resampled(64).reversed()
        cmap.set_bad('w')
        im = ax.imshow(image,cmap=cmap,origin='lower',vmin=0,vmax=30)
        fig.tight_layout()

        fig2, ax2 = plt.subplots(figsize=(6, 1), layout='constrained')
        cbar = fig2.colorbar(mpl.cm.ScalarMappable(cmap=cmap,norm=mpl.colors.Normalize(vmin=0, vmax=30)),cax=ax2,
                             orientation='horizontal', label=f'Temperature (ºC)',
                             location='top',extend='both')

    def image_zoomed(self,image,lat_range_img,lon_range_img,lat_min_zoom,lat_max_zoom,lon_min_zoom,lon_max_zoom,tmin=0,tmax=30,show_zoom=True,save_fig=False):
        fig = plt.figure(constrained_layout=True)
        ax = fig.add_subplot(111)
        cmap = mpl.colormaps['Spectral'].resampled(128).reversed()
        cmap.set_bad('w')
        im = ax.imshow(image,cmap=cmap,origin='lower',vmin=tmin,vmax=tmax,aspect='equal')
        xticks = np.concatenate([np.arange(0,len(lon_range_img),4*30),[len(lon_range_img)-1]])
        yticks = np.concatenate([np.arange(0,len(lat_range_img),4*10),[len(lat_range_img)-1]])
        ax.set_xticks(xticks-0.5)
        # ax.set_xticklabels([int(np.round(i,0)) for i in lon_range_img[xticks]])
        ax.set_xticklabels([f'{int(np.round(i,0))}ºE' if int(np.round(i,0))<180 else f'{int(np.round(i,0))}º' if int(np.round(i,0))==180 else f'{int(np.round(360-i,0))}ºW' for i in lon_range_img[xticks]])
        ax.set_yticks(yticks-0.5)
        #ax.set_yticklabels([int(np.round(i,0)) for i in lat_range_img[yticks]])
        ax.set_yticklabels([f'{int(np.round(i,0))}ºN' if int(np.round(i,0))>0 else f'{int(np.round(i,0))}º' if int(np.round(i,0))==0 else f'{np.abs(int(np.round(i,0)))}ºS' for i in lat_range_img[yticks]])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_xlim(0-0.5,len(lon_range_img)-0.5)
        ax.set_ylim(0-0.5,len(lat_range_img)-0.5)
        if show_zoom:
            axins = zoomed_inset_axes(ax, zoom=2, loc='center',bbox_to_anchor=(280,415))
            axins.imshow(image, origin="lower",cmap=cmap,vmin=tmin,vmax=tmax)
            axins.set_xticks([])
            axins.set_yticks([])
            idx_lat = np.where(np.logical_and(lat_range_img>=lat_min_zoom,lat_range_img<=lat_max_zoom))[0]
            idx_lon = np.where(np.logical_and(lon_range_img>=lon_min_zoom,lon_range_img<=lon_max_zoom))[0]
            lat1,lat2 = idx_lat[0],idx_lat[-1]
            lon1,lon2 = idx_lon[0],idx_lon[-1]
            print(f'Lat min: {lat1}\t Lat max: {lat2}')
            print(f'Lon min: {lon1}\t Lon max: {lon2}')
            axins.set_xlim(lon1,lon2)
            axins.set_ylim(lat1,lat2)
            mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.2")#1: lower left (clockwise)
        else:
            idx_lat = np.where(np.logical_and(lat_range_img>=lat_min_zoom,lat_range_img<=lat_max_zoom))[0]
            idx_lon = np.where(np.logical_and(lon_range_img>=lon_min_zoom,lon_range_img<=lon_max_zoom))[0]
            lat1,lat2 = idx_lat[0],idx_lat[-1]
            lon1,lon2 = idx_lon[0],idx_lon[-1]
            rectangle = mpl.patches.Rectangle(xy=(lon1,lat1),width=lon2-lon1,height=lat2-lat1,edgecolor='k',facecolor='none',linewidth=1)
            ax.add_patch(rectangle)
            
            
        #fig.tight_layout()
        # color bar
        fig2, ax2 = plt.subplots(figsize=(10, 1), layout='constrained')
        cbar = fig2.colorbar(mpl.cm.ScalarMappable(cmap=cmap,norm=mpl.colors.Normalize(vmin=tmin, vmax=tmax)),cax=ax2,
                             orientation='horizontal',
                             location='top',extend='both')
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label('Temperature (ºC)',size=20)
        if save_fig:
            if show_zoom:
                fname = f'{self.save_path}MapZoomed_AreaLat{lat_range_img[0]}To{lat_range_img[-1]}_AreaLon{lon_range_img[0]}To{lon_range_img[-1]}_ZoomLat{lat_min_zoom}To{lat_max_zoom}_ZoomLon{lon_min_zoom}To{lon_max_zoom}.png'
            else:
                fname = f'{self.save_path}MapSquared_AreaLat{lat_range_img[0]}To{lat_range_img[-1]}_AreaLon{lon_range_img[0]}To{lon_range_img[-1]}_ZoomLat{lat_min_zoom}To{lat_max_zoom}_ZoomLon{lon_min_zoom}To{lon_max_zoom}.png'
            fig.savefig(fname,dpi=300,format='png',bbox_inches='tight')
            print(f'saved in {fname}')
            fname = f'{self.save_path}MapZoomed_AreaLat{lat_range_img[0]}To{lat_range_img[-1]}_AreaLon{lon_range_img[0]}To{lon_range_img[-1]}_ZoomLat{lat_min_zoom}To{lat_max_zoom}_ZoomLon{lon_min_zoom}To{lon_max_zoom}_cmap_Tmin{tmin}_Tmax{tmax}.png'
            fig2.savefig(fname,dpi=600,format='png')
            print(f'Colorbar saved in {fname}')
        

    def dataset_map(self,data,save_fig=False):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cmap = mpl.colormaps['Spectral'].resampled(64).reversed()
        cmap.set_bad('w')
        im = ax.imshow(data,cmap=cmap,origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right',size='5%', pad=0.05)
        cbar = fig.colorbar(im,cax=cax,orientation='vertical')
        cbar.set_ticks(np.arange(np.nanmin(data),np.nanmax(data),0.5))
        cbar.set_ticklabels([np.round(i,1) for i in cbar.get_ticks()])
        cbar.set_label('Temperature (ºC)')
        fig.tight_layout()
        if save_fig:
            fname = f'{self.save_path}SST_map.png'
            fig.savefig(fname,dpi=300,format='png',bbox_inches='tight')
            print(f'Figure saved at {fname}')
        return fig

#%%
if __name__=='__main__':
    abs_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/files/'
    fname = 'sst.mon.mean.nc'
    rootgrp =  netCDF4.Dataset(file_path+fname,'r')
    print(f'List of variables: {rootgrp.variables.keys()}')
    
    # total number of pixels and valid pixels with measurements
    land_vals = -9.96921e+36
    lat_range = np.array([rootgrp.variables['lat'][i].data.item() for i in range(rootgrp.variables['lat'].shape[0])])
    lon_range = np.array([rootgrp.variables['lon'][i].data.item() for i in range(rootgrp.variables['lon'].shape[0])])
    n_samples = rootgrp.variables['sst'].shape[0]
    n_pixels = rootgrp.variables['sst'].shape[1]*rootgrp.variables['sst'].shape[2]   
    n_land = rootgrp.variables['sst'][0].mask.sum()

    # show example figure
    sample_plot = 25#494 # <=383 for samples from 1981-09 until 2013-08
    date_range = pd.date_range(start='1981-09-01 00:00:00',freq='M',periods=n_samples)
    print(f'Showing measurements for {date_range[sample_plot]}')
    plots = Figures(save_path=file_path,figx=3.5,figy=2.5,marker_size=1,
                    fs_label=12,fs_ticks=7,fs_legend=6,fs_title=10,
                    show_plots=True)
    data = rootgrp.variables['sst'][sample_plot].data
    mask = rootgrp.variables['sst'][0].mask
    data[mask] = np.nan
    plots.SST_map(data,show_coords=False,save_fig=False)

    # windowing parameters
    windowing = Windowing()
    lat_min,lat_max = -20,20#-30,30 # the difference (*4) determine number of rows
    lon_min,lon_max = 120,300#120,300 # the difference (*4) determine number of columns
    img_windowed,_ = windowing.filter(image=data,lat_range=lat_range,lon_range=lon_range,
                                      lat_min=lat_min,lat_max=lat_max,
                                      lon_min=lon_min,lon_max=lon_max)
    lat_range_img,lon_range_img = windowing.get_coordinates_range(lat_range=lat_range,lat_min=lat_min,lat_max=lat_max,
                                                                  lon_range=lon_range,lon_min=lon_min,lon_max=lon_max)
    print(f'Tempereature range in windowed image: {np.nanmin(img_windowed):.2f} - {np.nanmax(img_windowed):.2f}')
    lat_min_zoom,lat_max_zoom = -5,5
    lon_min_zoom,lon_max_zoom = 215,240
    img_zoom,_ = windowing.filter(image=data,lat_range=lat_range,lon_range=lon_range,
                                  lat_min=lat_min_zoom,lat_max=lat_max_zoom,
                                  lon_min=lon_min_zoom,lon_max=lon_max_zoom)


    plots.image_zoomed(img_windowed,
                       lat_range_img =lat_range_img,lon_range_img=lon_range_img,
                       lat_min_zoom=lat_min_zoom,lat_max_zoom=lat_max_zoom,
                       lon_min_zoom=lon_min_zoom,lon_max_zoom=lon_max_zoom,
                       tmin=20,tmax=30,show_zoom=False,save_fig=True)
    plt.show()
    sys.exit()
    #downsampling parameters
    scale_factor_x = 100/data.shape[1]
    scale_factor_y = 100/data.shape[0]

    windowing = False
    if windowing:
        df = create_dataframe(rootgrp,preprocessing='windowing',
                              lon_min=lon_min,lon_max=lon_max,
                              lat_min=lat_min,lat_max=lat_max,
                              lat_range=lat_range,lon_range=lon_range,
                              save_dataset=False)
    downsampling = True
    if downsampling:
        df = create_dataframe(rootgrp,preprocessing='downsampling',
                              scale_factor_x=scale_factor_x,scale_factor_y=scale_factor_y,
                              save_dataset=False)

    print(f'Genrated dataset has {df.shape[0]} measurements for {df.shape[1]} locations')
