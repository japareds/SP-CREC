#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
from abc import ABC, abstractmethod
import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
#%%
# load files
class Files(ABC):
    @abstractmethod
    def load(self,**kwargs):
        raise NotImplementedError
class DictFiles(Files):
    def load(self,results_path:str,string:str,N:int,batch_size:int,b:int,b_total:int):
        fname = f'{results_path}{string}_N{N}_batchSize{batch_size}_batch{b}of{b_total}.pkl'
        with open(fname,'rb') as f:
            dict_file = pickle.load(f)
        return dict_file
    
# generate figures
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

    def curve_cumulative_execution_times_batch(self,time_batches,batch_sizes,deg_polyfit=0,time_format='seconds',save_fig=False):
        """ Plot computational time of the IRNet algorithm per batch size. Time is in seconds but can bre presented in other formats""" 
        fig = plt.figure()
        ax = fig.add_subplot(111)
        colors = ['#cb4335', '#dc7633', '#196f3d','#1a5276','#6c3483', '#566573']
        for b,c in zip(batch_sizes,colors):
            time_batches_ = time_batches[b]
            n_indices = [i*b for i in time_batches_.keys() if time_batches_[i]!=0]
            if time_format =='seconds':
                exec_times_batch = [i for i in time_batches_.values() if i !=0]
            elif time_format == 'hours':
                exec_times_batch = [i/3600 for i in time_batches_.values() if i !=0]
            elif time_format == 'days':
                exec_times_batch = [i/(3600*24) for i in time_batches_.values() if i !=0]
            ax.plot(n_indices,np.cumsum(exec_times_batch),color=c,label='$|\mathcal{B}|=$'+f'{b}')
            # polynomial fit
            if deg_polyfit!=0:
                coeffs = np.polyfit(x=n_indices,y=np.cumsum(exec_times_batch),deg=deg_polyfit)
                exec_times_batch_fit = [sum(coeff*(x**(deg_polyfit-p)) for p,coeff in enumerate(coeffs)) for x in n_indices]
                
        xrange = np.arange(0,N+1e3,1e3)
        ax.set_xticks(xrange)
        ax.set_xlim(0,N)
        ax.set_xlabel('$|\mathcal{G}_k|$')
        if time_format == 'seconds':
            yrange = np.arange(0,400e3+50e3,50e3)
            ax.set_yticks(yrange)
            ax.ticklabel_format(style='sci',scilimits=(0,4),axis='y')
            ax.set_ylim(0,400e3)
            ax.set_ylabel('Computational time (s)')
        elif time_format == 'hours':
            yrange = np.arange(0,120+20,20)
            ax.set_yticks(yrange)
            ax.set_ylim(0,120)
            ax.set_ylabel('Computational time (hours)')
        elif time_format == 'days':
            yrange = np.arange(0,5+1,1)
            ax.set_yticks(yrange)
            ax.set_ylim(0,5)
            ax.set_ylabel('Computational time (days)')

        ax.legend(loc='upper left',ncol=2,columnspacing=0.5,framealpha=0.1,
                  handlelength=0.5,handletextpad=0.1,bbox_to_anchor=(-0.03,1.05))
        fig.tight_layout()

        if save_fig:
            fname = f'{self.save_path}Curve_CumulativeComputationalTime_LargeScaleDeployment_batches{batch_sizes}.png'
            fig.savefig(fname,dpi=600,format='png',bbox_inches='tight')
            print(f'Saved in {fname}')
    
    def curve_cumulative_number_monitored_locations_batch(self,minibatches_indices_monitored,batch_sizes,save_fig=False):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        colors = ['#cb4335', '#dc7633', '#196f3d','#1a5276','#6c3483', '#566573']
        #colors = ['orange','#148f77','#76448a','#7b241c','#1a5276','k']
        for b,c in zip(batch_sizes,colors):
            minibatches_indices_monitored_ = minibatches_indices_monitored[b]
            n_indices = [i*b for i in minibatches_indices_monitored_ if len(minibatches_indices_monitored_[i]) !=0]
            n_monitored_indices_batch = [len(i) for i in minibatches_indices_monitored_.values() if len(i) !=0]
            ax.plot(n_indices,np.cumsum(n_monitored_indices_batch),color=c,label='$|\mathcal{B}|$='+f'{b}')
        xrange = np.arange(0,N+1e3,1e3)
        ax.set_xticks(xrange)
        ax.set_xlim(0,N)
        yrange = np.arange(0,600+100,100)
        ax.set_yticks(yrange)
        ax.set_ylim(0,600)
        ax.set_xlabel('$|\mathcal{G}_k|$')
        ax.set_ylabel('Cumulative $|\mathcal{M}|$')
        #ax.legend(loc='center',ncol=2,framealpha=0.1,bbox_to_anchor=(0.5,1.2))
        ax.legend(loc='upper left',ncol=2,columnspacing=0.5,framealpha=0.1,
                  handlelength=0.5,handletextpad=0.1,bbox_to_anchor=(-0.03,1.05))
        fig.tight_layout()
        if save_fig:
            fname =f'{self.save_path}Curve_CumulativeNumberMonitoredLocations_LargeScaleDeployment_batches{batch_sizes}.png'
            fig.savefig(fname,dpi=600,format='png',bbox_inches='tight')
            print(f'Saved in {fname}')

    def curve_monitored_locations_vs_batchsize(self,minibatches_indices_monitored,time_batches,batch_sizes,time_format='seconds',save_fig=False):
        n_monitored_indices_batchsize = {el:0 for el in batch_sizes}
        total_computational_time_batchsize = {el:0 for el in batch_sizes}
        for b in batch_sizes:
            minibatches_indices_monitored_ = minibatches_indices_monitored[b]
            n_monitored_indices_batchsize[b] = np.sum([len(i) for i in minibatches_indices_monitored_.values()])
            time_batches_ = time_batches[b]
            if time_format == 'seconds':
                total_computational_time_batchsize[b] = np.sum([i for i in time_batches_.values()])
            elif time_format == 'hours':
                total_computational_time_batchsize[b] = np.sum([i/3600 for i in time_batches_.values()])
            elif time_format == 'days':
                total_computational_time_batchsize[b] = np.sum([i/(3600*24) for i in time_batches_.values()])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # number of deployed sensors
        l1 = ax.plot(batch_sizes,[i for i in n_monitored_indices_batchsize.values()],color='#1a5276',label='$|\mathcal{M}|$')
        ax.set_xlabel('$|\mathcal{B}|$')
        ax.set_xticks(batch_sizes)
        ax.set_xlim(0,batch_sizes[-1])
        ax.set_ylabel('Total $|\mathcal{M}|$')
        yrange = np.arange(300,600+100,100)
        ax.set_yticks(yrange)
        ax.set_ylim(300,600)
        
        # computational time
        ax2 = ax.twinx()
        l2 = ax2.plot(batch_sizes,[i for i in total_computational_time_batchsize.values()],color='orange',label='Computational time')
        if time_format == 'seconds':
            ax2.set_ylabel('Computational time (s)')
            ax2.ticklabel_format(style='sci',scilimits=(0,4),axis='y')
            yrange = np.arange(0,400e3+50e3,50e3)
            ax2.set_yticks(yrange)
            ax2.set_ylim(0,400*1e3)
        elif time_format == 'hours':
            yrange = np.arange(0,120+20,20)
            ax2.set_yticks(yrange)
            ax2.set_ylim(0,120)
            ax2.set_ylabel('Computational time (hours)')
        elif time_format == 'days':
            yrange = np.arange(0,5+1,1)
            ax2.set_yticks(yrange)
            ax2.set_ylim(0,5)
            ax2.set_ylabel('Computational time (days)')


        # legend
        lines = l1+l2
        labels = [l.get_label() for l in lines]
        ax.legend(lines,labels,loc='center',ncol=1,framealpha=0.1,
                  handlelength=0.5,handletextpad=0.1,columnspacing=0.2,bbox_to_anchor=(0.53,0.85))
        fig.tight_layout()

        if save_fig:
            fname =f'{self.save_path}Curve_TotalMonitoredLocations_TotalTime_LargeScaleDeployment_batches{batch_sizes}.png'
            fig.savefig(fname,dpi=600,format='png',bbox_inches='tight')
            print(f'Saved in {fname}')



#%%
if __name__=='__main__':
    abs_path = os.path.dirname(os.path.realpath(__file__))
    files_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/files/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/test/'    
    # batches solved
    N = 4000

    batch_size = [4,10,20,50,80,100]
    b=[1000,400,200,80,50,40]
    b_total=[int(N/i) for i in batch_size]
    time_batches = {el:[] for el in batch_size}
    minibatches_indices_monitored = {el:[] for el in batch_size}
    # plot computational time 
    dict_files = DictFiles()
    for i in range(len(batch_size)):
        time_batches[batch_size[i]] = dict_files.load(results_path=results_path,string='time_batches',N=N,batch_size=batch_size[i],b=b[i],b_total=b_total[i])
        minibatches_indices_monitored[batch_size[i]] = dict_files.load(results_path=results_path,string='minibatches_indices_monitored',N=N,batch_size=batch_size[i],b=b[i],b_total=b_total[i])

    plots = Figures(save_path=results_path,figx=3.5,figy=2.5,
                    marker_size=1,
                    fs_label=12,fs_ticks=9,fs_legend=10.5,fs_title=10,
                    show_plots=True)
    plots.curve_monitored_locations_vs_batchsize(minibatches_indices_monitored,time_batches,batch_size,
                                                 time_format='days',save_fig=False)
    # plots of cumulative metrics vs batch size
    plots.curve_cumulative_execution_times_batch(time_batches,batch_size,time_format='days',save_fig=False)
    plots.curve_cumulative_number_monitored_locations_batch(minibatches_indices_monitored,batch_size,save_fig=False)
    
    plt.show()
    