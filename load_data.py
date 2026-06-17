import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
from scipy.special import ive, binom
from scipy.io import loadmat
from scipy.optimize import curve_fit
from scipy.signal.windows import tukey
import ufss
from order_separation import SeparateOrders

class loadStage:
    def __init__(self,*,root='Stage'):
        self.T = loadmat(os.path.join(root,'T_axis_Stage.mat'))['T_axis'][0,:]
        self.data = loadmat(os.path.join(root,'DataStage.mat'))['threeDmatrix']
        self.Is = loadmat(os.path.join(root,'Stage/I_Stage.mat'))['I_Stage'][0,:]
        self.wt = loadmat(os.path.join(root,'wt_axisStage.mat'))['wt_axis'][:,0]
        self.tau = loadmat(os.path.join(root,'wtau_axis_Stage.mat'))['tau_axis'][0,:]

class loadShaper:
    def __init__(self,*,root='Shaper'):
        self.T = loadmat(os.path.join(root,'T_axisShaper.mat'))['T_axis'][0,:]
        self.data = loadmat(os.path.join(root,'DataShaper.mat'))['threeDmatrix']
        self.Is = loadmat(os.path.join(root,'I_Shaper.mat'))['I_Shaper'][0,:]
        self.wt = loadmat(os.path.join(root,'wt_axis_Shaper.mat'))['wt_axis'][:,0]
        self.tau = loadmat(os.path.join(root,'tau_axis_Shaper.mat'))['tau_axis'][0,:]

class loadDSQBC2:
    def __init__(self,*,root='dSQBC2'):
        self.data = loadmat(os.path.join(root,'2_16_05_07_dSQBC_Matrix_PowerCy_1_KM_MA_4StepsLin.mat'))['Matrix_twoD']
        self.tau = loadmat(os.path.join(root,'dSQBC2/2_tau_axis.mat'))['tau_axis'][0,:]
        self.wt = loadmat(os.path.join(root,'dSQBC2/2_wt_axis.mat'))['wt_axis'][:,0]
        self.Is = np.array([5,4,3,2,0.3])

class loadDSQBC1:
    def __init__(self,*,root='dSQBC1'):
        self.data = loadmat(os.path.join(root,'1-16_05_03_dSQBC_Matrix_PowerCy_1_KM_MA_4StepsLin.mat'))['Matrix_twoD']
        self.tau = loadmat(os.path.join(root,'dSQBC1/1-tau_axis.mat'))['tau_axis'][0,:]
        self.wt = loadmat(os.path.join(root,'dSQBC1/1-wt_axis.mat'))['wt_axis'][:,0]
        self.Is = np.array([4.2,3.2,2.2,1.2,0.3])

class loadDSQBC3:
    def __init__(self,*,root='dSQBC3'):
        self.data = loadmat(os.path.join(root,'24_05_23_173_dSQBC_Matrix_PowerCy_1_KM_MA_4StepsLin.mat'))['Matrix_twoD']
        self.tau = loadmat(os.path.join(root,'dSQBC3/tau_axis_24_05_23_17.mat'))['tau_axis'][0,:]
        self.wt = loadmat(os.path.join(root,'dSQBC3/wt_axis_24_05_23_17.mat'))['wt_axis'][:,0]
        self.Is = np.array([6,5,4,3,1.5])

class loadDSQBC123:
    def __init__(self):
        l1 = loadDSQBC1()
        l2 = loadDSQBC2()
        l3 = loadDSQBC3()
        self.Is = np.concatenate((np.array([0.3]),l1.Is[:-1],l2.Is[:-1],l3.Is))
        data_low = (l1.data[:,0:1,:] + l2.data[:,0:1,:])/2
        print(data_low.shape)
        self.data = np.concatenate((data_low,l1.data[:,:-1,:], 
                                    l2.data[:,:-1,:], l3.data),axis=1)
        sort_inds = self.Is.argsort()[::-1]
        self.Is = self.Is[sort_inds]
        self.data = self.data[:,sort_inds,:]
        self.tau = l1.tau
        self.wt = l1.wt        

class loadDSQBC4:
    '''Data from 2024_05_31 on dSQBC'''
    def __init__(self,*,root='dSQBC4'):
        self.data = loadmat(os.path.join(root,'24_05_31_03_dSQBC_Matrix_PowerCy_1_KM_MA.mat'))['Matrix_twoD']
        self.tau = loadmat(os.path.join(root,'tau_axis310503.mat'))['tau_axis'][0,:]
        self.wt = loadmat(os.path.join(root,'wt_axis_310503.mat'))['wt_axis'][:,0]
        #I_steps_310503.txt is a csv text file with the pulse power values. Read it in and store values in self.Is
        with open(os.path.join(root,'I_steps_310503.txt'),'r') as f:
            self.Is = np.array([float(x) for x in f.read().split(',')])
        # self.Is = np.array([6,5,4,3,1.5])
        self.name = 'dSQBC20240531'

class loadDSQBC_50kHz:
    '''Data from 2024_06_12 on dSQBC at 50 kHz'''
    root = '/Users/jacob/Dropbox/Research/NLO/raw_data_dSQBC_12_06_24-08/'
    def __init__(self, root = root):        
        self.data = loadmat(root + '24_06_12_08_dSQBC_Matrix_PowerCy_1_KM_RZ_SB.mat')['Matrix_twoD']
        self.tau = loadmat( root + 'tau_axis.mat')['tau_axis'][0,:]
        self.wt = loadmat(  root + 'wt_axis.mat')['wt_axis'][:,0]
        #I_steps_310503.txt is a csv text file with the pulse power values. Read it in and store values in self.Is
        with open(root + 'Powers_12_06_24-08.txt','r') as f:
            self.Is = np.array([float(x) for x in f.read().split(',')])
        # self.Is = np.array([6,5,4,3,1.5])
        self.name = 'dSQBC_50kHz'

class load_polymer_50kHz:
    '''Data from 2024_06_12 on polymer at 50 kHz'''
    root = '/Users/jacob/Dropbox/Research/NLO/raw_data_P18_11_06_24-05/'
    def __init__(self, root=root):        
        self.data = loadmat(root + '24_06_11_05_P18_Matrix_PowerCy_1_KM_MA.mat')['Matrix_twoD']
        self.tau = loadmat( root + 'tau_axis.mat')['tau_axis'][0,:]
        self.wt = loadmat(  root + 'wt_axis.mat')['wt_axis'][:,0]
        #I_steps_310503.txt is a csv text file with the pulse power values. Read it in and store values in self.Is
        with open(root + 'power_11_06_24-05_P18_in_nJ.txt','r') as f:
            self.Is = np.array([float(x) for x in f.read().split(',')])
        # self.Is = np.array([6,5,4,3,1.5])
        self.name = 'polymer_50kHz'

class loadDSQBC_50kHzNoise:
    '''Data from 2024_06_12 on dSQBC at 50 kHz'''
    root='dSQBC_50kHzNoise'
    def __init__(self,root=root):
        self.data = loadmat(os.path.join(root,'24_06_17_06_2DnoSample_Matrix_tau_x_1_x_wt.mat'))['Matrix_twoD']
        self.tau = loadmat(os.path.join(root,'tau_axis.mat'))['tau_axis'][0,:]
        self.wt = loadmat(os.path.join(root,'wt_axis.mat'))['wt_axis'][:,0]
        self.Is = np.array([np.nan])
        self.name = 'dSQBC_50kHzNoise'

class loadDSQBC_50kHz_Jul03:
    '''Data from 2024_07_03 on dSQBC at 50 kHz'''
    root = '/Users/jacob/Dropbox/Research/NLO/IntensityDependent2DandTA/HO2D_data_dSQBC_03July_14/'
    def __init__(self,root=root):
        print(f"root={root}")
        self.data = loadmat(os.path.join(root,'24_07_03_04_SQBC_Matrix_PowerCy_1_KM_LB_SB_data.mat'))['Matrix_twoD']
        self.tau = loadmat(os.path.join(root,'tau_axis.mat'))['tau_axis'][0,:]
        self.wt = loadmat(os.path.join(root,'wt_axis.mat'))['wt_axis'][:,0]
        with open(root + 'intensities_in_nJ.txt','r') as f:
            self.Is = np.array([float(x) for x in f.read().split(',')])
        self.name = 'dSQBC_50kHz_Jul03'
        try:
            self.T_info = loadmat(os.path.join(root,'T_AxisPowerCy.mat'))
            print(self.T)
        except FileNotFoundError:
            print('T_AxisPowerCy.mat not found')

class load_P18_50kHz_Jul03:
    '''Data from 2024_07_03 on P18 at 50 kHz'''
    root = '/Users/jacob/Dropbox/Research/NLO/IntensityDependent2DandTA/HO2D_data_dSQBC_03July_14/'
    def __init__(self,root=root):
        print(f"root={root}")
        self.data = loadmat(os.path.join(root,'24_07_04_04_P18_Matrix_PowerCy_1_KM_LB_SB_data3.mat'))['Matrix_twoD']
        self.tau = loadmat(os.path.join(root,'tau_axis.mat'))['tau_axis'][0,:]
        self.wt = loadmat(os.path.join(root,'wt_axis.mat'))['wt_axis'][:,0]
        with open(root + 'intensities_in_nJ.txt','r') as f:
            self.Is = np.array([float(x) for x in f.read().split(',')])
        self.name = 'P18_50kHz_Jul03'
        self.T_info = loadmat(os.path.join(root,'PowerCy.mat'))
        print(self.T_info)

class load_P18_paper_data:
    root = 'paper_data'
    def __init__(self,root=root):
        data_arch = loadmat(os.path.join(root,'24_07_04_04_P18_Matrix_PowerCy_dataas3_wo35.mat'))
        self.data = data_arch['Matrix_twoD']
        tau_arch = loadmat(os.path.join(root,'tau_axis.mat'))
        self.tau = tau_arch['tau_axis'][0,:]
        wt_arch  = loadmat(os.path.join(root,'wt.mat'))
        self.wt = wt_arch['wt'][0,:]/6.582119E-1
        self.name = 'paper_data'
        self.Is = np.array([60.04,55.1,39.9,21.84,4.978,0.627])
        

loader_dictionary = {'Stage':loadStage,'Shaper':loadShaper,'dSQBC1':loadDSQBC1,
                     'dSQBC2':loadDSQBC2,'dSQBC3':loadDSQBC3,'dSQBC123':loadDSQBC123,
                     'dSQBC4':loadDSQBC4, 'dSQBC_50kHz':loadDSQBC_50kHz,
                     'polymer_50kHz':load_polymer_50kHz,
                     'dSQBC_50kHzNoise':loadDSQBC_50kHzNoise,
                     'dSQBC_50kHz_Jul03':loadDSQBC_50kHz_Jul03,
                     'P18_50kHz_Jul03':load_P18_50kHz_Jul03,
                     'P18_paper_data':load_P18_paper_data}

class visualize:
    hbar =  6.582119E-1
    region_1Q = [[1.4,1.9],[1.5,1.8]]
    region_2Q = [[2.8,3.9],[1.5,1.8]]

    def __init__(self,folder,*,root='auto',tau_reflect = False):
        """Load data and perform DFT wrt tau
        Args:
            folder : specifies which data to load, must be a key from
                loader_dictionary.
        Kwargs:
            root : specifies the full or relative directory location of the data to load
                if set to 'auto', argument folder is taken to also be the relative
                directory location
        """
        
        if root == 'auto':
            root = folder
        self.load_data(folder,root=root)
        self.reshape_data()
        if tau_reflect:
            self.ft_tau_reflect()
        else:
            self.ft()
        self.full_data = self.data.copy()
        self.full_data_ft = self.data_ft.copy()
        self.full_wt = self.wt.copy()
        self.full_tau = self.tau.copy()
        self.savedir = os.path.join(root,'figures')

    def load_data(self,folder,*,root=''):
        loader = loader_dictionary[folder](root=root)
        self.data = loader.data
        self.tau = loader.tau
        self.wt = loader.wt
        self.all_Is = loader.Is / 4
        self.name = loader.name

    def bin_omega_t(self,n):
        sh = self.full_data.shape
        sh = (sh[0],sh[1]//n,sh[2])
        sh_ft = self.full_data_ft.shape
        sh_ft = (sh_ft[0],sh_ft[1]//n,sh_ft[2])
        new_data_ft = np.zeros(sh_ft)
        new_data = np.zeros(sh)
        new_wt = np.zeros(sh[1])
        for i in range(new_data_ft.shape[1]):
            slc = slice(i*n,(i+1)*n)
            new_data_ft[:,i,:] = np.average(self.full_data_ft[:,slc,:],axis=1)
            new_data[:,i,:] = np.average(self.full_data[:,slc,:],axis=1)
            new_wt[i] = np.average(self.full_wt[slc])

        self.data = new_data
        self.data_ft = new_data_ft
        self.wt = new_wt

    def reshape_data(self):
        self.data = np.moveaxis(self.data,1,2)
        self.data = self.data[:,:,::-1]
        self.all_Is = self.all_Is[::-1]

    def truncate_tau_data(self,num):
        self.data = self.full_data[:num,:,:]
        self.tau = self.full_tau[:num]

    def ft(self,*,factor2=True,remove_ave = True,remove_offset = False, 
           window_alpha = 0, z_ratio = 2, end_pad = False):
        if factor2:
            data = self.data.copy()
            if remove_ave:
                av = np.average(data,axis=0)
                data -= av[np.newaxis,...]
            win = tukey(self.tau.size*2,window_alpha)[self.tau.size:]
            data *= win[:,np.newaxis,np.newaxis]
            data[0,...] *= 0.5
            if remove_ave:
                # Add back in the 0Q signal (ie, not tau dependent) but multiply it by 0.5, which makes self-consistency checks work correctly, 
                # since half of the 0Q is associated with the -nQ signals.
                data += av[np.newaxis,...]/2
        else:
            data = self.data
        dtau = self.tau[1] - self.tau[0]
        # new_tau = np.arange(0,self.tau[-1]*z_ratio+dtau/2,dtau)
        new_tau = np.arange(0,self.tau.size*z_ratio) * dtau
        new_shape = list(data.shape)
        new_shape[0] = new_tau.size
        new_data = np.zeros(new_shape)
        new_data[:self.tau.size,...] = data
        if remove_ave and not end_pad:
            print('ave-padding')
            new_data[self.tau.size:,...] += av[np.newaxis,...]/2
        else:
            print('end-padding')
            new_data[self.tau.size:,...] += data[-1,...]
        print(f"tau.shape: {new_tau.shape}, data.shape: {new_data.shape}")
        self.wtau, data_ft = ufss.signals.SignalProcessing.ft1D(new_tau,new_data,axis=0)
        self.data_ft = np.real(data_ft) / (2*np.pi) * (self.wtau[1]-self.wtau[0])
        if remove_offset:
            self.remove_offset()
        else:
            self.set_ave_noise_vs_wt()

    def ft_tau_reflect(self,*,remove_offset=False):
        new_tau_size = self.tau.size * 2 - 1
        new_tau = np.zeros(new_tau_size)
        new_tau[:new_tau_size//2 + 1] = -self.tau[::-1]
        new_tau[new_tau_size//2:] = self.tau
        new_data = np.zeros((new_tau_size,self.data.shape[1],self.data.shape[2]))
        new_data[:new_tau_size//2 + 1,:,:] = self.data[::-1,:,:]
        new_data[new_tau_size//2:,:,:] = self.data
        print(f"tau.shape: {new_tau.shape}, data.shape: {new_data.shape}")
        self.wtau, data_ft = ufss.signals.SignalProcessing.ft1D(new_tau,new_data,axis=0)
        self.data_ft = np.real(data_ft) / (2*np.pi) * (self.wtau[1]-self.wtau[0])
        if remove_offset:
            self.remove_offset()
        else:
            self.set_ave_noise_vs_wt()

    def plot_data(self,I_index,wtau_start = 1):
        x = self.wtau * self.hbar
        x_inds = np.where(x>wtau_start)[0]
        y = self.wt * self.hbar
        # print(x[x_inds].shape,y.shape)
        ufss.signals.plot2D(x[x_inds],y,self.data_ft[x_inds,:,I_index],part='real')
        plt.title(r'I = {:.0f} $\mu$W'.format(self.all_Is[I_index]))

    def savefig(self,filename):
        #Create the directories required
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)            
        plt.savefig(os.path.join(self.savedir,filename))

    def plot_offset_removal(self,I_index):
        self.ft(remove_ave = False, remove_offset=False)
        self.plot_1Q(I_index,region=[[0.5,4.5],[1.5,1.8]])
        plt.axvline(x=0.6,color='k')
        plt.axvline(x=1,color='k')
        plt.axvline(x=2.3,color='orange')
        plt.axvline(x=2.8,color='orange')
        plt.axvline(x=3.8,color='green')
        plt.axvline(x=4.4,color='green')
        self.savefig('Before_offset_removal')
        self.remove_offset()
        Z = np.ones((self.wtau.size,self.wt.size)) *self.DC_vs_wt[np.newaxis,:,I_index]
        x = self.wtau*self.hbar
        y = self.wt*self.hbar
        ufss.signals.plot2D(x,y,Z,part='real')#,vmax=np.max(np.abs(z)))
        plt.xlim([0.5,4.5])
        plt.ylim([1.5,1.8])
        self.savefig('DC_removed')
        self.plot_1Q(I_index,region=[[0.5,4.5],[1.5,1.8]])
        self.savefig('After_offset_removal')

    def plot_offset_removal_wt_slice(self,I_index,wt):
        self.ft(remove_offset=False)
        fig, ax = plt.subplots()
        self.plot_wt_slice(I_index,wt,fig=fig,ax=ax)
        self.ft(remove_offset=True)
        self.plot_wt_slice(I_index,wt,fig=fig,ax=ax)
        fig.legend(['Before removal','After removal'])
        self.savefig('Remove_offset_wt_slice.png')

    def remove_offset(self):
        region1 = [[0.6,1],[-np.inf,np.inf]]
        region2 = [[2.3,2.8],[-np.inf,np.inf]]
        region3 = [[3.8,4.4],[-np.inf,np.inf]]
        x1,y1,z1 = self.get_region(region1)
        x2,y2,z2 = self.get_region(region2)
        x3,y3,z3 = self.get_region(region3)        
        print(f"remove_offset: region1 shape {z1.shape},region 2 shape {z2.shape}, region 3 shape {z3.shape}")
        z = np.concatenate((z1,z2,z3),axis=0)
        self.DC_vs_wt = np.average(z,axis=0)
        self.data_ft = self.data_ft - self.DC_vs_wt[np.newaxis,:,:]
        z_dev = z - self.DC_vs_wt[np.newaxis,:,:]
        self.ave_noise_vs_wt = np.sqrt(np.average(np.abs(z_dev)**2,axis=0))

    def set_ave_noise_vs_wt(self):
        region1 = [[0.6,0.9],[-np.inf,np.inf]]
        region2 = [[2.3,2.8],[-np.inf,np.inf]]
        region3 = [[3.8,4.2],[-np.inf,np.inf]]
        x1,y1,z1 = self.get_region(region1)
        x2,y2,z2 = self.get_region(region2)
        x3,y3,z3 = self.get_region(region3)        
        # z = np.concatenate((z1,z2,z3),axis=0)
        z = z1
        DC_vs_wt = np.average(z,axis=0)
        z_dev = z - DC_vs_wt[np.newaxis,:,:]
        self.ave_noise_vs_wt = np.sqrt(np.average(np.abs(z_dev)**2,axis=0))

    def plot_noise_vs_I(self,wt):
        wt_ind = np.argmin(np.abs(self.wt * self.hbar-wt))
        wt = self.wt[wt_ind] * self.hbar
        noise = self.ave_noise_vs_wt[wt_ind,:]
        plt.figure()
        plt.plot(self.all_Is,noise)
        plt.xlabel(r'Pulse Power ($\mu$W)')
        plt.ylabel('Noise')
        plt.title(r'$\hbar\omega_t =$ {:.2f}'.format(wt))

    def plot_noise_2D(self):
        x = self.wt
        y = self.all_Is
        X,Y = np.meshgrid(x,y,indexing='xy')
        plt.figure()
        plt.pcolormesh(X,Y,self.ave_noise_vs_wt)
        plt.xlabel(r'$\omega_t$')
        plt.ylabel(r'Pulse Power ($\mu$W)')

    def plot_wt_slice(self,I_index,wt,ft=True,fig=None,ax=None,
                      factor2 = False,remove_mean=False,window_alpha = 0):
        wt_ind = np.argmin(np.abs(self.wt * self.hbar-wt))
        wt = self.wt[wt_ind] * self.hbar
        if ft:
            y = self.data_ft[:,wt_ind,I_index]
            x = self.wtau * self.hbar
            xlabel=r'$\hbar\omega_\tau$ (eV)'
        else:
            y = self.data[:,wt_ind,I_index].copy()
            if factor2:
                if remove_mean:
                    av = np.average(y)
                    y -= av
                win = tukey(self.tau.size*2,window_alpha)[self.tau.size:]
                y *= win
                y[0] *= 0.5
                if remove_mean:
                    y += av
            x = self.tau
            xlabel = r'$\tau$ (fs)'
        print('max value',np.max(np.abs(y)))
        if fig == None:
            fig, ax = plt.subplots()
        ax.plot(x,y,'-o')
        ax.set_xlabel(xlabel)
        ax.set_title(r'$\hbar\omega_t =$ {0:.2f}, I = {1:.1f} $\mu$W'.format(wt, self.all_Is[I_index]))

    def get_region(self,region):
        return self.get_region_function(self.data_ft,region)
    
    def get_region_function(self,z,region):
        [xa,xb],[ya,yb] = region

        x = self.wtau * self.hbar
        x_inds = np.where((x>xa) & (x<xb))[0]
        # print(x_inds)
        y = self.wt * self.hbar
        y_inds = np.where((y>ya) & (y<yb))[0]
        z = z[x_inds,:,:]
        z = z[:,y_inds,:]
        return x[x_inds],y[y_inds],z
    
    def get_noise_for_region(self,region):
        [xa,xb],[ya,yb] = region

        x = self.wtau * self.hbar
        x_inds = np.where((x>xa) & (x<xb))[0]
        y = self.wt * self.hbar
        y_inds = np.where((y>ya) & (y<yb))[0]

        noise = self.ave_noise_vs_wt[y_inds,:]
        noise = np.average(noise,axis=0)
        noise = noise*np.sqrt(x[1]-x[0])*(y[1]-y[0]) * np.sqrt(x_inds.size*y_inds.size)
        return noise
    
    def get_integrated_region(self,region):
        x,y,z = self.get_region(region)
        z = np.trapz(z,x=x,axis=0)
        z = np.trapz(z,x=y,axis=0)
        return z
    
    def plot_region(self,region,*,color='k'):
        [xa,xb],[ya,yb] = region
        plt.plot([xa,xa],[ya,yb],color)
        plt.plot([xb,xb],[ya,yb],color)
        plt.plot([xa,xb],[ya,ya],color)
        plt.plot([xa,xb],[yb,yb],color)

    def plot_1Q(self,I_index,*,region='default',ax=None,fig=None):
        '''
        Despite its name, this method is designed to plot the 2D data at specified I_index.
        If region is unspecified, it will plot the self.region_1Q region.
        '''
        if region == 'default':
            region = self.region_1Q
        x,y,z = self.get_region(region)
        ufss.signals.plot2D(x,y,z[...,I_index],part='real',ax=ax,fig=fig)
        if fig is None:
            fig = plt.gcf()
        if ax is None:
            ax=plt.gca()
        ax.set_title(r'I = {:.0f} $\mu$W'.format(self.all_Is[I_index]))
        ax.set_xlabel(r'$\omega_\tau$ (eV)')
        ax.set_ylabel('Detection Frequency (eV)')
        fig.tight_layout()

    def plot_2Q(self,I_index,*,region='default',vmax='max'):
        if region == 'default':
            region = self.region_2Q
        x,y,z = self.get_region(region)
        ufss.signals.plot2D(x,y,z[...,I_index],part='real',vmax=vmax)
        plt.title(r'I = {:.0f} $\mu$W'.format(self.all_Is[I_index]))

    def sqrt_model(self,x,x0,a):
        return a*np.sqrt(x/x0)

    def sat_model(self,x,x0,a):
        return a*(-1+np.exp(-x/x0))
    
    def quad_sat_model(self,x,x0,a):
        return a*(1-(1+x/x0)*np.exp(-x/x0))
    
    def quad2_sat_model(self,x,x0,a):
        xp = x/x0
        return a*(xp**2-6+2*np.exp(-xp)*(xp**2+3*xp+3))/(2*xp**2)
    
    def get_nQ_bessel_model(self,n):
        def nQ_bessel_model(x,x0,a):
            xp = x/x0
            if n == 0:
                f = (ive(0,xp/2) - 1)
            else:
                f = ive(n,xp/2)
            return a*(-1)**n*f
        return nQ_bessel_model
    
    def fit_TA_sat_curve(self,*,wt='integrated'):
        z = self.data[0,...]
        if wt == 'integrated':
            z = np.trapz(z,x=self.wt,axis=0)
        else:
            wt_ind = np.argmin(np.abs(self.wt*self.hbar - wt))
            z = z[wt_ind,:]
        popt, pcov = self.sat_curve(z)
        print(popt)
        print(pcov)
        self.sat_I0, self.sat_a = popt
        plt.figure()
        plt.plot(self.all_Is,z,'o')
        plt.plot(self.all_Is, self.sat_model(self.all_Is,self.sat_I0,self.sat_a),'--k')
        plt.ylabel('TA signal')
        plt.xlabel(r'Pump Pulse Power ($\mu$W)')
        plt.title(r'$I_0$ = {:.1e}, a = {:.1e}'.format(*popt))
        plt.tight_layout()

    def plot_1Q_vs_I(self,*,region='default',integrated=True,
                     include_fits = True):
        if region == 'default':
            region = self.region_1Q
        if integrated:
            x,y,z = self.get_region(region)
            z = np.trapz(z,x=x,axis=0)
            z = np.trapz(z,x=y,axis=0)
            n = self.get_noise_for_region(region)
        else:
            x,y,z = self.get_peak(region)
            wt_ind = np.argmin(np.abs(self.wt*self.hbar-y))
            n = self.ave_noise_vs_wt[wt_ind,:]
        plt.figure()
        plt.plot(self.all_Is,z,'oC1')
        popt, pcov = self.sat_curve(z)
        x0,a = popt
        # popt2, pcov2 = self.bessel_sat_curve(1,z)
        # x02,a2 = popt2
        # popt, pcov = self.sqrt_curve(z)
        # x0b,b = popt
        if include_fits:
            plt.text(self.all_Is[-1]/3,z[-1]/3,'y={1:.1e}(1-exp(-x/{0:.1e}))'.format(x0,a))
            plt.plot(self.all_Is, self.sat_model(self.all_Is,x0,a),'--k')
            plt.plot(self.all_Is, self.get_nQ_bessel_model(1)(self.all_Is,self.sat_I0,self.sat_a),'--r')
            plt.plot(self.all_Is,n,'C2')
            plt.legend(['Data','Sat fit','Bessel (TA fit)','Noise floor'])
        else:
            plt.plot(self.all_Is,n,'C2')
            plt.legend(['Data','Noise floor'])
        plt.xlabel(r'Pulse Power ($\mu$W)')
        if not integrated:
            plt.title('Peak at ({:.2f},{:.2f})'.format(x,y))
        else:
            plt.title('Integrated')

    # def plot_2Q_vs_I(self,*,region='default'):
    #     if region == 'default':
    #         region = self.region_2Q
    #     x,y,z = self.get_region(region)
    #     z = np.trapz(z,x=x,axis=0)
    #     z = np.trapz(z,x=y,axis=0)
    #     slope = (z[-1] - z[0])/(self.all_Is[-1]-self.all_Is[0])
    #     plt.figure()
    #     plt.plot(self.all_Is,z,'.')
    #     plt.plot(self.all_Is,self.all_Is*slope,'--k')
    #     plt.title('Integrated 2Q region')
    #     plt.xlabel('Pulse Power ($\mu$W)')

    def get_peak(self,region):
        '''
        Find the largest amplitude feature in the region, 
        using the last intensity (usually the highest).
        Return the x,y,z values at the peak, where z has length of all the intensities.'''
        x,y,z = self.get_region(region)
        ind = np.unravel_index(np.argmax(np.abs(z[:,:,-1])),z.shape[:-1])
        z = z[ind[0],ind[1],:]
        return x[ind[0]],y[ind[1]],z

    def plot_1Q_manual_peak_vs_I(self,wtau,wt,fig=None,ax=None):
        x = self.wtau*self.hbar
        wtau_ind = np.argmin(np.abs(x-wtau))
        wtau = x[wtau_ind]
        y = self.wt*self.hbar
        wt_ind = np.argmin(np.abs(y-wt))
        wt = y[wt_ind]
        z = self.data_ft[wtau_ind,wt_ind,:]

        if fig == None:
            fig,ax = plt.subplots()
        ax.plot(self.all_Is,z)
        popt, pcov = self.sat_curve(z)
        x0,a = popt
        # popt, pcov = self.sqrt_curve(z)
        # x0b = popt
        ax.plot(self.all_Is, self.sat_model(self.all_Is,x0,a),'--k')
        # ax.plot(self.all_Is, self.sqrt_model(self.all_Is,x0b),'or')
        ax.legend(['Data','Sat fit','Sqrt fit'])
        ax.title('Peak at ({:.2f},{:.2f})'.format(wtau,wt))

    def plot_2Q_manual_peak_vs_I(self,wtau,wt,fig=None,ax=None):
        x = self.wtau*self.hbar
        wtau_ind = np.argmin(np.abs(x-wtau))
        wtau = x[wtau_ind]
        y = self.wt*self.hbar
        wt_ind = np.argmin(np.abs(y-wt))
        wt = y[wt_ind]
        z = self.data_ft[wtau_ind,wt_ind,:]

        slope = (z[-1] - z[0])/(self.all_Is[-1]-self.all_Is[0])
        if fig == None:
            fig,ax = plt.subplots()
        ax.plot(self.all_Is,z,'.')
        # ax.plot(self.all_Is,self.all_Is*slope,'--k')
        # ax.legend(['Data','Linear model'])
        # ax.text(self.all_Is[-1]/2,z[-1]/4,'y={:.1e} x'.format(slope))
        # ax.set_title('Peak at ({:.2f},{:.2f})'.format(wtau,wt))

    def plot_2Q_vs_I(self,*,region='default',integrated=True):
        if region == 'default':
            region = self.region_2Q
        if integrated:
            x,y,z = self.get_region(region)
            z = np.trapz(z,x=x,axis=0)
            z = np.trapz(z,x=y,axis=0)
            n = self.get_noise_for_region(region)
        else:
            x,y,z = self.get_peak(region)
            wt_ind = np.argmin(np.abs(self.wt*self.hbar-y))
            n = self.ave_noise_vs_wt[wt_ind,:]
        print(z.shape)
        slope = (z[-1] - z[0])/(self.all_Is[-1]-self.all_Is[0])
        plt.figure()
        plt.plot(self.all_Is,z,'o')
        plt.plot(self.all_Is,self.all_Is*slope,'--k')
        popt, pcov = self.quad_sat_curve(z)
        popt2, pcov2 = self.quad2_sat_curve(z)
        print(popt,pcov)
        print(popt2,pcov2)
        x0,a = popt
        x02, a2 = popt2
        plt.text(self.all_Is[-1]/3,z[-1]/8,'y={1:.1e}(1-(1+x/{0:.1e})exp(-x/{0:.1e}))'.format(x0,a))
        plt.plot(self.all_Is, self.quad_sat_model(self.all_Is,x0,a),'--g')
        plt.plot(self.all_Is, self.get_nQ_bessel_model(2)(self.all_Is,self.sat_I0,self.sat_a),'--r')
        #plt.plot(self.all_Is, self.quad2_sat_model(self.all_Is,x02,a2),'--g')
        plt.plot(self.all_Is,n)
        plt.legend(['Data','Linear model','Quad sat','Bessel (TA fit)','Noise floor'])
        plt.text(self.all_Is[-1]/4,z[-1]/2,'y={:.1e} x'.format(slope))
        if not integrated:
            plt.title('Peak at ({:.2f},{:.2f})'.format(x,y))
        else:
            plt.title('Integrated 2Q region')

    def sat_curve(self,z):
        f = self.sat_model
        popt, pcov = curve_fit(f,self.all_Is,z,[self.all_Is[-1],z[-1]])
        return popt, pcov
    
    def quad_sat_curve(self,z):
        f = self.quad_sat_model
        popt, pcov = curve_fit(f,self.all_Is,z,[self.all_Is[-1],z[-1]])
        return popt, pcov
    
    def quad2_sat_curve(self,z):
        f = self.quad2_sat_model
        bounds = [[0,0],[self.all_Is[-1]*10,z[-1]*10]]
        print(bounds)
        popt, pcov = curve_fit(f,self.all_Is,z,[self.all_Is[-1],z[-1]],bounds=bounds)
        return popt, pcov
    
    def bessel_sat_curve(self,n,z):
        f = self.get_nQ_bessel_model(n)
        if z[-1] > 0:
            bounds = [[0,0],[self.all_Is[-1]*10,z[-1]*10]]
        else:
            bounds = [[0,z[-1]*10],[self.all_Is[-1]*10,0]]
        #print(bounds)
        popt, pcov = curve_fit(f,self.all_Is,z,[self.all_Is[-1],z[-1]])#,bounds=bounds)
        return popt, pcov
    
    def sqrt_curve(self,z):
        f = self.sqrt_model
        popt, pcov = curve_fit(f,self.all_Is[:2],z[:2],[self.all_Is[-1],z[-1]])
        return popt, pcov
    
    def get_integrated_nQ_vs_I(self):
        y_list = []
        z_list = []
        for region in self.regions:
            x,y,z = self.get_region(region)
            z = np.trapz(z,x=x,axis=0)/(2*np.pi)
            z_list.append(z)
            y_list.append(y)

        return y_list, z_list
    
    def plot_integrated_nQ_signals_vs_I(self,wt):
        y_list, z_list = self.get_integrated_nQ_vs_I()
        y = y_list[0]
        wt_ind = np.argmin(np.abs(y-wt))
        wt = y[wt_ind] * self.hbar
        x = self.all_Is
        num_z = len(z_list)
        fig, axs = plt.subplots(num_z,1,sharex=True)
        for i in range(num_z):
            a = axs[i]
            z = z_list[i][wt_ind,:]            
            popt, pcov = self.bessel_sat_curve(i,z)
            a.plot(x,z,marker='o')
            f = self.get_nQ_bessel_model(i)
            a.plot(x,f(x,*popt))            
            print(popt)
        a.set_xlabel('I')
        fig.suptitle(r'$\omega_t$ = {:.2f}'.format(wt))

    def fit_TA_plot_nQ_vs_I(self,wt):
        self.fit_TA_sat_curve(wt=wt)
        y_list, z_list = self.get_integrated_nQ_vs_I()
        wt = y_list[0]
        wt_ind = np.argmin(np.abs(wt-1.57))
        all_slices = np.array([])
        for i in range(len(z_list)):
            new_slice = z_list[i][wt_ind,:]
            all_slices = np.hstack((all_slices,new_slice))

        Is = self.all_Is
        sz = Is.size

        fig, axes = plt.subplots(5,1,sharex=True,figsize=(5,10))
        for i in range(5):
            axes[i].plot(Is,all_slices[i*sz:(i+1)*sz],'o')
            f = self.get_nQ_bessel_model(i)
            axes[i].plot(Is,f(Is,self.sat_I0,self.sat_a),'--k')
            axes[i].set_ylabel('{}Q'.format(i))
        axes[-1].set_xlabel(r'Pump Pulse Power ($\mu$W)')
        fig.suptitle(r'$I_0$ = {:.1e}, a = {:.1e}'.format(self.sat_I0,self.sat_a))
        fig.tight_layout()

    def fit_all_nQ_vs_I(self,wt):
        y_list, z_list = self.get_integrated_nQ_vs_I()
        wt = y_list[0]
        wt_ind = np.argmin(np.abs(wt-1.57))
        all_slices = np.array([])
        factors = []
        for i in range(len(z_list)):
            new_slice = z_list[i][wt_ind,1:]
            factor = np.max(np.abs(new_slice))
            factors.append(factor)
            all_slices = np.hstack((all_slices,new_slice/factor))

        Is = self.all_Is[1:]
        sz = Is.size

        def fit_fun(x,x0,a):
            ans = np.array([])
            for i in range(5):
                xp = x[i*sz:(i+1)*sz]
                f = self.get_nQ_bessel_model(i)
                y = f(xp,x0,a)/factors[i]
                ans = np.hstack((ans,y))
            return ans
        
        Is_repeated = np.hstack((Is,Is,Is,Is,Is))
        popt, pcov = curve_fit(fit_fun,Is_repeated,all_slices,[1500,0.2])
        print(popt)
        fig, axes = plt.subplots(5,1,sharex=True,figsize=(5,10))
        for i in range(5):
            axes[i].plot(Is,all_slices[i*sz:(i+1)*sz],'o')
            f = self.get_nQ_bessel_model(i)
            axes[i].plot(Is,f(Is,*popt)/factors[i],'--k')
            axes[i].set_ylabel('{}Q'.format(i))
        axes[-1].set_xlabel(r'Pump Pulse Power ($\mu$W)')
        fig.suptitle(r'$I_0$ = {:.1e}, a = {:.1e}'.format(*popt))
        fig.tight_layout()


class modifiedSeparateOrders(SeparateOrders):
    def eval_intensities(self,inds):
        self.Is = self.all_Is[inds]
        self.I0 = np.max(self.all_Is)/4
        self.f_of_Is = self.data_ft[...,inds]
        self.f_with_noise = self.f_of_Is.copy()
        low_bound = np.argmin(np.abs(self.wt*self.hbar - 1.5))
        high_bound = np.argmin(np.abs(self.wt*self.hbar - 1.8))
        self.noise_floor = np.average(self.ave_noise_vs_wt[low_bound:high_bound])

    def eval_intensities_integrated_region(self,inds,region):
        self.Is = self.all_Is[inds]
        self.I0 = np.max(self.Is)/4
        x,y,z = self.get_region(region)
        z = np.trapz(z,x=x,axis=0)
        z = np.trapz(z,x=y,axis=0)
        self.f_of_Is = z[inds]
        self.noise_floor = np.average(self.get_noise_for_region(region))
        self.f_with_noise = self.f_of_Is.copy()

    def plot_order_with_integration_regions(self,order_index):
        a = self.regions[0][0][0]
        b = self.regions[-1][0][-1]
        wt_region = self.regions[0][-1]
        region = [[a,b],wt_region]
        x,y,z = self.get_region_function(self.f_orders,region)
        ufss.signals.plot2D(x,y,z[:,:,order_index-1],part='real')
        colors = ['C0','C1','C2','C3','C4']
        for region, color in zip(self.regions,colors):
            self.plot_region(region,color=color)

        plt.xlabel(r'$\omega_\tau$ (eV)')
        plt.ylabel(r'$\omega_t$ (eV)')

    def plot_order_sum(self,I,region = 'default'):
        if region == 'default':
            region = [[1,4],[1.5,1.8]]
        x,y,z = self.get_region_function(self.f_orders,region)
        z_sum = np.zeros(z.shape[:-1])
        for i in range(z.shape[-1]):
            z_sum += z[...,i] * (I/self.I0)**(i+1)
        ufss.signals.plot2D(x,y,z_sum,part='real')

    def plot_order(self,order_index,region = 'default',vmax='max'):
        if region == 'default':
            region = [[1,4],[1.5,1.8]]
        x,y,z = self.get_region_function(self.f_orders,region)
        fig,ax = plt.subplots(figsize=(4,2.2))
        ufss.signals.plot2D(x,y,z[...,order_index],part='real',fig=fig,ax=ax,vmax=vmax)
        ax.text(0.05,0.85,'$S^{('+'{}'.format(str(2*order_index+3)) + ')}$',transform=ax.transAxes)

    def plot_orders(self,region = 'default',vmax='max'):
        if region == 'default':
            region = [[1,4],[1.5,1.8]]
        x,y,z = self.get_region_function(self.f_orders,region)
        height = self.Is.size * 2.2
        fig,axes = plt.subplots(self.Is.size,1,figsize=(4,height))
        for i in range(self.Is.size):
            ufss.signals.plot2D(x,y,z[...,i],part='real',fig=fig,ax=axes[i],vmax=vmax)
            axes[i].text(0.05,0.85,'$S^{('+'{}'.format(str(2*i+3)) + ')}$',transform=axes[i].transAxes)

    def plot_orders_separate_regions(self,region_inds =[1,2],fit=False, plot_data=False,data_region=None,fontsize=12):
        if fit:
            Z = self.fit_params
        else:
            Z = self.f_orders
        N = Z.shape[-1]
        regions = [self.regions[k] for k in region_inds]
        m = len(regions)
        # If plot_data is true, add a column at left showing the raw data
        plot_cols = m+1 if plot_data else m
        offset = 1 if plot_data else 0
        fig, axes = plt.subplots(N,plot_cols,sharex="col",sharey="col",figsize=(m*4.5,2.5*N))
        region_names = ['0Q','1Q','2Q','3Q','4Q']
        region_names = [region_names[k] for k in region_inds]
        for i in range(m):
            axes[0][i+offset].set_title(region_names[i])
            x,y,z = self.get_region_function(Z,regions[i])
            print(x.shape,y.shape,z.shape)
            for j in range(N):
                ufss.signals.plot2D(x,y,z[...,j],part='real',fig=fig,ax=axes[j][i+offset])
                I_0_power = '' if j==0 else str(j+1) 
                axes[j][i+offset].text(0.82,0.8,'$S^{('+'{}'.format(str(2*j+3)) + ')}' 
                                       + 'I_0^{' + I_0_power + '}$',
                                       transform=axes[j][i+offset].transAxes,fontsize=fontsize)
        if plot_data:
            # Add a column of plots showing the raw data
            if data_region is None:
                data_region = self.region_1Q
            for j in range(N):
                self.plot_1Q(j,region=data_region, ax=axes[j][0])
                axes[j][0].set_title(f'I={self.Is[j]}')
        #Add a supertitle to indicate what intensities were used        
        title_text = self.name if hasattr(self,'name') else ''
        title_text += " fit" if fit else " VdM"
        title_text += f' Intensities: {self.Is}'
        fig.suptitle(title_text)
        fig.tight_layout()
        return fig, axes

    def plot_fit_orders(self,region = 'default'):
        if region == 'default':
            region = [[1,4],[1.5,1.8]]
        x,y,z = self.get_region_function(self.fit_params,region)
        N = self.fit_params.shape[-1]
        height = N * 2.2
        fig,axes = plt.subplots(N,1,figsize=(4,height))
        for i in range(N):
            ufss.signals.plot2D(x,y,z[...,i],part='real',fig=fig,ax=axes[i])
            axes[i].text(0.05,0.85,'$S^{('+'{}'.format(str(2*i+3)) + ')}$',transform=axes[i].transAxes)

    def plot_fit_orders_wt_slice(self,wt):
        wt_ind = np.argmin(np.abs(self.wt*self.hbar - wt))
        z = self.fit_params[:,wt_ind,:]
        N = self.fit_params.shape[-1]
        height = N * 2.2
        fig,axes = plt.subplots(N,1,figsize=(4,height))
        for i in range(N):
            axes[i].plot(self.hbar*self.wtau,z[:,i])
            axes[i].text(0.05,0.85,'$S^{('+'{}'.format(str(2*i+3)) + ')}I_0^{}$'.format(i+1),transform=axes[i].transAxes)

    def set_integration_regions(self,regions):
        self.regions = regions

    def compare_nQ_multiples_visual(self,order,*,fit=True, fig=None, ax=None, save=True, fontsize=10,
                                    linestyle='-',legend = True):
        if fit:
            Z = self.fit_params
        else:
            Z = self.f_orders_no_noise
        y_list = []
        z_list = []
        for region in self.regions:
            x,y,z = self.get_region_function(Z,region)
            # z = np.trapz(z,x=x,axis=0)
            z = np.sum(z,axis=0)
            z_list.append(z[:,order-1])
            y_list.append(y)
        if fig is None and ax is None:
            fig = plt.figure()
            ax = plt.gca()
        elif fig is None:
            fig = plt.gcf()
        elif ax is None:
            ax = plt.gca()
        lines = []
        colors = ['C0','C1','C2','C3','C4']
        for i in range(len(self.regions)-1,-1,-1):
            y = y_list[i]
            z = z_list[i]
            factor = binom(2*order,order-i)
            if factor > 0:
                z = z / factor
            # if i == 0:
            #     z = z/2
            ln, = ax.plot(y,z,color=colors[i],linestyle=linestyle)
            lines.append(ln)

        ax.set_xlabel(r'$\omega_t$ (eV)',fontsize=fontsize)
        ax.set_ylabel('$S_{nQ}^{('+'{}'.format(2*order+1)+')}$',fontsize=fontsize)
        if legend:
            ax.legend(lines[::-1],['0Q','1Q','2Q','3Q','4Q'],title='Normalized')
        fig.tight_layout()
        if save:
            self.savefig('Integrated_comparison')

    def L2_norm(self,A,B):
        numer = np.sum(np.abs(A-B)**2)
        denom = np.sum(np.abs(A)**2)
        return np.sqrt(numer/denom)

    def compare_nQ_multiples(self,order,*,fit=True):
        if fit:
            Z = self.fit_params
            low_bound = np.argmin(np.abs(self.wt*self.hbar - 1.5))
            high_bound = np.argmin(np.abs(self.wt*self.hbar - 1.8))
            noise = np.average(self.ave_noise_vs_wt[low_bound:high_bound])
        else:
            Z = self.f_orders
            noise = self.noise_by_order[order-1]
        z_list = []
        y_list = []
        noises = []
        for region in self.regions:
            x,y,z = self.get_region_function(Z,region)
            noises.append(noise * np.sqrt((x[-1]-x[0])*(y[-1]-y[0])))
            z = np.trapz(z,x=x,axis=0)
            z_list.append(z[:,order-1])
            y_list.append(y)

        z_ref = z_list[0]/binom(2*order,order)
        noise_ref = noises[0]/binom(2*order,order)
        deviations = []
        for i in range(1,len(self.regions)):
            z = z_list[i]
            y = y_list[i]
            factor = binom(2*order,order-i)
            noise = noises[i]
            if factor > 0:
                z = z / factor
                noise = noise / factor
                abs_diff = np.sqrt(np.trapz(np.abs(z-z_ref)**2,x=y))
                dev = abs_diff
                # noise = np.sqrt(noise**2 + noise_ref**2)
                # print(abs_diff,noise)
                # if abs_diff < noise:
                #     dev = 0
                # else:
                #     dev = abs_diff - noise
            else:
                background = np.sqrt(np.trapz(np.abs(z)**2,x=y))
                dev = background
                # print(background,noise)
                # if background < noise:
                #     dev = 0
                # else:
                #     dev = background - noise
            deviations.append(dev)
            # print(dev)

        return np.sum(deviations)
    
    def optimize_Vandermonde_intensities(self,n):
        nmax = self.all_Is.size
        intensity_combinations = list(itertools.combinations(np.arange(1,nmax),n))
        deviations = np.zeros(len(intensity_combinations))
        for i in range(deviations.size):
            self.eval_intensities(np.array(intensity_combinations[i]))
            self.invert()
            for k in range(n):
                deviations[i] += self.compare_nQ_multiples(k+1,fit=False)
        sort_inds = deviations.argsort()
        deviations = deviations[sort_inds]
        intensity_combinations_sorted = [intensity_combinations[j] for j in sort_inds]
        plt.figure()
        plt.plot(deviations)
        return intensity_combinations_sorted, deviations

    def plot_low_power_reference_comparison(self,region_number,*,fit=True):
        rn = region_number
        if fit:
            Z = self.fit_params
        else:
            Z = self.f_orders
        reg = self.regions[rn]
        x_ref, y_ref, z_ref = self.get_region(reg)
        z_ref = z_ref[...,0]
        I_ref = self.all_Is[0]
        x_ext, y_ext, z_ext = self.get_region_function(Z,reg)
        if rn == 0:
            z_ext = z_ext[...,rn] * (I_ref/self.I0)
            title = '$S^{(3)}_{0Q}\\frac{I_{ref}}{I_0}$'
        else:
            z_ext = z_ext[...,rn-1] * (I_ref/self.I0)**rn
            title = '$S^{('+str(rn*2+1)+')}_{'+str(rn)+'Q}(\\frac{I_{ref}}{I_0})^'+str(rn)+'$'
        fig, axes = plt.subplots(3,1,figsize=(4,10))
        #fig.suptitle('{}Q comparison of \n extracted to reference'.format(rn))
        ufss.signals.plot2D(x_ref,y_ref,z_ref,part='real',fig=fig,ax=axes[0])
        axes[0].set_title('Low power reference')
        ufss.signals.plot2D(x_ext,y_ext,z_ext,part='real',fig=fig,ax=axes[1])
        axes[1].set_title(title)
        ufss.signals.plot2D(x_ext,y_ext,z_ref-z_ext,part='real',fig=fig,ax=axes[2])
        axes[2].set_title('Difference')
        for i in range(3):
            axes[i].set_ylabel('Detection Frequency (eV)')
            axes[i].set_xlabel(r'$\omega_\tau$')
        plt.tight_layout()
        self.savefig('{}Q_low_power_ref_comparison'.format(rn))

class save_and_load_fits:
    def save_fit_orders(self):
        base_path = os.path.split(self.savedir)[0]
        fit_path = os.path.join(base_path,'fit_params')
        os.makedirs(fit_path,exist_ok=True)
        N = self.fit_params.shape[-1]
        np.savez(os.path.join(fit_path,'N'+str(N)),fit_params = self.fit_params, fit_errors = self.fit_errors)

    def load_fit_orders(self,N):
        base_path = os.path.split(self.savedir)[0]
        fit_path = os.path.join(base_path,'fit_params')
        arch = np.load(os.path.join(fit_path,'N'+str(N) + '.npz'))
        self.fit_params = arch['fit_params']
        self.fit_errors = arch['fit_errors']

class analyze(visualize,modifiedSeparateOrders,save_and_load_fits):
    def __init__(self,folder,*,root='auto',tau_reflect = False):
        super().__init__(folder,root=root,tau_reflect=tau_reflect) 
