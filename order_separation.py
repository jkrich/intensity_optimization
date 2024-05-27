import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
    

class SeparateOrders:
    def __init__(self,f):
        self.f = f
        self.I0 = 1

    def set_taylor_series(self,ser):
        self.taylor = ser

    def set_intensities(self,Is):
        self.Is = Is

    def eval_intensities(self):
        self.f_of_Is = self.eval_intensities_manual(self.Is)

    def eval_intensities_manual(self,Is):
        f_list = [self.f(I) for I in Is]
        f_shape = f_list[0].shape
        new_shape = f_shape + (Is.size,)
        f_of_Is = np.zeros(new_shape,dtype='complex')
        for i in range(Is.size):
            f_of_Is[...,i] = f_list[i]
        return f_of_Is

    def add_noise(self,w):
        sh = self.f_of_Is.shape
        sz = self.f_of_Is.size
        rng = np.random.default_rng()
        noise = rng.standard_normal(sz).reshape(sh) * w
        self.f_with_noise = noise + self.f_of_Is
        self.noise_floor = w

    def fit(self,p_order = 1):
        sh = self.f_with_noise[...,0].shape
        sz = self.f_with_noise[...,0].size
        num_Is = len(self.Is)
        def model(x,*params):
            return sum(x**(i+1)*params[i] for i in range(len(params)))
        f_flat_with_noise = self.f_with_noise.reshape((sz,num_Is))
        initial_guess = np.ones(p_order)
        fit_params = np.zeros((sz,p_order))
        fit_errors = np.zeros((sz,p_order))
        for i in range(sz):
            ydata = f_flat_with_noise[i,:]
            fit_res = curve_fit(model,self.Is/self.I0,ydata,p0=initial_guess)
            fit_params[i,:] = fit_res[0]
            perr = np.sqrt(np.diag(fit_res[1]))
            fit_errors[i,:] = perr

        tot_sh = sh+(p_order,)
        self.fit_params = fit_params.reshape(tot_sh)
        self.fit_errors = fit_errors.reshape(tot_sh)

    def chi_squared_reduced(self):
        N = self.f_with_noise.size
        k = self.fit_params.size
        residuals = []
        for n in range(N):
            I = self.Is[n]
            model_res = self.evaluate_model(I)
            diff = self.f_with_noise[n] - model_res
            residuals.append(diff**2)
        return np.sum(residuals)/(N-k)

    def evaluate_model(self,I):
        #num_params = self.fit_params.shape[-1]
        f_sh = self.f_with_noise[...,0].shape
        f_sz = self.f_with_noise[...,0].size
        #tot_sh = f_sh + (num_params,)
        if f_sz == 1:
            ans = self.interpolate_fun(self.fit_params,I)
        else:
            ans = np.zeros((f_sz,I.size),dtype='complex')
            fit_params = self.fit_params.reshape((f_sz,-1))
            for i in range(f_sz):
                ans[i,:] = self.interpolate_fun(fit_params[i,:],I)

            ans = ans.reshape(f_sh + (I.size,))
        return ans

    def invert(self):
        sz = self.Is.size
        V = np.zeros((sz,sz))
        for i in range(sz):
            V[:,i] = (self.Is/self.I0)**(i+1)

        V_inv = np.linalg.inv(V)
        # print(V_inv)
        f_orders = np.tensordot(V_inv,self.f_with_noise,axes=(1,-1))
        self.f_orders = np.moveaxis(f_orders,0,-1)
        
        f_orders_no_noise = np.tensordot(V_inv,self.f_of_Is,axes=(1,-1))
        self.f_orders_no_noise = np.moveaxis(f_orders_no_noise,0,-1)
        # print(self.f_orders)
        # corr = np.array([self.I0**(n+1) for n in range(sz)])
        # self.f_orders /= corr
        self.noise_by_order = np.sqrt(np.sum(V_inv**2,axis=1))*self.noise_floor

    def interpolate_fun(self,ser,I):
        sh = ser.shape
        new_shape = sh[:-1] + (I.size,)
        ans = np.zeros(new_shape,dtype='complex')
        for i in range(sh[-1]):
            if len(new_shape) == 1:
                ans += ser[...,i] * (I/self.I0)**(i+1)
            else:
                ans += np.tensordot(ser[...,i],(I/self.I0)**(i+1),axes=0)

        return ans

    def interpolate(self,I,*,noise=True):
        if noise:
            orders = self.f_orders
        else:
            orders = self.f_orders_no_noise

        ans = self.interpolate_fun(orders,I)

        return ans
    
    def plot(self,indices = None):
        if indices == None:
            indices = slice(None,None,None)
        else:
            indices = indices + (slice(None,None,None),)
        z = self.f_of_Is[indices]
        zn = self.f_with_noise[indices]
        
        xmin = np.min(self.Is)
        xmax = np.max(self.Is)
        x = np.linspace(xmin,xmax,num = 100)#/self.I0
        zfull = self.eval_intensities_manual(x)[indices]
        interp = self.interpolate(x,noise=True)[indices]
        
        fit = self.evaluate_model(x)[indices]
        interp_no_noise = self.interpolate(x,noise=False)[indices]
        plt.figure()
        # plt.plot(self.Is/self.I0,z,'o')
        plt.plot(self.Is,zn,'^')
        plt.plot(x,zfull,'--k')
        plt.plot(x,interp,'--')
        plt.plot(x,interp_no_noise,'--')
        plt.plot(x,fit,'--')

        plt.legend(['Evaluated with noise','True function','Interpolation',
                    'Interp w/o noise','Fit'])

    def plot_taylor(self,indices = None,n = 1):
        if indices == None:
            indices = slice(None,None,None)
        else:
            indices = indices + slice(None,None,None)

        taylor = self.taylor[indices]

        xmin = np.min(self.Is)
        xmax = np.max(self.Is)
        x = np.linspace(xmin,xmax,num = 100)
        plt.figure()
        keys = []
        for i in range(n):
            cut_off = -n + i + 1
            if cut_off == 0:
                ser = taylor
            else:
                ser = taylor[...,:cut_off]
            keys.append(str(ser.shape[-1]*2+1))
            z = self.interpolate_fun(ser,x)
            plt.plot(x,z)
        plt.legend(keys,title='max order')
        

def intensity_cycling(I0,N):
    phis = np.arange(N)*np.pi/(2*N)
    Is = 4*I0*np.cos(phis)**2
    Is.sort()
    return Is

def linear_sampling(I0,N):
    Is = intensity_cycling(I0,N)
    return np.linspace(Is[0],Is[-1],num=N)

# def minimum_random_error_sampling(I0,N):
#     mrn = MinimzeRandomNoise(N)
#     Is = mrn.minimize_average_noise()*I0
#     return Is

# def minimum_contamination_sampling(I0,N):
#     mse = MinimzeSystematicError(N)
#     Is = mse.minimize_average_contamination()*I0
#     return Is

def chebyshev_sampling(I0,N):
    pts = np.polynomial.chebyshev.chebpts1(N)
    pts = pts + 1
    pts *= 1/2
    pts /= pts[-1]
    return pts*4*I0

intensity_dict = {'Intensity Cycling':intensity_cycling,
                 'Linear Sampling':linear_sampling,
                 'Chebyshev':chebyshev_sampling}

class Optimize_I0_and_N(SeparateOrders):
    def __init__(self,f,f_series,noise,*,
                 intensity_sampling='Intensity Cycling'):
        self.f = f
        self.f_series = f_series
        self.noise_floor = noise
        self.intensity_sampling = intensity_sampling
        self.set_number_of_noise_realizations(1)
        self.sys_weight = 1

    def extract_orders(self,I0,N):
        int_fun = intensity_dict[self.intensity_sampling]
        intensities = int_fun(I0,N)
        self.set_intensities(intensities)
        self.I0 = I0
        self.eval_intensities()
        self.add_noise(self.noise_floor)
        self.invert()

        ext_orders = self.f_orders[:self.num]
        return ext_orders

    def extract_orders2(self,intensities):
        I0 = np.mean(intensities)
        self.set_intensities(intensities)
        self.I0 = I0
        self.eval_intensities()
        self.add_noise(self.noise_floor)
        self.invert()

    def set_number_of_noise_realizations(self,num_realizations):
        self.num_realizations = num_realizations

    def set_differences(self,I0,N):
        true_orders = np.array([self.f_series(n,I0)
                                for n in range(1,self.num+1)])
        
        ext_orders = self.extract_orders(I0,N)
        self.abs_diffs = np.abs(ext_orders - true_orders)
        for i in range(self.num_realizations-1):
            ext_orders = self.extract_orders(I0,N)
            diffs = np.abs(ext_orders - true_orders)
            self.abs_diffs += diffs
        self.abs_diffs /= self.num_realizations
        self.rel_diffs = self.abs_diffs/np.abs(true_orders)

    def get_abs_and_rel_error(self,I0,N):
        if (type(I0) is list) or (type(I0) is np.ndarray):
            I0 = I0[0]
        I0 = np.abs(I0)
        self.set_differences(I0,N)
        abs_av = np.sqrt(np.sum(self.abs_diffs**2))
        rel_av = np.sqrt(np.sum(self.rel_diffs**2))

        return abs_av, rel_av

    def get_average_diffs(self,I0,N):
        abs_av, rel_av = self.get_abs_and_rel_error(I0,N)
        
        return (abs_av/100 + rel_av)

    def print_relative_errors(self,I0,N):
        true_orders = np.array([self.f_series(n,I0)
                                for n in range(1,N+1)])
        self.extract_orders(I0,N)
        ext = np.real(self.f_orders)
        # print('Correct orders',np.round(true_orders,5))
        # print('Extracted orders',np.round(ext,5))
        # print('Relative deviation',np.round((ext - true_orders)/true_orders,3))
        
        re = self.noise_by_order/np.abs(true_orders)
        se = np.abs(true_orders - self.f_orders_no_noise)/np.abs(true_orders)
        print('Relative random error',np.round(re,5),
              '\n','Relative systematic error',np.round(se,5))

    def get_random_and_systematic_error_arrays(self,I0,N):
        if (type(I0) is list) or (type(I0) is np.ndarray):
            I0 = I0[0]
        I0 = np.abs(I0)
        true_orders = np.array([self.f_series(n,I0)
                                for n in range(1,self.num+1)])
        # print(true_orders)
        self.extract_orders(I0,N)
        re = np.abs(self.noise_by_order[:self.num]/true_orders)
        se = np.abs((true_orders - self.f_orders_no_noise[:self.num])/true_orders)
        return re, se

    def get_random_and_systematic_errors(self,I0,N):
        re, se = self.get_random_and_systematic_error_arrays(I0,N)
        re_av = np.sqrt(np.sum(re**2))
        se_av = np.sqrt(np.sum(se**2))
        return re_av, se_av
        
    def get_total_error(self,I0,N):
        re_av, se_av = self.get_random_and_systematic_errors(I0,N)
        w = self.sys_weight
        return np.sqrt(re_av**2 + w**2 * se_av**2)

    def minimize_errors_single_N(self,number_of_orders,N):
        self.num = number_of_orders
        if N < self.num:
            raise Exception('Cannot extract more orders than intensities')
        f = lambda x: self.get_total_error(x,N)
        x0 = 1
        res = minimize(f,x0,method='Powell',bounds=[(1E-15,20)])
        I0 = np.abs(res.x)
        r_err, s_err = self.get_random_and_systematic_errors(I0,N)
        rms_err = np.sqrt(s_err**2+r_err**2)
        return I0, r_err, s_err, rms_err

    def minimize_errors(self,number_of_orders,*,max_additional_points = 4):
        num = number_of_orders
        for N in range(num,num+max_additional_points+1):
            I0, r_err, s_err, rms_err = self.minimize_errors_single_N(num,N)
            print('Intensities:',np.round(self.Is,2),', sys err={:.2e}, rand err={:.2e}, RMS err={:.2e}'.format(s_err,r_err,rms_err))
    # 
    #     self.num = number_of_orders
    #     
    #         f = lambda x: self.get_total_error(x,N)
    #         x0 = 1
    #         res = minimize(f,x0,method='Powell',bounds=[(0.1,20)])
    #         I0 = np.abs(res.x)
    #         r_err, s_err = self.get_random_and_systematic_errors(I0,N)
    #         rms_err = np.sqrt(s_err**2+r_err**2)
    #         print('Intensities:',np.round(self.Is,2),', sys err={:.3f}, rand err={:.3f}, RMS err={:.3f}'.format(s_err,r_err,rms_err))
    #         # print('N = {}, I0 = {}, systematic error = {:.4f}, random error = {:.4f}'.format(N,I0,s_err,r_err))

    def minimize_diffs(self,number_of_orders):
        self.num = number_of_orders
        for N in range(self.num,self.num+10):
            f = lambda x: self.get_average_diffs(x,N)
            x0 = 1
            res = minimize(f,x0,method='Nelder-Mead')
            I0 = np.abs(res.x)
            abs_av, rel_av = self.get_abs_and_rel_error(I0,N)
            print('N = {}, I0 = {}, abs err = {}, rel err = {}'.format(N,I0,abs_av,rel_av))

class OptimizeIntensities(SeparateOrders):
    def __init__(self,f,f_series,noise):
        self.f = f
        self.f_series = f_series
        self.noise_floor = noise
        self.set_number_of_noise_realizations(1)
        self.sys_weight = 1

    def extract_orders(self,intensities):
        I0 = np.mean(intensities)
        self.set_intensities(intensities)
        self.I0 = I0
        self.eval_intensities()
        self.add_noise(self.noise_floor)
        self.invert()

    def set_number_of_noise_realizations(self,num_realizations):
        self.num_realizations = num_realizations

    def set_differences(self,intensities):
        ext_orders = self.extract_orders(intensities)
        true_orders = np.array([self.f_series(n,self.I0)
                                for n in range(1,self.num+1)])
        
        self.abs_diffs = np.abs(ext_orders - true_orders)
        for i in range(self.num_realizations-1):
            ext_orders = self.extract_orders(I0,N)
            diffs = np.abs(ext_orders - true_orders)
            self.abs_diffs += diffs
        self.abs_diffs /= self.num_realizations
        self.rel_diffs = self.abs_diffs/np.abs(true_orders)

    def get_abs_and_rel_error(self,intensities):
        intensities = np.abs(intensities)
        self.set_differences(intensities)
        abs_av = np.sqrt(np.sum(self.abs_diffs**2))
        rel_av = np.sqrt(np.sum(self.rel_diffs**2))

        return abs_av, rel_av

    def get_average_diffs(self,intensities):
        abs_av, rel_av = self.get_abs_and_rel_error(intensities)
        
        return (abs_av/100 + rel_av)

    def print_relative_errors(self,intensities):
        ext_orders = self.extract_orders(intensities)
        true_orders = np.array([self.f_series(n,self.I0)
                                for n in range(1,self.num+1)])
        ext = np.real(self.f_orders)
        # print('Correct orders',np.round(true_orders,5))
        # print('Extracted orders',np.round(ext,5))
        # print('Relative deviation',np.round((ext - true_orders)/true_orders,3))
        
        re = self.noise_by_order/np.abs(true_orders)
        se = np.abs(true_orders - self.f_orders_no_noise)/np.abs(true_orders)
        print('Relative random error',np.round(re,5),
              '\n','Relative systematic error',np.round(se,5))

    def get_random_and_systematic_errors(self,intensities):
        intensities = np.abs(intensities)
        self.extract_orders(intensities)
        true_orders = np.array([self.f_series(n,self.I0)
                                for n in range(1,self.num+1)])
        # print(true_orders)
        
        # print(self.noise_by_order)
        # print('f orders',self.f_orders_no_noise[:self.num])
        # print('noise by order',self.noise_by_order[:self.num])
        re = np.abs(self.noise_by_order[:self.num]/true_orders)
        # print('random error',re)
        se = np.abs((true_orders - self.f_orders_no_noise[:self.num])/true_orders)
        # print(se)
        re_av = np.sqrt(np.sum(re**2))
        se_av = np.sqrt(np.sum(se**2))
        return re_av, se_av
        
    def get_total_error(self,intensities):
        re_av, se_av = self.get_random_and_systematic_errors(intensities)
        w = self.sys_weight
        return np.sqrt(re_av**2 + w**2 * se_av**2)

    def minimize_errors_single_N(self,number_of_orders,N):
        self.num = number_of_orders
        if N < self.num:
            raise Exception('Cannot extract more orders than intensities')
        f = self.get_total_error
        x0 = np.arange(1,N+1)/N
        res = minimize(f,x0,method='Nelder-Mead')#,bounds=[(0.01,100)]*N)
        intensities = np.abs(res.x)
        intensities.sort()
        r_err, s_err = self.get_random_and_systematic_errors(intensities )
        rms_err = np.sqrt(s_err**2+r_err**2)
        return intensities, r_err, s_err, rms_err

    def minimize_errors(self,number_of_orders,*,max_additional_points = 4):
        num = number_of_orders
        for N in range(num,num+max_additional_points+1):
            Is, r_err, s_err, rms_err = self.minimize_errors_single_N(num,N)
            print('Intensities:',np.round(Is,2),', sys err={:.2e}, rand err={:.2e}, RMS err={:.2e}'.format(s_err,r_err,rms_err))

    # def minimize_errors(self,number_of_orders):
    #     self.num = number_of_orders
    #     for N in range(self.num,self.num+7):
    #         f = self.get_total_error
    #         x0 = np.arange(1,N+1)/N
    #         res = minimize(f,x0,method='Nelder-Mead')#,bounds=[(0.01,100)]*N)
    #         intensities = np.abs(res.x)
    #         r_err, s_err = self.get_random_and_systematic_errors(intensities)
    #         rms_err = np.sqrt(s_err**2+r_err**2)
    #         print('Intensities:',np.round(intensities,2),', sys err={:.3f}, rand err={:.3f}, RMS err={:.3f}'.format(s_err,r_err,rms_err))

    def minimize_diffs(self,number_of_orders):
        self.num = number_of_orders
        for N in range(self.num,self.num+10):
            f = self.get_average_diffs
            x0 = np.arange(N)/N
            res = minimize(f,x0,method='Nelder-Mead')
            intensities = np.abs(res.x)
            abs_av, rel_av = self.get_abs_and_rel_error(intensities)
            print('Intensities: ',np.round(intensities,3),', abs err = {}, rel err = {}'.format(abs_av,rel_av))
        
    
