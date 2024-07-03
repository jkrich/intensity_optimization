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
        # print(f"I0 ={self.I0:0.2f}")
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

def chebyshev_sampling(I0,N):
    pts = np.polynomial.chebyshev.chebpts1(N)
    pts = pts + 1
    pts *= 1/2
    pts /= pts[-1]
    return pts*4*I0

intensity_dict = {'Intensity Cycling':intensity_cycling,
                 'Linear Sampling':linear_sampling,
                 'Chebyshev':chebyshev_sampling}

class OptimizeIntensities(SeparateOrders):
    def __init__(self,f,f_series,noise):
        self.f = f
        self.f_series = f_series
        self.noise_floor = noise
        self.set_number_of_noise_realizations(1)
        self.sys_weight = 1

    def extract_orders(self,intensities):
        '''Given input set of N intensities, take |intensities| and extract 
        the first N orders using self.invert for the van der Monde method'''
        intensities = np.abs(intensities)
        I0 = np.mean(intensities)/10
        self.set_intensities(intensities)
        self.I0 = I0
        self.eval_intensities()
        self.add_noise(self.noise_floor)
        self.invert()

        ext_orders = self.f_orders[:self.num]
        return ext_orders

    def set_number_of_noise_realizations(self,num_realizations):
        self.num_realizations = num_realizations

    def set_differences(self,intensities):
        ext_orders = self.extract_orders(intensities)
        true_orders = np.array([self.f_series(n,self.I0)
                                for n in range(1,self.num+1)])
        
        self.abs_diffs = np.abs(ext_orders - true_orders)
        for i in range(self.num_realizations-1):
            ext_orders = self.extract_orders(intensities)
            diffs = np.abs(ext_orders - true_orders)
            self.abs_diffs += diffs
        self.abs_diffs /= self.num_realizations
        denominator = true_orders.copy()
        zero_inds = np.where(np.isclose(denominator,0,atol=1E-12))
        denominator[zero_inds] = 1
        self.rel_diffs = self.abs_diffs/np.abs(denominator)

    def get_abs_and_rel_error(self,intensities):
        intensities = np.abs(intensities)
        self.set_differences(intensities)
        abs_av = np.sqrt(np.sum(self.abs_diffs**2))
        rel_av = np.sqrt(np.sum(self.rel_diffs**2))

        return abs_av, rel_av

    def get_average_diffs(self,intensities):
        abs_av, rel_av = self.get_abs_and_rel_error(intensities)
        
        return (abs_av/100 + rel_av)
    
    def get_true_orders(self):
        true_orders = np.zeros(self.f_orders.shape,dtype=self.f_orders.dtype)
        for n in range(1,self.Is.size+1):
            true_orders[...,n-1] = self.f_series(n,self.I0)
        return true_orders
    
    def get_random_and_systematic_error_arrays(self,intensities):
        self.extract_orders(intensities)
        true_orders = self.get_true_orders()[...,:self.num]
        denominator = np.abs(true_orders.copy())**2
        while len(denominator.shape) > 1:
            denominator = np.sum(denominator,axis=0)
        denominator = np.sqrt(denominator)
        zero_inds = np.where(np.isclose(denominator,0,atol=1E-12))
        denominator[zero_inds] = 1
        re = np.abs(self.noise_by_order[:self.num]*np.sqrt(true_orders[...,0].size)/denominator)
        se = np.abs((true_orders - self.f_orders_no_noise[...,:self.num])/denominator)**2
        while len(se.shape) > 1:
            se = np.sum(se,axis=0)
        se = np.sqrt(se)
        return re, se

    def print_relative_errors(self,intensities):
        re, se = self.get_random_and_systematic_error_arrays(intensities)
        print('Relative random error for each order',np.round(re,5),
              '\n','Relative systematic error for each order',np.round(se,5),
              '\n Total relative error for each order',np.round(np.sqrt(re**2+se**2),5))

    def get_random_and_systematic_errors(self,intensities):
        """Get total random and systematic errors, separately, for the orders of interest"""
        re, se = self.get_random_and_systematic_error_arrays(intensities)
        re_av = np.sqrt(np.sum(re**2))
        se_av = np.sqrt(np.sum(se**2))
        return re_av, se_av
        
    def get_total_error(self,intensities):
        """Sum random and systamic errors for all orders of interest, including a weight factor
        that is set to 1 by default in the constructor of the class"""
        re_av, se_av = self.get_random_and_systematic_errors(intensities)
        w = self.sys_weight
        return np.sqrt(re_av**2 + w**2 * se_av**2)

    def minimize_errors_single_N(self,number_of_orders,N,intensity_selection='any',bounds=[[1E-15,2],]):
        """Minimize random and systematic errors for the Vandermonde inversion procedure.
        Args:
            number_of_orders (int) : number of orders for which you wish to minimize the error <= N
            N (int) : number of intensities to use
        Kwargs:
            intensity_selection : 'any' runs a Nelder-Mead optimization over all N intensity values. three
                other options run a 1D optimization over I_0, which determines all N intensity values via
                the three sampling strategies included in intensity_dict, defined above this class. The 
                options are : 'Linear Sampling', 'Intensity Cycling' and 'Chebyshev'.
            bounds : bounds for I_0 for the 1D optimization procedure. This is ignored if intensity_selection
                is 'any'
        Returns: intensities, r_err, s_err, rms_err
            intensities : the N intensities that minimize the total error scaled with I_sat = 1
        """
        self.num = number_of_orders
        if N < self.num:
            raise Exception('Cannot extract more orders than intensities')
        if intensity_selection == 'any':
            f = self.get_total_error
            x0 = np.arange(1,N+1)/N/4
            res = minimize(f,x0,method='Nelder-Mead')
            intensities = np.abs(res.x)
            intensities.sort()
        elif intensity_selection == 'bounded':
            # we map the region [-inf, inf] to the given bounds, 
            # use unconstrained optimization, and then map back
            # Many possibilities. Let's use this one, from https://math.stackexchange.com/questions/75077/mapping-the-real-line-to-the-unit-interval            
            # print(f"bounded: {bounds}")
            def h(x,bounds):
                # (a,b) to (0,1)
                a,b = bounds
                return (x-a)/(b-a)
            def h_inv(y,bounds):
                # (0,1) to (a,b)
                a,b = bounds
                return y*(b-a)+a
            def g(x):
                #(0,1) to (-inf,inf)
                # return (2*x-1)/(x-x**2)
                return np.log((1-x)/x)
            def g_inv(y):
                #(-inf,inf) to (0,1)
                # return ((y-2) + np.sqrt((y-2)**2+4))/(2*y)
                return 1/(1+np.exp(y))
            def bound2R(x,bounds):
                return g(h(x,bounds))
            def R2bound(y,bounds):
                return h_inv(g_inv(y),bounds)
            f = lambda x: self.get_total_error(R2bound(x,bounds))
            x0 = np.arange(1,N+1)/N/4
            x0 = bound2R(x0,bounds)
            res = minimize(f,x0,method='Nelder-Mead')
            # print(res.x)
            # print(R2bound(res.x,bounds))
            # intensities = np.abs(res.x) 
            intensities = R2bound(res.x,bounds)
            intensities.sort()
        else:
            # This code is a generic else, but it really only does intensity cycling ratios
            print('Intensity selection:',intensity_selection)
            def f(I):
                intensities = intensity_dict[intensity_selection](I,N)
                return self.get_total_error(intensities)
            x0 = 1
            res = minimize(f,x0,method='Powell',bounds = bounds)
            intensities = intensity_cycling(res.x,N)
        # print(intensities)
        r_err, s_err = self.get_random_and_systematic_errors(intensities)
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
        
    
