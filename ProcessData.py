# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %matplotlib widget
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
import matplotlib.pyplot as plt
from load_data import analyze

# %%
folders = ['Shaper','Stage','dSQBC1','dSQBC2']
folder = folders[3]
ana = analyze(folder)
print(ana.all_Is)

# %%
# set regions for self-consistency checks
wt_range = [1.5,1.8]
reg0Q = [[-0.8,0.8],wt_range]
reg1Q = [[0.8,2.5],wt_range]
reg2Q = [[2.5,4],wt_range]
ana.set_integration_regions([reg0Q,reg1Q,reg2Q])

# %%
# find optimum intensity combinations for given number of intensities
num_intensities = 2
I_combinations, deviations = ana.optimize_Vandermonde_intensities(num_intensities)
plt.ylabel('Deviation of checks from 0')
plt.xlabel('Intensity combination')
print('In order of best to worst \n',I_combinations)

# %%
plt.close('all')

# %%
Intensities = list(I_combinations[0])
print(Intensities)
ana.truncate_tau_data(800)
print(ana.tau[-1])
ana.ft(window_alpha=0)
ana.eval_intensities(Intensities)
print(ana.Is)
ana.I0 = ana.all_Is[-1]/4
ana.invert()
ana.plot_orders_separate_regions(region_inds=[1,2])
ana.savefig('2D_region_orders_Vandermonde')

# %%
ana.plot_order(0,reg2Q,vmax=0.012/4)
ana.savefig('Scaled_3rd_order_2Q')
ana.plot_order(1,reg2Q,vmax=0.012/4**2)
ana.savefig('Scaled_5th_order_2Q')

# %%
#Sum the extracted orders for a given value of I
I = ana.all_Is[-1]
ana.plot_order_sum(I,region=reg2Q)

# %%
# plot the raw data for the given I_index
I_index = -1
#ana.truncate_tau_data(800)
ana.ft(window_alpha=0)
ana.plot_2Q(I_index,region=reg2Q)
ana.savefig('2Q_max_intensity')

# %%
# fit the data
ana.eval_intensities([1,2,3,4])
ana.fit(3)
ana.save_fit_orders()

# %%
# load and plot the fits
ana.load_fit_orders(3)
#ana.fit_params[ana.wtau.size//2,:,:] /= 2
ana.plot_orders_separate_regions(region_inds = [1,2],fit=True)
ana.savefig('2D_region_orders_fit')

# %%
#visualize self-consistency checks
fit_flag = False
if fit_flag:
    title = 'Fit with {} orders'.format(ana.fit_params.shape[-1])
else:
    title_end = ['{:.1f}, '.format(I) for I in ana.Is]
    title = 'Vandermonde, Powers = ' + ''.join(title_end)
    title = title[:-2]

print(ana.compare_nQ_multiples(1,fit=fit_flag))
ana.compare_nQ_multiples_visual(1,fit=fit_flag)
plt.title(title)
plt.tight_layout()
ana.savefig('S3_comparison')
print(ana.compare_nQ_multiples(2,fit=fit_flag))
ana.compare_nQ_multiples_visual(2,fit=fit_flag)
plt.title(title)
plt.tight_layout()
ana.savefig('S5_comparison')
print(ana.compare_nQ_multiples(3,fit=fit_flag))
ana.compare_nQ_multiples_visual(3,fit=fit_flag)
plt.title(title)
plt.tight_layout()
ana.savefig('S7_comparison')
print(ana.compare_nQ_multiples(4,fit=fit_flag))
ana.compare_nQ_multiples_visual(4,fit=fit_flag)
plt.title(title)
plt.tight_layout()
ana.savefig('S9_comparison')
print(ana.compare_nQ_multiples(5,fit=fit_flag))
ana.compare_nQ_multiples_visual(5,fit=fit_flag)
plt.title(title)
plt.tight_layout()
ana.savefig('S11_comparison')

# %%
ana.plot_order_with_integration_regions(1)
ana.savefig('3rd_order_with_integration_regions')
