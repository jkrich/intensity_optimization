# Optimal Intensities for Intensity-Dependent Spectroscopy

This repository is intended to help choose the **optimal set of pump
intensities** for separating orders of nonlinear response by intensity
variation. 
Given a saturation model for your signal and an estimate of your
noise level, it returns the intensities that minimize the total error in the
extracted response orders. Or, given a set of intensities, the errors in extracted orders can be estimated.

The method is described in:

> Krich, Brenneis, Rose, Mayershofer, Büttner, Lüttig, Malý, Brixner,
> "Separating Orders of Response in Transient Absorption
> and Coherent Multidimensional Spectroscopy by Intensity Variation,"
> *J. Phys. Chem. Lett.* **2025**.
> DOI: [10.1021/acs.jpclett.5c01177](https://pubs.acs.org/doi/10.1021/acs.jpclett.5c01177)
> · preprint: [arXiv:2504.13082](https://arxiv.org/abs/2504.13082)


## What's here

| File | Purpose |
|------|---------|
| [`nQ_optimizer.py`](nQ_optimizer.py) | Main module: built-in signal models and `optimize_nQ_intensities()`. |
| [`nQ_optimizer_demo.ipynb`](nQ_optimizer_demo.ipynb) | Worked examples, including reproducing Table 1 of the paper and defining your own model. |
| [`order_separation.py`](order_separation.py) | Lower-level engine (`OptimizeIntensities`) that `nQ_optimizer.py` calls. |

The two files you need are **`nQ_optimizer.py`** and the
**`nQ_optimizer_demo.ipynb`** notebook.

**Note on naming:** The files are named `nQ_optimizer` because the original
application was selecting intensities for *n*-quantum (nQ) coherence signals
in 2D spectroscopy. The same tools apply directly to **transient absorption
(TA)**. Dedicated TA models
are built in (see Example 4 of the demo notebook), so you can use this for TA without any nQ
interpretation.

## Installation

No installation step beyond Python and a few standard scientific packages:

```bash
pip install numpy scipy tabulate
```

Then run the notebook or import from the repository directory.


## What you need


### Random errors
You must characterize the random error $\sigma$ in your signals. The dimensionless quantity `noise_over_S_max` expresses that noise in units of the high-intensity saturation of the TA signal

### Systematic errors 
Systematic errors can be estimated given the saturation form of the TA signal $S_{TA}(I)$, where $I$ is the intensity (or energy) of the pump pulse (for TA). If you are interested in extractions of an nQ signal for 2D spectra, this code is designed to use the TA saturation form to determine the optimal pump pulse intensities for the 2D pump-pair. Note that the output intensities of the pump pair are the intensities of *each* of the pump-pair pulses in a 2D experiment; if you measure the intensity of the overlapped pump pair, that is a factor of 4 larger. This code handles the case discussed in the manuscript, where the pump-pair pulses have identical envelope and vary together.

#### Built-in signal models
The code has built-in models for $S_{TA}(I)$ have exponential of saturable absorption forms; if you are optimizing for an nQ or TA signal, you must choose the correct factory.

| Factory | Signal | Saturation form |
|---------|--------|-----------------|
| `nQ_exponential_saturation_model` | nQ coherence | S(x) = 1 − exp(−x) |
| `nQ_saturable_absorption_model`   | nQ coherence | S(x) = x / (x + 1) |
| `ta_exponential_model`            | transient absorption | S(x) = 1 − exp(−x) |
| `ta_saturable_absorption_model`   | transient absorption | S(x) = x / (x + 1) |

Here x = I / I_sat is the dimensionless intensity. 

You can also supply your own
model: a factory `get_model(n)` returning `(f, f_series)`, where `f(x)` is the
signal and `f_series(m, x)` is the m<sup>th</sup> term of its power-series expansion. The
demo notebook shows a custom-model example.

## Key inputs to set

- **`noise_over_S_max`** — your noise level $\sigma$ divided by the saturated signal
  S_max. This sets the balance between random error (favors high intensities)
  and systematic contamination from unextracted higher orders (favors low
  intensities).
- **`I_sat`** — the saturation intensity in whatever units you want. All reported
  intensities come back in these units.
- **`I_max`** — the largest pump intensity your setup can deliver; the pulse optimization
  is bounded to [0, I_max]. 
- **`Ns`** — how many distinct intensities you want to measure at.

See Eq. 15 of Krich *et al.* (2025) for the error model.

## Questions

Please send any questions to [Jacob Krich](mailto:jkrich@uottawa.ca) or [Peter Rose](mailto:prose@uottawa.ca). We're happy to help.

## License

See [`LICENSE`](LICENSE).

