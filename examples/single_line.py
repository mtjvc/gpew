#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
import numpy as np

import emcee
import george
from george import kernels

import os
import sys

currentframe = inspect.currentframe()
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(currentframe)))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import profiles
import gpew


def single_kernel_noisemodel(p):
    """
    Simple one squared-exponential kernel noise model.
    """
    return george.GP(p[0] * kernels.ExpSquaredKernel(p[1]))


def single_kernel_lnprior(p):
    amp, xcen, sigma, lna, lnalpha = p

    if (-50. < lna < 0. and amp > 0. and sigma > 0. and xcen > 8685 and
            xcen < 8690):
        return 0.0

    return -np.inf


# Load the spectrum
d = np.loadtxt('spec.txt').T

# Select the region around the line
sel = (d[0] > 8680) & (d[0] < 8696)

# Come up with uncertainties for S/N = 100
yerr = np.ones_like(d[0][sel]) * 0.01

# Store the line in the lines array
lines = [(d[0][sel], d[1][sel], yerr)]

# Define the profile for the line
pfiles = [profiles.gaussian]

# Generate the array that stores how many parameters each profile
# has. There is only one and we are using a Gaussian profile so we
# now we have 3 parameters but this way we don't need to think about it.
pparn = np.cumsum([0] +\
        [len(inspect.getargspec(i)[0]) - 1 for i in pfiles])

# Initial values for the parameters. The first three are for the Gaussian
# profile, the next two for the one kernel GP noise model. The values
# should be close to the optimal (this is important).
initial = [0.28, # profile amplitude
           8687.82, # profile center wavelength
           1.53, # profile sigma
           -6.1, # kernel amplitude
           0.3 # kernel scale-length
           ]

# Sampler initialization
nwalkers = 128
ndim = len(initial)

# 100 is not enough! Make sure the convergence is satisfacory before
# accepting any results!
niter = 500

# Replace with None to get a trivial chi2 like noise model
noisemodel = single_kernel_noisemodel 

data = [lines, pfiles, pparn, noisemodel, single_kernel_lnprior]

# Initial states of the walkers - N-dim Gaussian around the initial values
p0 = np.array([np.array(initial) + 1e-2 * np.random.randn(ndim)
               for i in xrange(nwalkers)])

# Sampler object
sampler = emcee.EnsembleSampler(nwalkers, ndim, gpew.lnprob, args=data)

# Let's run it!
p0, lnp, _ = sampler.run_mcmc(p0, niter)
sampler.reset()

# Let's get the best lnp value, re-initialize it and run it again.
p = p0[np.argmax(lnp)]
p0 = [p + 1e-2 * np.random.randn(ndim) for i in xrange(nwalkers)]
p0, _, _ = sampler.run_mcmc(p0, niter)

# Collect the samples
samples = sampler.flatchain

# Plot stuff:
# error bars: observed line
# red: +-1 sigma of the complete model
# blue: +-1 sigma of the profile model
#  gpew.plot_lines(lines, pfiles, pparn, single_kernel_noisemodel, samples,
gpew.plot_lines(lines, pfiles, pparn, noisemodel, samples,
                nwalkers, wlwidth=8.1, gpsamples=100)

