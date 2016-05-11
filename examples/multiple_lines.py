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


def multiple_lines_lnprior(p):
    amp1, xcen1, sigma1, amp2, xcen2, sigma2, gamma2,\
            amp3, xcen3, sigma3, amp4, xcen4, sigma4, lna, lnalpha = p

    if (-50. < lna < 0.):
        return 0.0

    return -np.inf


# Line centers
lcs = [8434.4, 8467.6, 8581.3, 8735.1]

# Load the spectrum
d = np.loadtxt('spec.txt').T

# Select the region around the line
lines = []
for lc in lcs:
    sel = (d[0] > lc - 8) & (d[0] < lc + 8)
    yerr = np.ones_like(d[0][sel]) * 0.01
    lines.append((d[0][sel], d[1][sel], yerr))


pfiles = [profiles.gaussian, profiles.voigt, profiles.gaussian,
          profiles.gaussian]

pparn = np.cumsum([0] +\
        [len(inspect.getargspec(i)[0]) - 1 for i in pfiles])

initial = [0.1, lcs[0], 1.0, # line 1
           0.1, lcs[1], 1.0, 1.0, # line 2 - lorentzian
           0.1, lcs[2], 1.0, # line 3
           0.1, lcs[3], 1.0, # line 4
           -6.1, 0.3 # kernel
          ]

nwalkers = 128
ndim = len(initial)
niter = 500
data = [lines, pfiles, pparn, single_kernel_noisemodel,
        multiple_lines_lnprior]

p0 = np.array([np.array(initial) + 1e-2 * np.random.randn(ndim)
               for i in xrange(nwalkers)])

sampler = emcee.EnsembleSampler(nwalkers, ndim, gpew.lnprob, args=data)
p0, lnp, _ = sampler.run_mcmc(p0, niter)
sampler.reset()

p = p0[np.argmax(lnp)]
p0 = [p + 1e-2 * np.random.randn(ndim) for i in xrange(nwalkers)]
p0, _, _ = sampler.run_mcmc(p0, niter)

samples = sampler.flatchain

# If we wanted to save samples for later
#  gpew.save_samples('samples.npy', samples)

# We need to provied indices of the amplitudes and offsets of 
# individual lines so that the thing knows where to plot what
gpew.plot_lines(lines, pfiles, pparn, single_kernel_noisemodel, samples,
                nwalkers, [0, 3, 7, 10], [1, 4, 8, 11],
                wlwidth=8.1, gpsamples=100)


