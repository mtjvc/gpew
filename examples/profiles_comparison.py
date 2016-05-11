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


def lnprior(p):
    amp1, xcen1, sigma1, amp2, xcen2, fwhm2, amp3, xcen3, sigma3, gamma3, lna, lnalpha = p

    if (-50. < lna < 0. and
            xcen1 > 8685 and xcen1 < 8690 and
            xcen2 > 8685 and xcen2 < 8690 and
            xcen3 > 8685 and xcen3 < 8690
            ):
        return 0.0

    return -np.inf


d = np.loadtxt('spec.txt').T
sel = (d[0] > 8680) & (d[0] < 8696)
yerr = np.ones_like(d[0][sel]) * 0.01

# three of the same
lines = [(d[0][sel], d[1][sel], yerr), (d[0][sel], d[1][sel], yerr), (d[0][sel], d[1][sel], yerr)]

pfiles = [profiles.gaussian, profiles.lorentzian, profiles.voigt]

pparn = np.cumsum([0] +\
        [len(inspect.getargspec(i)[0]) - 1 for i in pfiles])

initial = [0.28, 8687.82, 1.53, 0.28, 8687.82, 0.5, 0.28, 8687.82, 0.5, 0.5,
           -6.1, 0.3]

nwalkers = 128
ndim = len(initial)
niter = 500

noisemodel = single_kernel_noisemodel 
data = [lines, pfiles, pparn, noisemodel, lnprior]
p0 = np.array([np.array(initial) + 1e-2 * np.random.randn(ndim)
               for i in xrange(nwalkers)])

sampler = emcee.EnsembleSampler(nwalkers, ndim, gpew.lnprob, args=data)

p0, lnp, _ = sampler.run_mcmc(p0, niter)
sampler.reset()

p = p0[np.argmax(lnp)]
p0 = [p + 1e-2 * np.random.randn(ndim) for i in xrange(nwalkers)]
p0, _, _ = sampler.run_mcmc(p0, niter)

samples = sampler.flatchain

gpew.plot_lines(lines, pfiles, pparn, noisemodel, samples,
                nwalkers, iamp=[0, 3, 6], ixcen=[1, 4, 7], wlwidth=8.1,
                gpsamples=100)

