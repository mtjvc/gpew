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
    amp1, xcen1, sigma1, amp2, xcen2, sigma2, lna, lnalpha = p

    if (-50. < lna < 0. and
            amp1 > 0. and sigma1 > 0. and
            xcen1 > 6518.2 and xcen1 < 6518.5 and
            amp2 > 0. and sigma2 > 0. and
            xcen2 > 6518.5 and xcen2 < 6518.9
            ):
        return 0.0

    return -np.inf


d = np.loadtxt('spec_blend.txt').T

lines = [(d[0][::3], d[1][::3], d[2][::3])]

pfiles = [profiles.two_gaussian_blend]
noisemodel = single_kernel_noisemodel 

pparn = np.cumsum([0] +\
        [len(inspect.getargspec(i)[0]) - 1 for i in pfiles])

initial = [0.34, 6518.37, 0.0055, 0.118, 6518.75, 0.0075, -9.4, -4.7]

nwalkers = 128
ndim = len(initial)
niter = 100

data = [lines, pfiles, pparn, noisemodel, lnprior]

p0 = np.array([np.array(initial) + np.array([0.001, 0.0001, 0.0001, 0.001,
               0.0001, 0.0001, 0.01, 0.01]) * np.random.randn(ndim)
               for i in xrange(nwalkers)])

sampler = emcee.EnsembleSampler(nwalkers, ndim, gpew.lnprob, args=data)
p0, lnp, _ = sampler.run_mcmc(p0, niter)
sampler.reset()

p = p0[np.argmax(lnp)]
p0 = [p + 1e-2 * np.random.randn(ndim) for i in xrange(nwalkers)]
p0, _, _ = sampler.run_mcmc(p0, niter)

samples = sampler.flatchain

gpew.plot_lines(lines, pfiles, pparn, noisemodel, samples,
                nwalkers, wlwidth=2, gpsamples=100, profilepoints=200)

