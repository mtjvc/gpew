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

import matplotlib.pyplot as pl


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


def chi2_lnprior(p):
    amp, xcen, sigma = p

    if (amp > 0. and sigma > 0. and xcen > 8685 and xcen < 8690):
        return 0.0

    return -np.inf


d = np.loadtxt('spec.txt').T
sel = (d[0] > 8680) & (d[0] < 8696)
yerr = np.ones_like(d[0][sel]) * 0.01

lines = [(d[0][sel], d[1][sel], yerr)]
pfiles = [profiles.gaussian]

pparn = np.cumsum([0] +\
        [len(inspect.getargspec(i)[0]) - 1 for i in pfiles])

###############################################################################
# GP modelled line
initial = [0.28, 8687.82, 1.53, -6.1, 0.3]

nwalkers = 128
ndim = len(initial)
niter = 100

noisemodel = single_kernel_noisemodel 

data = [lines, pfiles, pparn, noisemodel, single_kernel_lnprior]
p0 = np.array([np.array(initial) + 1e-2 * np.random.randn(ndim)
               for i in xrange(nwalkers)])

sampler = emcee.EnsembleSampler(nwalkers, ndim, gpew.lnprob, args=data)
p0, lnp, _ = sampler.run_mcmc(p0, niter)
sampler.reset()
p = p0[np.argmax(lnp)]
p0 = [p + 1e-2 * np.random.randn(ndim) for i in xrange(nwalkers)]
p0, _, _ = sampler.run_mcmc(p0, niter)

samples = sampler.flatchain

xcen = samples[:, 1]
mxcen = np.mean(xcen)
xs = np.linspace(-8.1, 8.1, 100)

models = []
clean_models = []
ew = []
for s in samples[np.random.randint(len(samples), size=100)]:
    pars = s[pparn[0]:pparn[1]]
    profile = 1 - pfiles[0](lines[0][0], *pars)
    profilexs = 1 - pfiles[0](xs + mxcen, *pars)
    clean_models.append(profilexs)
    ew.append(np.sum((1 - profilexs[1:]) * (xs[1:] - xs[:-1])))

    if noisemodel is not None:
        nmp = np.exp(s[pparn[-1]:])
        nm = noisemodel(nmp)
        nm.compute(lines[0][0], lines[0][2])

        m = nm.sample_conditional(lines[0][1] - profile,
                                  xs + mxcen) + profilexs
        models.append(m)

offset = 0.0

pl.errorbar(lines[0][0] - mxcen, lines[0][1] + offset, yerr=lines[0][2],
                fmt=".k", capsize=0)

pl.text(xs[0], offset + 1.02, '%.2f +- %.2f' % (np.mean(ew),
        np.std(ew)))

la = np.array(clean_models).T
lstd = np.std(la, axis=1)
lavg = np.average(la, axis=1)
y1, y2 = lavg + lstd + offset, lavg - lstd + offset
pl.fill_between(xs, y1, y2, alpha=0.3)

gpa = np.array(models).T
gpstd = np.std(gpa, axis=1)
gpavg = np.average(gpa, axis=1)
y1, y2 = gpavg + gpstd + offset, gpavg - gpstd + offset
pl.fill_between(xs, y1, y2, color='r', alpha=0.3)

###############################################################################
# Chi2 modelled line
initial = [0.28, 8687.82, 1.53]

ndim = len(initial)

noisemodel = None

data = [lines, pfiles, pparn, noisemodel, chi2_lnprior]
p0 = np.array([np.array(initial) + 1e-2 * np.random.randn(ndim)
               for i in xrange(nwalkers)])

sampler = emcee.EnsembleSampler(nwalkers, ndim, gpew.lnprob, args=data)
p0, lnp, _ = sampler.run_mcmc(p0, niter)
sampler.reset()
p = p0[np.argmax(lnp)]
p0 = [p + 1e-2 * np.random.randn(ndim) for i in xrange(nwalkers)]
p0, _, _ = sampler.run_mcmc(p0, niter)

samples = sampler.flatchain

xcen = samples[:, 1]
mxcen = np.mean(xcen)

clean_models = []
ew = []
for s in samples[np.random.randint(len(samples), size=100)]:
    pars = s[pparn[0]:pparn[1]]
    profilexs = 1 - pfiles[0](xs + mxcen, *pars)
    clean_models.append(profilexs)
    ew.append(np.sum((1 - profilexs[1:]) * (xs[1:] - xs[:-1])))

offset = 0.3

pl.errorbar(lines[0][0] - mxcen, lines[0][1] + offset, yerr=lines[0][2],
                fmt=".k", capsize=0)

pl.text(xs[0], offset + 1.02, '%.2f +- %.2f' % (np.mean(ew),
        np.std(ew)))

la = np.array(clean_models).T
lstd = np.std(la, axis=1)
lavg = np.average(la, axis=1)
y1, y2 = lavg + lstd + offset, lavg - lstd + offset
pl.fill_between(xs, y1, y2, alpha=0.3)

pl.show()
