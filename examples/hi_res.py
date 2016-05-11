#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
import numpy as np

import emcee
import george
from george import kernels

import profiles
import gpew


def single_kernel_noisemodel(p):
    """
    Simple one squared-exponential kernel noise model.
    """
    return george.GP(p[0] * kernels.ExpSquaredKernel(p[1]))


def three_kernel_noisemodel(p):
    return george.GP(p[0] * kernels.ExpSquaredKernel(p[1]) +
                     p[2] * kernels.ExpSquaredKernel(p[3]) +
                     p[4] * kernels.ExpSquaredKernel(p[5]))


def single_kernel_lnprior(p):
    amp, xcen, sigma, lna, lnalpha = p

    lnp = 0.0

    if (-50. < lna < 0. and p[1] > 0. and sigma > 0.):
        return lnp

    return -np.inf


def three_kernel_lnprior(p):
    amp, xcen, sigma, lna, lnalpha, lnb, lnbeta, lnc, lnzeta = p

    lnp = 0.0
    lnp += -(lnalpha + 0.5) ** 2 / (2 * 0.6 ** 2)
    lnp += -(lnbeta - 1.25) ** 2 / (2 * 0.8 ** 2)
    lnp += -(lnzeta - 3.0) ** 2 / (2 * 0.6 ** 2)

    if (-50. < lna < 0. and -50. < lnb < 0. and -50. < lnc < 0.
        and 8685 < xcen < 8690.
        and amp > 0.
        and sigma > 0.):
        return lnp

    return -np.inf


def multiple_lines_lnprior(p):
    amp1, xcen1, sigma1, amp2, xcen2, sigma2, gamma2,\
            amp3, xcen3, sigma3, amp4, xcen4, sigma4, lna, lnalpha = p

    lnp = 0.0
    #  lnp += -(lnalpha + 0.3) ** 2 / (2 * 1.0 ** 2)

    if (-50. < lna < 0.):
        return lnp

    return -np.inf


def single_line_single_kernel(compute_samples=True):

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
    initial = [0.28, 8687.82, 1.53, -6.1, 0.3]

    # Sampler initialization
    nwalkers = 128
    ndim = len(initial)

    # 100 is not enough! Make sure the convergence is satisfacory before
    # accepting any results!
    niter = 500
    data = [lines, pfiles, pparn, single_kernel_noisemodel,
            single_kernel_lnprior]

    # Initial states of the walkers - N-dim Gaussian around the initial values
    p0 = np.array([np.array(initial) + 1e-2 * np.random.randn(ndim)
                   for i in xrange(nwalkers)])

    if compute_samples:
        # Sampler object
        sampler = emcee.EnsembleSampler(nwalkers, ndim, gpew.lnprob, args=data)

        # Let's run it!
        p0, lnp, _ = sampler.run_mcmc(p0, niter)
        sampler.reset()

        # Let's get the best lnp value and re-initialize it again from there.
        p = p0[np.argmax(lnp)]
        p0 = [p + 1e-2 * np.random.randn(ndim) for i in xrange(nwalkers)]
        p0, _, _ = sampler.run_mcmc(p0, niter)

        # Collect the samples
        samples = sampler.flatchain

        gpew.save_samples('test.npy', samples)

    else:
        samples = gpew.load_samples('test.npy')

    # Plot stuff:
    # error bars: observed line
    # red: +-1 sigma of the complete model
    # blue: +-1 sigma of the profile model
    gpew.plot_lines(lines, pfiles, pparn, single_kernel_noisemodel, samples,
                    nwalkers, wlwidth=9, gpsamples=20)


def single_line_three_kernels(compute_samples=True):

    d = np.loadtxt('spec.txt').T
    sel = (d[0] > 8680) & (d[0] < 8696)
    yerr = np.ones_like(d[0][sel]) * 0.013

    lines = [(d[0][sel], d[1][sel], yerr)]
    pfiles = [profiles.gaussian]

    pparn = np.cumsum([0] +\
            [len(inspect.getargspec(i)[0]) - 1 for i in pfiles])

    initial = [0.28, 8687.82, 1.53, -6.1, 0.3, -3.0, 1.0, -3.0, 1.0]

    nwalkers = 128
    ndim = len(initial)
    niter = 100
    data = [lines, pfiles, pparn, three_kernel_noisemodel,
            three_kernel_lnprior]

    p0 = np.array([np.array(initial) + 1e-2 * np.random.randn(ndim)
                   for i in xrange(nwalkers)])

    if compute_samples:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, gpew.lnprob, args=data)
        p0, lnp, _ = sampler.run_mcmc(p0, niter)
        sampler.reset()

        p = p0[np.argmax(lnp)]
        p0 = [p + 1e-2 * np.random.randn(ndim) for i in xrange(nwalkers)]
        p0, _, _ = sampler.run_mcmc(p0, niter)

        samples = sampler.flatchain
        gpew.save_samples('test.npy', samples)

    else:
        samples = gpew.load_samples('test.npy')

    gpew.plot_lines(lines, pfiles, pparn, three_kernel_noisemodel, samples,
                    nwalkers, wlwidth=9, gpsamples=20)


def multiple_lines_single_kernel(compute_samples=True):

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

    if compute_samples:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, gpew.lnprob, args=data)
        p0, lnp, _ = sampler.run_mcmc(p0, niter)
        sampler.reset()

        p = p0[np.argmax(lnp)]
        p0 = [p + 1e-2 * np.random.randn(ndim) for i in xrange(nwalkers)]
        p0, _, _ = sampler.run_mcmc(p0, niter)

        samples = sampler.flatchain
        gpew.save_samples('test.npy', samples)

    else:
        samples = gpew.load_samples('test.npy')

    # We need to provied indices of the amplitudes and offsets of 
    # individual lines so that the thing knows where to plot what
    gpew.plot_lines(lines, pfiles, pparn, single_kernel_noisemodel, samples,
                    nwalkers, [0, 3, 7, 10], [1, 4, 8, 11],
                    wlwidth=9, gpsamples=100)


def hi_res_lnprior(p):
    amp, xcen, sigma, gamma, lna, lnalpha = p

    lnp = 0.0

    if (-50. < lna < 0. and amp > 0.1 and sigma > 0. and sigma < 1.0 and gamma > 0. and
            xcen > 5895.5 and xcen < 5896.5):
        return lnp

    return -np.inf


def hi_res_spectrum(compute_samples=True):
    d = np.loadtxt('spec_hires_short.txt').T

    sel = (d[0] > 589.3) & (d[0] < 589.9)

    lines = [(d[0][sel][::10] * 10, d[1][sel][::10], d[2][sel][::10])]

    pfiles = [profiles.lorentzian]

    pparn = np.cumsum([0] +\
            [len(inspect.getargspec(i)[0]) - 1 for i in pfiles])

    initial = [0.5, 5896, 0.5, 1.0, -6.1, 0.3]

    nwalkers = 128
    ndim = len(initial)

    niter = 100
    data = [lines, pfiles, pparn, single_kernel_noisemodel,
            hi_res_lnprior]

    p0 = np.array([np.array(initial) + 1e-2 * np.random.randn(ndim)
                   for i in xrange(nwalkers)])

    if compute_samples:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, gpew.lnprob, args=data)
        p0, lnp, _ = sampler.run_mcmc(p0, niter)
        sampler.reset()

        p = p0[np.argmax(lnp)]
        p0 = [p + 1e-2 * np.random.randn(ndim) for i in xrange(nwalkers)]
        p0, _, _ = sampler.run_mcmc(p0, niter)

        samples = sampler.flatchain
        gpew.save_samples('test.npy', samples)

    else:
        samples = gpew.load_samples('test.npy')

    gpew.plot_lines(lines, pfiles, pparn, single_kernel_noisemodel, samples,
                    nwalkers, wlwidth=5, gpsamples=20, profilepoints=500)


#  single_line_single_kernel(compute_samples=1)
single_line_three_kernels(compute_samples=1)
#  multiple_lines_single_kernel(compute_samples=0)
#  hi_res_spectrum(1)

