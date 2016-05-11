#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import scipy.special
import matplotlib.pyplot as pl


def lnlike(p, lines, profiles, pparn, noisemodel):
    """
    Log-likelihood function that is passed to the sampler.

    :param p:
        First parameters for all the profiles, then parameters
        for the noise model in ln.
        
    :param lines:
        Array of lines. Each element should have three subarrays:
        wavelength, flux and flux uncertainty.

    :param profiles:
        Profiles for each of the lines. The length of this array must
        match the length of `lines`.

    :param pparn:
        Array of cumsum of number of parameters for each profile starting
        with 0.

    :param noisemodel:
        Can be `None` or something like:
        gp = george.GP(a * kernels.ExpSquaredKernel(alpha) +
                       b * kernels.ExpSquaredKernel(beta) +
                       c * kernels.ExpSquaredKernel(zeta))

    """
    
    # Noise model parameters after profile parameters
    if noisemodel is not None:
        nmp = np.exp(p[pparn[-1]:])
        nm = noisemodel(nmp)

    try:
        lnl = 0.0
        for i, d in enumerate(zip(lines, profiles)):
            x = d[0][0]
            y = d[0][1]
            yerr = d[0][2]
            pars = (p[pparn[i]:pparn[i + 1]])
            profile = 1 - d[1](x, *pars)
        
            if noisemodel is not None:
                try:
                    nm.compute(x, yerr)
                #  except np.linalg.linalg.LinAlgError, ValueError:
                except:
                    return -np.inf

                if np.inf in abs(profile):
                    lnl = -np.inf
                else:
                    lnl += nm.lnlikelihood(y - profile, quiet=True)

            else:
                lnl += -0.5 * (np.sum(((y - profile) / yerr) ** 2))

        print p, lnl
        return lnl
    except np.linalg.linalg.LinAlgError, ValueError:
        return -np.inf


def lnprob(p, lines, profiles, pparn, noisemodel, lnprior):
    lp = lnprior(p)
    ll = lnlike(p, lines, profiles, pparn, noisemodel)
    return lp + ll if np.isfinite(lp) else -np.inf


def save_samples(filename, samples):
    f = open(filename, 'wb')
    np.save(filename, samples)
    f.close()


def load_samples(filename):
    samples = np.load(filename)
    return samples


def plot_lines(lines, profiles, pparn, noisemodel, samples, nwalkers, iamp=[0],
               ixcen=[1], wlwidth=12, gpsamples=10, profilepoints=100):

    offset = 0.0
    xs = np.linspace(-wlwidth, wlwidth, profilepoints)

    mamp = []
    for i, line in enumerate(lines):
        amp = samples[:, iamp[i]]
        mamp.append(np.mean(amp))
    
    mamp = max(mamp)


    for i, line in enumerate(lines):
        xcen = samples[:, ixcen[i]]
        mxcen = np.mean(xcen)

        models = []
        clean_models = []
        ew = []
        for s in samples[np.random.randint(len(samples), size=gpsamples)]:
            pars = s[pparn[i]:pparn[i + 1]]
            profile = 1 - profiles[i](line[0], *pars)
            profilexs = 1 - profiles[i](xs + mxcen, *pars)
            clean_models.append(profilexs)
            ew.append(np.sum((1 - profilexs[1:]) * (xs[1:] - xs[:-1])))

            if noisemodel is not None:
                nmp = np.exp(s[pparn[-1]:])
                nm = noisemodel(nmp)
                nm.compute(line[0], line[2])

                m = nm.sample_conditional(line[1] - profile,
                                          xs + mxcen) + profilexs
                models.append(m)

        offset += mamp if i else 0

        pl.errorbar(line[0] - mxcen, line[1] + offset, yerr=line[2],
                        fmt=".k", capsize=0)

        pl.text(xs[0], offset + 1.02, '%.2f +- %.2f' % (np.mean(ew),
                np.std(ew)))

        la = np.array(clean_models).T
        lstd = np.std(la, axis=1)
        lavg = np.average(la, axis=1)
        y1, y2 = lavg + lstd + offset, lavg - lstd + offset
        pl.fill_between(xs, y1, y2, alpha=0.3)

        if len(models):
            gpa = np.array(models).T
            gpstd = np.std(gpa, axis=1)
            gpavg = np.average(gpa, axis=1)
            y1, y2 = gpavg + gpstd + offset, gpavg - gpstd + offset
            pl.fill_between(xs, y1, y2, color='r', alpha=0.3)

    pl.show()
