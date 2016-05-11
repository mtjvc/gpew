#!/usr/bin/env python
# -*- coding: utf-8 -*-

# If you want to add your own profile, make sure that the first three
# arguments are (in this particular order):
#   xarr - array of wavelengths in which we compute the profile
#   amp  - amplitude of the profile
#   xcen - central position of the profile
# Whatever follow is optional.

import numpy as np
import scipy.special


def gaussian(xarr, amp, xcen, sigma):
    g = amp * np.exp(-(xarr - xcen) ** 2 / np.sqrt(2 * sigma ** 2))
    return g


def lorentzian(xarr, amp, xcen, fwhm) :
    l = amp / (1.0 + ((xarr - xcen) / fwhm)**2)
    return l


def voigt(xarr, amp, xcen, sigma, gamma):
    z = ((xarr - xcen) + 1j * gamma) / (sigma * np.sqrt(2))
    v = amp * np.real(scipy.special.wofz(z))
    return v


def two_gaussian_blend(xarr, amp1, xcen1, sigma1, amp2, xcen2, sigma2):
    g1 = amp1 * np.exp(-(xarr - xcen1) ** 2 / np.sqrt(2 * sigma1 ** 2))
    g2 = amp2 * np.exp(-(xarr - xcen2) ** 2 / np.sqrt(2 * sigma2 ** 2))
    return g1 + g2


if __name__ == '__main__':
    import matplotlib.pyplot as pl
    xarr = np.linspace(-10, 10, 300)
    p = two_gaussian_blend(xarr, 0.3, -2.0, 1.0, 0.5, 0.0, 1.0)
    pl.plot(xarr, 1 - p)
    pl.show()
