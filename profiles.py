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
