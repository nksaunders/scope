#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Scope Math
----------
Mathematical functions for simulating light curves.
'''

from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
import matplotlib.pyplot as pl
from scipy.special import erf
from scipy.integrate import quad, dblquad
from scipy.optimize import fmin_powell
from sklearn.decomposition import PCA
from itertools import combinations_with_replacement as multichoose
import timeit, builtins
import george

from .utils import *

def Polynomial(x, coeffs):
  """Returns a polynomial with coefficients `coeffs` evaluated at `x`."""


  return np.sum([c * x ** m for m, c in enumerate(coeffs)], axis = 0)

class GaussInt(object):
    """Returns the definite integral of x^n * exp(-ax^2 + bx + c) from 0 to 1."""

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        p = np.sqrt(self.a)
        q = self.b / (2 * p)
        # self.GI0 = np.exp(q ** 2 + self.c) * np.sqrt(np.pi) * (erf(q) + erf(p - q)) / (2 * p)
        self.GI0 = (np.sqrt(np.pi) / 2 * np.sqrt(a)) * np.exp(b**2 / (4 * a) + c) * (erf((2 * a - b) / (2 * np.sqrt(a))) + erf(b / (2 * np.sqrt(a))))

    def __call__(self, n):
        a = self.a
        b = self.b
        c = self.c
        if n == 0:
            return self.GI0
        elif n == 1:
            # return (1 / (2 * self.a)) * (np.exp(self.c) * (1 - np.exp(self.b - self.a)) + self.b * self.GI0)
            return (np.exp(c - a) / (4 * a**(3 / 2))) * (np.sqrt(np.pi) * b * np.exp(b**2 / (4 * a) + a) * \
                   (erf((2 * a - b) / (2 * np.sqrt(a))) + erf(b / (2 * np.sqrt(a)))))
        elif n == 2:
            # return (1 / (4 * self.a ** 2)) * (np.exp(self.c) * (self.b - (2 * self.a + self.b) * np.exp(self.b - self.a)) + (2 * self.a + self.b ** 2) * self.GI0)
            return (np.exp(c - a) / (8 * a**(5/2))) * (np.sqrt(np.pi) * (2 * a + b) * np.exp(b**2 / (4 * a) + a) * \
                   (erf((2 * a - b) / (2 * np.sqrt(a))) + erf(b / (2 * np.sqrt(a)))) - 2 * np.sqrt(a) * (np.exp(b) * \
                   (2 * a + b) - np.exp(a) * b))
        elif n == 3:
            return (1 / (8 * self.a ** 3)) * (np.exp(self.c) * (4 * self.a + self.b ** 2 - (4 * self.a ** 2 + 4 * self.a + 2 * self.a + self.b + self.b ** 2) * np.exp(self.b - self.a)) + self.b * (6 * self.a + self.b ** 2) * self.GI0)
        else:
            raise NotImplementedError("Intrapixel variability above 3rd order still needs to be added.")

def PolyGaussIntegrand1D(y, cx, cy, amp, x0, y0, sx, sy, rho):
    """
    This is the product of the 2D Gaussian PSF, a polynomial in y,
    and a polynomial in x, *analytically integrated along x*. The integral
    along y is not analytic and must be done numerically. However, the
    analytical integration along the first dimension speeds up the entire
    calculation by a factor of ~20.
    """

    amp = np.atleast_1d(amp)
    x0 = np.atleast_1d(x0)
    y0 = np.atleast_1d(y0)
    sx = np.atleast_1d(sx)
    sy = np.atleast_1d(sy)
    rho = np.atleast_1d(rho)

    # Dimensions
    N = len(cy)
    K = len(amp)

    # Get the y IPV
    f = Polynomial(y, cy)

    # Our integrand is the expression f * g
    g = y * 0.

    # Loop over the components of the PSF
    for k in range(K):

        # Get the x Gaussian integrals

        '''
        a = 1 / (2 * (1 - rho[k] ** 2) * sx[k] ** 2)
        b = ((y - y0[k]) * rho[k] * sx[k] + x0[k] * sy[k]) / ((1 - rho[k] ** 2) * sx[k] ** 2 * sy[k])
        c = -(x0[k] ** 2 / sx[k] ** 2 + (y - y0[k]) ** 2 / sy[k] ** 2 + 2 * x0[k] * (y - y0[k]) * rho[k] / (sx[k] * sy[k])) / (2 * (1 - rho[k] ** 2))
        norm = (2 * np.pi * sx[k] * sy[k] * np.sqrt(1 - rho[k] ** 2))


        a = 1 / (4 * (1 - rho ** 2) * sx ** 2)
        b = ((y - y0[k]) * rho * sx + x0[k] * sy) / ((1 - rho ** 2) * sx ** 2 * sy)
        c = -(x0[k] ** 2 / sx ** 2 + (y - y0[k]) ** 2 / sy ** 2 + 2 * x0[k] * (y - y0[k]) * rho / (sx * sy)) / (2 * (1 - rho ** 2))
        '''

        norm = (2 * np.pi * sx * sy * np.sqrt(1 - rho ** 2))
        a = 1 / (4 * sx ** 2 * (1 - rho ** 2))
        b = (- 1 / (2 * (1 - rho ** 2))) * ((x0[k] / sx ** 2) - (2 * rho * (y - y0[k])) / (sx * sy))
        c = (- 1 / (2 * (1 - rho ** 2))) * ((x0 ** 2 / (2 * sx ** 2)) + (y0 ** 2 / (2 * sy ** 2)) - (2 * rho * x0[k] * (y - y0)) / (sx * sy))

        GI = GaussInt(a, b, c)

        # Loop over the orders of the x IPV
        for n in range(N):
            g += cx[n] * GI(n)

    # We're done!
    return f * g * np.sum(amp) / (2 * norm ** len(amp))

def Gauss2D(x, y, amp, x0, y0, sx, sy, rho):
    """A two-dimensional gaussian with arbitrary covariance."""

    norm = (2 * np.pi * sx * sy * np.sqrt(1 - rho ** 2))
    return (amp / norm) * np.exp(-((x - x0) ** 2 / sx ** 2 + (y - y0) ** 2 / sy ** 2 - 2 * rho * (x - x0) * (y - y0) / (sx * sy)) / (2 * (1 - rho ** 2)))

def PolyGaussIntegrand2D(x, y, cx, cy, amp, x0, y0, sx, sy, rho):
    """
    This is straight up the product of the 2D Gaussian PSF, a polynomial in y,
    and a polynomial in x, at a given location on the pixel. Integrating this
    in 2D across the entire pixel yields the total flux in that pixel.
    """

    # Dimensions
    K = len(x0)

    amp = np.atleast_1d(amp)
    x0 = np.atleast_1d(x0)
    y0 = np.atleast_1d(y0)
    sx = np.atleast_1d(sx)
    sy = np.atleast_1d(sy)
    rho = np.atleast_1d(rho)

    # Get the IPV functions
    f = Polynomial(y, cy)
    g = Polynomial(x, cx)

    # Loop over the components of the PSF
    h = np.sum([Gauss2D(x, y, amp[k], x0[k], y0[k], sx[k], sy[k], rho[k]) for k in range(K)], axis = 0)

    # We're done!
    return f * g * h

def PixelFlux(cx, cy, amp, x0, y0, sx, sy, rho, fast=True, **kwargs):
    """
    The flux in a given pixel of the detector, calculated from the integral
    of the convolution of a 2D gaussian with a polynomial.

    `cx`:
        The intra-pixel variability polynomial coefficients along the `x` axis, expressed as a list from 0th to 3rd order.
    `cy`:
        The intra-pixel variability polynomial coefficients along the `y` axis, expressed as a list from 0th to 3rd order.
    `amp`:
        The amplitude of the normalized gaussian (the integral of the gaussian over the entire xy plane is equal to this value).
    `x0`:
        The `x` position of the center of the gaussian relative to the left pixel edge.
    `y0`:
        The `y` position of the center of the gaussian relative to the bottom pixel edge.
    `sx`:
        The standard deviation of the gaussian in the `x` direction (before rotation).
    `sy`:
        The standard deviation of the gaussian in the `y` direction (before rotation).
    `rho`:
        The correlation coefficient between `x` and `y`, a value between -1 and 1. See en.wikipedia.org/wiki/Pearson_correlation_coefficient. If this is 0, `x` and `y` are uncorrelated (zero rotation).
    `fast`:
        If `True`, analytically integrates the function along the `x` axis and numerically integrates it along the `y` axis. This can greatly speed things up, with no loss of accuracy. If `False`, numerically integrates in both dimensions (not recommended).
    """

    if fast:
        F = lambda y: PolyGaussIntegrand1D(y, cx, cy, amp, x0, y0, sx, sy, rho)
        res, err = quad(F, 0, 1, **kwargs)
    else:
        F = lambda y, x: PolyGaussIntegrand2D(x, y, cx, cy, amp, x0, y0, sx, sy, rho)
        res, err = dblquad(F, 0, 1, lambda x: 0, lambda x: 1, **kwargs)

    return res

def TestIntegration():
    """
    Compares the fast and slow ways of computing the flux. Reports the
    time each method took and the difference in the flux between the
    two methods.
    """

    # Define the params
    cx = np.random.randn(3); cx[0] = np.abs(cx[0])
    cy = np.random.randn(3); cy[0] = np.abs(cy[0])
    amp = [1.]
    x0 = np.random.randn(1)
    y0 = np.random.randn(1)
    sx = 0.5 + 0.1 * np.random.randn(1)
    sy = 0.5 + 0.1 * np.random.randn(1)
    rho = 2 * (np.random.rand(1) - 0.5)

    # Define our semi-analytic and numerical integrators
    fsem = lambda: PixelFlux(cx, cy, amp, x0, y0, sx, sy, rho, semi = True)
    fnum = lambda: PixelFlux(cx, cy, amp, x0, y0, sx, sy, rho, semi = False)

    # Time the calls to each function
    builtins.__dict__.update(locals())
    tsem = timeit.timeit('fsem()', number = 100) / 100.
    tnum = timeit.timeit('fnum()', number = 100) / 100.

    # Print
    print("Semi-analytic (%.1e s): %.9e" % (tsem, fsem()))
    print("Numerical     (%.1e s): %.9e" % (tnum, fnum()))
    print("Difference    (   %.1f x): %.9e" % (tnum/tsem, np.abs(1 - fnum()/fsem())))

def PSF(A, psf_args, ccd_args, xpos, ypos):
    """
    Computes a stellar Point Spread Function (PSF) from given parameters.

    Parameters
    ----------
    psf_args : dict
        dictionary of PSF parameters
    ccd_args : dict
        dictionary of CCD parameters
    xpos : arraylike
        array of PSF motion in x around detector relative to `(x_0, y_0)`
    ypos : arraylike
        array of PSF motion in y around detector relative to `(x_0, y_0)`
    """

    # Read in detector arguments
    cx = ccd_args['cx']
    cy = ccd_args['cy']
    apsize = ccd_args['apsize']
    background_level = ccd_args['background_level']
    inter = ccd_args['inter']
    photnoise_conversion = ccd_args['photnoise_conversion']

    # Define apertures
    psf = np.zeros((apsize, apsize))
    psferr = np.zeros((apsize, apsize))
    target = np.zeros((apsize, apsize))

    for i in range(apsize):
        for j in range(apsize):

            # read in PSF arguments
            x0 = np.atleast_1d(psf_args['x0'])
            y0 = np.atleast_1d(psf_args['y0'])
            sx = np.atleast_1d(psf_args['sx'])
            sy = np.atleast_1d(psf_args['sy'])
            rho = np.atleast_1d(psf_args['rho'])

            # contribution to pixel from target
            psf[i][j] = PixelFlux(cx, cy, A,
                                  [(x-i+xpos) for x in x0],
                                  [(y-j+ypos) for y in y0],
                                  sx, sy, rho)

            target[i][j] = psf[i][j]

            # add background noise
            noise = np.sqrt(np.abs(background_level * np.random.randn()))
            psf[i][j] += noise

            # add photon noise
            psferr[i][j] = np.sqrt(np.abs(psf[i][j]) * photnoise_conversion)
            randnum = np.random.randn()
            psf[i][j] += psferr[i][j] * randnum

            # ensure positive
            while psf[i][j] < 0:
                psf[i][j] = np.sqrt(np.abs(background_level * np.random.randn()))

    # multiply each cadence by inter-pixel sensitivity variation
    psf *= inter

    return psf, target, psferr

def PLD(fpix, ferr, trninds, t, aperture):
    """
    Perform first order PLD on a light curve

    Returns
    -------
    flux :
        detrended light curve
    rawflux :
        raw light curve
    """

    aperture = [aperture for i in range(len(fpix))]

    M = lambda x: np.delete(x, trninds, axis=0)

    #  generate flux light curve
    fpix = M(fpix)
    aperture = M(aperture)
    fpix_rs = (fpix*aperture).reshape(len(fpix),-1)
    fpix_ap = np.zeros((len(fpix),len(np.delete(fpix_rs[0],np.where(np.isnan(fpix_rs[0]))))))

    for c in range(len(fpix_rs)):
        naninds = np.where(np.isnan(fpix_rs[c]))
        fpix_ap[c] = np.delete(fpix_rs[c],naninds)

    fpix = fpix_ap
    rawflux = np.sum(fpix.reshape(len(fpix),-1), axis=1)

    # First order PLD
    f1 = fpix / rawflux.reshape(-1, 1)
    pca = PCA(n_components=10)
    X1 = pca.fit_transform(f1)

    # Second order PLD
    f2 = np.product(list(multichoose(f1.T, 2)), axis = 1).T
    pca = PCA(n_components=10)
    X2 = pca.fit_transform(f2)

    # Combine them and add a column vector of 1s for stability
    X = np.hstack([np.ones(X1.shape[0]).reshape(-1, 1), X1, X2])

    # Mask transits in design matrix
    MX = M(X)

    try:
        # Define gaussian process parameters
        y = M(rawflux) - np.dot(X, np.linalg.solve(np.dot(X.T, X), np.dot(X.T, M(rawflux))))
        amp = np.nanstd(y)
        tau = 30.
    except:
        raise ScopeError('`numpy.linalg.solve` returned a singular matrix. The '
                         'flux array may contain too few cadences to compute '
                         'Gaussian Process parameters.')

    # Set up gaussian process
    gp = george.GP(amp ** 2 * george.kernels.Matern32Kernel(tau ** 2))
    sigma = gp.get_matrix(M(t)) + np.diag(M(np.sum(ferr.reshape(len(ferr),-1), axis = 1))**2)

    # Compute
    A = np.dot(MX.T, np.linalg.solve(sigma, MX))
    B = np.dot(MX.T, np.linalg.solve(sigma, M(rawflux)))
    C = np.linalg.solve(A, B)

    # Compute detrended light curve
    model = np.dot(X, C)
    flux = rawflux - model + np.nanmean(rawflux)

    return flux, rawflux

def _calculate_PSF_amplitude(mag):
    """
    Returns the amplitude of the PSF for a star of a given magnitude.

    Parameters
    ----------
    `mag`: float
        Input magnitude.

    Returns
    -------
    amp : float
        Corresponding PSF applitude.
    """

    # mag/flux relation constants
    amp = 10**(-0.4*(mag - 12))*1.74e5
    return amp

if __name__ == '__main__':

  TestIntegration()
