#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
intrapix.py
-----------

'''

from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
import matplotlib.pyplot as pl
from scipy.special import erf
from scipy.integrate import quad, dblquad
import timeit, builtins

def Polynomial(x, coeffs):
  '''
  Returns a polynomial with coefficients `coeffs` evaluated at `x`.
  
  '''
  
  return np.sum([c * x ** m for m, c in enumerate(coeffs)], axis = 0)

class GaussInt(object):
  '''
  Returns the definite integral of x^n * exp(-ax^2 + bx + c) from 0 to 1.
  
  '''
  
  def __init__(self, a, b, c):
    '''
    
    '''
    
    self.a = a
    self.b = b
    self.c = c
    p = np.sqrt(self.a)
    q = self.b / (2 * p)
    self.GI0 = np.exp(q ** 2 + self.c) * np.sqrt(np.pi) * (erf(q) + erf(p - q)) / (2 * p)

  def __call__(self, n):
    '''
    
    '''
    
    if n == 0:
      return self.GI0
    elif n == 1:
      return (1 / (2 * self.a)) * (np.exp(self.c) * (1 - np.exp(self.b - self.a)) + self.b * self.GI0)
    elif n == 2:
      return (1 / (4 * self.a ** 2)) * (np.exp(self.c) * (self.b - (2 * self.a + self.b) * np.exp(self.b - self.a)) + (2 * self.a + self.b ** 2) * self.GI0)
    elif n == 3:
      return (1 / (8 * self.a ** 3)) * (np.exp(self.c) * (4 * self.a + self.b ** 2 - (4 * self.a ** 2 + 4 * self.a + 2 * self.a + self.b + self.b ** 2) * np.exp(self.b - self.a)) + self.b * (6 * self.a + self.b ** 2) * self.GI0)
    else:
      raise NotImplementedError("Intrapixel variability above 3rd order still needs to be added.")

def PolyGaussIntegrand1D(y, cx, cy, amp, x0, y0, sx, sy, rho):
  '''
  This is the product of the 2D Gaussian PSF, a polynomial in y,
  and a polynomial in x, *analytically integrated along x*. The integral
  along y is not analytic and must be done numerically. However, the
  analytical integration along the first dimension speeds up the entire
  calculation by a factor of ~20.
  
  '''
  
  # Dimensions
  N = len(cy)
  K = len(x0)
  
  # Get the y IPV
  f = Polynomial(y, cy)
  
  # Our integrand is the expression f * g
  g = y * 0.
  
  # Loop over the components of the PSF
  for k in range(K):
  
    # Get the x Gaussian integrals
    a = 1 / (2 * (1 - rho[k] ** 2) * sx[k] ** 2)
    b = ((y - y0[k]) * rho[k] * sx[k] + x0[k] * sy[k]) / ((1 - rho[k] ** 2) * sx[k] ** 2 * sy[k])
    c = -(x0[k] ** 2 / sx[k] ** 2 + (y - y0[k]) ** 2 / sy[k] ** 2 + 2 * x0[k] * (y - y0[k]) * rho[k] / (sx[k] * sy[k])) / (2 * (1 - rho[k] ** 2))
    norm = (2 * np.pi * sx[k] * sy[k] * np.sqrt(1 - rho[k] ** 2))
    GI = GaussInt(a, b, c)
  
    # Loop over the orders of the x IPV
    for n in range(N):
      g += (amp[k] / norm) * cx[n] * GI(n)
  
  # We're done!
  return f * g

def Gauss2D(x, y, amp, x0, y0, sx, sy, rho):
  '''
  A two-dimensional gaussian with arbitrary covariance.
  
  '''
  
  norm = (2 * np.pi * sx * sy * np.sqrt(1 - rho ** 2))
  return (amp / norm) * np.exp(-((x - x0) ** 2 / sx ** 2 + (y - y0) ** 2 / sy ** 2 - 2 * rho * (x - x0) * (y - y0) / (sx * sy)) / (2 * (1 - rho ** 2)))

def PolyGaussIntegrand2D(x, y, cx, cy, amp, x0, y0, sx, sy, rho):
  '''
  This is straight up the product of the 2D Gaussian PSF, a polynomial in y,
  and a polynomial in x, at a given location on the pixel. Integrating this
  in 2D across the entire pixel yields the total flux in that pixel.
  
  '''
  
  # Dimensions
  K = len(x0)
  
  # Get the IPV functions
  f = Polynomial(y, cy)
  g = Polynomial(x, cx)
  
  # Loop over the components of the PSF
  h = np.sum([Gauss2D(x, y, amp[k], x0[k], y0[k], sx[k], sy[k], rho[k]) for k in range(K)], axis = 0)

  # We're done!
  return f * g * h

def PixelFlux(cx, cy, amp, x0, y0, sx, sy, rho, fast = True, **kwargs):
  '''
  The flux in a given pixel of the detector, calculated from the integral
  of the convolution of a 2D gaussian with a polynomial.
  
  `cx`: The intra-pixel variability polynomial coefficients along the `x`
        axis, expressed as a list from 0th to 3rd order.
  `cy`: The intra-pixel variability polynomial coefficients along the `y`
        axis, expressed as a list from 0th to 3rd order.
  `amp`: The amplitude of the normalized gaussian (the integral of the
        gaussian over the entire xy plane is equal to this value).
  `x0`: The `x` position of the center of the gaussian relative to the
        left pixel edge.
  `y0`: The `y` position of the center of the gaussian relative to the
        bottom pixel edge.
  `sx`: The standard deviation of the gaussian in the `x` direction
        (before rotation).
  `sy`: The standard deviation of the gaussian in the `y` direction
        (before rotation).
  `rho`: The correlation coefficient between `x` and `y`, a value between
         -1 and 1. See en.wikipedia.org/wiki/Pearson_correlation_coefficient.
         If this is 0, `x` and `y` are uncorrelated (zero rotation).
  `fast`: If `True`, analytically integrates the function along the `x`
          axis and numerically integrates it along the `y` axis. This
          can greatly speed things up, with no loss of accuracy. If `False`,
          numerically integrates in both dimensions (not recommended).
  
  '''
  
  if fast:
    F = lambda y: PolyGaussIntegrand1D(y, cx, cy, amp, x0, y0, sx, sy, rho)
    res, err = quad(F, 0, 1, **kwargs)
  else:
    F = lambda y, x: PolyGaussIntegrand2D(x, y, cx, cy, amp, x0, y0, sx, sy, rho)
    res, err = dblquad(F, 0, 1, lambda x: 0, lambda x: 1, **kwargs)
  return res

def TestIntegration():
  '''
  Compares the fast and slow ways of computing the flux. Reports the
  time each method took and the difference in the flux between the
  two methods.
  
  '''
  
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

if __name__ == '__main__':
  
  TestIntegration()
