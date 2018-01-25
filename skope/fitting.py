from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits
import pyfits
import os
from scipy.optimize import fmin_powell
from skopemath import PSF, PLD

class PSFFit(object):

    def __init__(self, fpix, ferr, xpos, ypos, ccd_args):

        self.xtol = 0.0001
        self.ftol = 0.0001
        self.fpix = fpix
        self.ferr = ferr
        self.index = 50
        self.ccd_args = ccd_args
        self.xpos = xpos
        self.ypos = ypos

    def Residuals(self, params):
        '''
        takes psf and ccd parameters and single index of fpix, ferr
        '''

        # Read in PSF arguments
        A, x0, y0, sx, sy, rho = params

        # priors
        if any(s > 1 or s < 0 for s in sx):
            return 1.0e30
        if any(s > 1 or s < 0 for s in sy):
            return 1.0e30

        if any(r >= 1 or r <= -1 for r in rho):
            return 1.0e30

        if ((2.5 - x0)**2 + (2.5 - y0)**2) > 4:
            return 1.0e30

        # Reject negative values for amplitude and position
        if any(a < 0 for a in A):
            return 1.0e30
        if x0 < 0:
            return 1.0e30
        if y0 < 0:
            return 1.0e30

        print(params)
        PSFfit = PSF(params, self.ccd_args, self.xpos[self.index], self.ypos[self.index])

        PSFres = np.nansum(((self.fpix[self.index] - PSFfit) / self.ferr[self.index]) ** 2)

        return PSFres


    def FindSolution(self, guess):
        '''
        minimize residuals to find best PSF fit for the data
        '''

        print(guess)
        answer, chisq, _, iter, funcalls, warn = fmin_powell(self.Residuals, guess, xtol = self.xtol, ftol = self.ftol,
                                                             disp = False, full_output = True)

        bic = chisq + len(answer) * np.log(len(fpix))

        return answer
