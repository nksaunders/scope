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
        if sx > 1 or sx < 0:
            return 1.0e30
        if sy > 1 or sy < 0:
            return 1.0e30

        if rho >= 1 or rho <= -1:
            return 1.0e30

        if ((2.5 - x0)**2 + (2.5 - y0)**2) > 4:
            return 1.0e30


        # Reject negative values for amplitude and position
        for elem in [A, x0, y0]:
            if elem < 0:
                return 1.0e30

        PSFfit = PSF(params, self.ccd_args, self.xpos[self.index], self.ypos[self.index])

        PSFres = np.nansum(((self.fpix[self.index] - PSFfit) / self.ferr[self.index]) ** 2)

        return PSFres


    def FindSolution(self, guess):
        '''
        minimize residuals to find best PSF fit for the data
        '''

        answer, chisq, _, iter, funcalls, warn = fmin_powell(self.Residuals, guess, xtol = self.xtol, ftol = self.ftol,
                                                             disp = False, full_output = True)

        bic = chisq + len(answer) * np.log(len(self.fpix))

        return answer
