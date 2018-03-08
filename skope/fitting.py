from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits
import pyfits
import os
from scipy.optimize import fmin_powell
from .skopemath import PSF, PLD

class PSFFit(object):

    def __init__(self, fpix, ferr, xpos, ypos, ccd_args):

        self.xtol = 0.0001
        self.ftol = 0.0001
        self.fpix = fpix
        self.ferr = ferr
        # self.index = 50
        self.ccd_args = ccd_args
        self.xpos = xpos
        self.ypos = ypos

    def Residuals(self, params):
        '''
        takes psf and ccd parameters and single index of fpix, ferr
        '''

        # Read in PSF arguments
        n = self.targets

        A = params[0:n]
        x0 = params[n:2*n]
        y0 = params[2*n:3*n]
        sx = params[3*n:4*n]
        sy = params[4*n:5*n]
        rho = params[5*n:6*n]

        # priors
        if sx.any() > 1 or sx.any() < 0:
            return 1.0e30
        if sy.any() > 1 or sy.any() < 0:
            return 1.0e30

        if rho.any() >= 1 or rho.any() <= -1:
            return 1.0e30

        if ((2.5 - x0[0])**2 + (2.5 - y0[0])**2) > 4:
            return 1.0e30


            # Reject negative values for amplitude and position
            for elem in [A, x0, y0]:
                if elem.any() < 0:
                    return 1.0e30

        PSFfit = PSF([[A],[x0],[y0],[sx],[sy],[rho]], self.ccd_args, self.xpos[self.index], self.ypos[self.index], targets)

        PSFres = np.nansum(((self.fpix[self.index] - PSFfit) / self.ferr[self.index]) ** 2)

        return PSFres


    def FindSolution(self, guess, index, targets):
        '''
        minimize residuals to find best PSF fit for the data
        '''

        self.index = index
        self.targets = targets

        import pdb; pdb.set_trace()

        answer, chisq, _, iter, funcalls, warn = fmin_powell(self.Residuals, guess, xtol = self.xtol, ftol = self.ftol,
                                                             disp = False, full_output = True)

        bic = chisq + len(answer) * np.log(len(self.fpix))

        return answer
