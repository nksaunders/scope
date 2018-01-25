import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits
import pyfits
import os
from scipy.optimize import fmin_powell
from .skopemath import PSF, PLD

class PSFFit(object):

    def __init__(self, fpix, ferr):

        self.xtol = 0.0001
        self.ytol = 0.0001
        self.fpix = fpix
        self.ferr = psferr


    def Residuals(fpix, ferr, psf_args, ccd_args, xpos, ypos):
        '''
        takes psf and ccd parameters and single index of fpix, ferr
        '''

        # Read in detector and PSF arguments
        cx, cy, A, x0, y0, sx, sy, rho = psf_args
        apsize, A, background_level, inter, photnoise_conversion = ccd_args

        # priors
        if sx > 1 or sx < 0:
            return 1.0e30
        if sy > 1 or sy < 0:
            return 1.0e30

        if rho >= 1 or rho <= -1:
            return 1.0e30

        if ((2.5 - x0)**2 + (2.5 - y0)**2) > 4:
            return 1.0e30
        if ((4. - x0)**2 + (4. - y0)**2) > 4:
            return 1.0e30

        # Reject negative values for amplitude and position
        for elem in [A,x0,x0,y0,y0]:
            if elem < 0:
                return 1.0e30

        PSFfit = PSF(psf_args, ccd_args, xpos, ypos)

        PSFres = np.nansum(((fpix - PSFfit) / ferr) ** 2)

        return PSFres


    def FindSolution(guess):
        '''
        minimize residuals to find best PSF fit for the data
        '''

        answer, chisq, _, iter, funcalls, warn = fmin_powell(Residuals, guess, xtol = 0.0001, ftol = 0.0001,
                                                             disp = False, full_output = True)

        bic = chisq + len(answer) * np.log(len(fpix))

        return answer
