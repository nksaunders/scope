import numpy as np
import matplotlib.pyplot as pl
import everest
from .skopemath import PSF, PLD
from random import randint
from astropy.io import fits
import pyfits
from everest import Transit
import k2plr
import os
from scipy.optimize import fmin_powell
from .skopemath import PSF, PLD

class PSFFit(object):

    def __init__(self, fpix, ferr, xpos, ypos):

        # initialize self variables
        self.nsrc = 2
        self.xtol = 0.0001
        self.ftol = 0.0001
        self.fpix = fpix
        self.ferr = ferr
        self.xpos = xpos
        self.ypos = ypos

    def Residuals(self, params):
        '''

        '''

        amp1,amp2,x01,x02,y01,y02,sx,sy,rho = params
        cadence = self.cadence

        # constrain parameter values
        if sx > 1 or sx < 0:
            return 1.0e30
        if sy > 1 or sy < 0:
            return 1.0e30

        if rho >= 1 or rho <= -1:
            return 1.0e30


        if ((3.5 - x01)**2 + (3.5 - y01)**2) > 3:
            return 1.0e30
        if ((3.5 - x02)**2 + (3.5 - y02)**2) > 5:
            return 1.0e30


        # Reject negative values for amplitude and position
        for elem in [amp1,amp2,x01,x02,y01,y02]:
            if elem < 0:
                return 1.0e30

        PSFfit = PSF(np.array([[amp1,amp2],[x01,x02],[y01,y02],[sx],[sy],[rho]]), self.ccd_args, self.xpos[self.cadence], self.ypos[self.cadence])[0]

        # sum squared difference between data and model
        PSFres = np.nansum((self.fpix[cadence] - PSFfit) ** 2)

        '''
        s_s = 1.
        sx0 = 0.5 + s_s * np.random.randn()
        sy0 = 0.5 + s_s * np.random.randn()
        PSFres += ((sx - sx0) / s_s)**2
        PSFres += ((sy - sy0) / s_s)**2
        PSFres += (rho / s_s)**2
        '''

        print("R = %.2e, x1 = %.2f, x2 = %.2f, y1 = %.2f, y2 = %.2f, sx = %.2f, sy = %.2f, rho = %.2f, a1 = %.2f, a2 = %.2f" % \
             (PSFres, x01, x02, y01, y02, sx, sy, rho, amp1, amp2))

        return PSFres

    def FindSolution(self, guess, ccd_args, cadence=100):
        '''
        Minimize residuals to find best PSF fit for the data
        '''
        self.guess = guess
        self.cadence = cadence
        self.ccd_args = ccd_args

        answer, chisq, _, iter, funcalls, warn = fmin_powell(self.Residuals, self.guess, xtol = self.xtol, ftol = self.ftol,
                                                             disp = False, full_output = True)

        self.bic = chisq + len(answer) * np.log(len(self.fpix))

        return answer
