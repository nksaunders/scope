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

class PSFFit(object):

    def __init__(self, fpix, ferr):

        # initialize self variables
        self.nsrc = 2
        self.xtol = 0.0001
        self.ftol = 0.0001
        self.fpix = fpix
        self.ferr = ferr

    def PSF(self, params):
        '''
        Create PSF model from params
        '''
        
        amp1,amp2,x01,x02,y01,y02,sx,sy,rho = params

        cx_1 = [1.,0.,0.]
        cx_2 = [1.,0.,0.]
        cy_1 = [1.,0.,0.]
        cy_2 = [1.,0.,0.]
        '''
        sx = 0.5
        sy = 0.5
        rho = 0.0
        '''


        model = np.zeros((5,5))
        for i in range(5):
            for j in range(5):
                model[i][j] = PixelFlux(cx_1, cy_1, [amp1], [x01-i], [y01-j], [sx], [sy], [rho]) + \
                              PixelFlux(cx_2, cy_2, [amp2], [x02-i], [y02-j], [sx], [sy], [rho])
        return model

    def Residuals(self, params):
        '''

        '''

        amp1,amp2,x01,x02,y01,y02,sx,sy,rho,background = params
        index = self.index

        # constrain parameter values
        if sx > 1 or sx < 0:
            return 1.0e30
        if sy > 1 or sy < 0:
            return 1.0e30

        if rho >= 1 or rho <= -1:
            return 1.0e30

        if ((2.5 - x01)**2 + (2.5 - y01)**2) > 4:
            return 1.0e30
        if ((4. - x02)**2 + (4. - y02)**2) > 4:
            return 1.0e30

        # Reject negative values for amplitude and position
        for elem in [amp1,amp2,x01,x02,y01,y02]:
            if elem < 0:
                return 1.0e30

        PSFfit = self.PSF(params)

        # sum squared difference between data and model
        PSFres = np.nansum(((self.fpix[index] - PSFfit) / self.ferr[index]) ** 2)

        '''
        s_s = 1.
        sx0 = 0.5 + s_s * np.random.randn()
        sy0 = 0.5 + s_s * np.random.randn()
        PSFres += ((sx - sx0) / s_s)**2
        PSFres += ((sy - sy0) / s_s)**2
        PSFres += (rho / s_s)**2
        '''

        print("R = %.2e, x1 = %.2f, x2 = %.2f, y1 = %.2f, y2 = %.2f, sx = %.2f, sy = %.2f, rho = %.2f, a1 = %.2f, a2 = %.2f, b = %.2e" % \
             (PSFres, x01, x02, y01, y02, sx, sy, rho, amp1, amp2,background))

        return PSFres

    def FindSolution(self, guess, index=100):
        '''
        Minimize residuals to find best PSF fit for the data
        '''
        self.guess = guess
        self.index = index

        answer, chisq, _, iter, funcalls, warn = fmin_powell(self.Residuals, self.guess, xtol = self.xtol, ftol = self.ftol,
                                                             disp = False, full_output = True)

        self.bic = chisq + len(answer) * np.log(len(self.fpix))

        return answer
