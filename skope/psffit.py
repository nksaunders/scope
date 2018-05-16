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

import scipy
import scipy.misc
scipy.derivative = scipy.misc.derivative
import pymc
from astroML.plotting.mcmc import plot_mcmc

class PSFFit(object):

    def __init__(self, fpix, ferr, xpos, ypos, ccd_args, apsize):

        # initialize self variables
        self.nsrc = 2
        self.xtol = 0.0001
        self.ftol = 0.0001
        self.fpix = fpix
        self.ferr = ferr
        self.xpos = xpos
        self.ypos = ypos
        self.ccd_args = ccd_args
        self.apsize = apsize

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

        if rho >= 0.5 or rho <= -0.5:
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


    def CalculatePSF(self, params, cadence, neighbor=False):
        '''

        '''

        amp1,amp2,x01,x02,y01,y02,sx,sy,rho = params

        if neighbor:
            amp1 = 0

        computed_psf = PSF(np.array([[amp1,amp2],[x01,x02],[y01,y02],[sx],[sy],[rho]]), self.ccd_args, self.xpos[cadence], self.ypos[cadence])[0]

        return computed_psf


    def MCMC_results(self, cadence=0):

        apsize = self.apsize

        # Set up MCMC sampling
        amp1 = pymc.Uniform('amp1', 200000., 300000., value=250000.)
        amp2 = pymc.Uniform('amp2', 50000., 200000., value=100000.)
        x01 = pymc.Uniform('x01', 0, apsize, value=(apsize / 2))
        x02 = pymc.Uniform('x02', 0, apsize, value=(apsize / 2))
        y01 = pymc.Uniform('y01', 0, apsize, value=(apsize / 2))
        y02 = pymc.Uniform('y02', 0, apsize, value=(apsize / 2))
        sx = pymc.Uniform('sx', 0, 1, value=0.5)
        sy = pymc.Uniform('sy', 0, 1, value=0.5)
        rho = pymc.Uniform('rho', -1, 1, value=0.01)

        @pymc.deterministic
        def psf_model(amp1=amp1, amp2=amp2, x01=x01, x02=x02, y01=y01, y02=y02, sx=sx, sy=sy, rho=rho):
            return PSF(np.array([[amp1,amp2],[x01,x02],[y01,y02],[sx],[sy],[rho]]), self.ccd_args, self.xpos[cadence], self.ypos[cadence])[0]

        y = pymc.Normal('y', mu=psf_model, tau=self.ferr[cadence], observed=True, value=self.fpix[cadence])

        model = dict(amp1=amp1, amp2=amp2, x01=x01, x02=x02, y01=y01, y02=y02, sx=sx, sy=sy, rho=rho)


        # Run the MCMC sampling
        def compute_MCMC_results(niter=250, burn=40):
            S = pymc.MCMC(model)
            S.sample(iter=niter, burn=burn)
            traces = [S.trace(s)[:] for s in ['amp1', 'amp2', 'x01', 'x02', 'y01', 'y02', 'sx', 'sy', 'rho']]

            M = pymc.MAP(model)
            M.fit()
            fit_vals = (M.amp1.value, M.amp2.value, M.x01.value, M.x02.value, M.y01.value, M.y02.value, M.sx.value, M.sy.value, M.rho.value)

            return traces, fit_vals

        traces, fit_vals = compute_MCMC_results()

        labels = ['amp1', 'amp2', 'x01', 'x02', 'y01', 'y02', 'sx', 'sy', 'rho']
        fig = pl.figure(figsize=(10, 10))

        plot_mcmc(traces, labels=labels, fig=fig, bins=30, colors='k')

        return fit_vals
