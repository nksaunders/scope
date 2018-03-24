#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Simulate K2 target
------------------
Generate a simulated K2 target with motion vectors
from a real K2 observation of a given EPIC ID #.
Optionally includes synthetic transit and variability injection.
'''

import numpy as np
import matplotlib.pyplot as pl
import everest
from everest.mathutils import SavGol
from .skopemath import PSF, PLD
from . import fitting
from . import psffit as pf
import random
from random import randint
from astropy.io import fits
import pyfits
from everest import Transit
import k2plr
from k2plr.config import KPLR_ROOT
from everest.config import KEPPRF_DIR
import os
from tqdm import tqdm

class Target(object):
    '''
    A simulated K2 object with a forward model of the Kepler detector sensitivity variation
    '''

    def __init__(self, ID=205998445, custom_ccd=False, transit=False, variable=False, neighbor=False, ftpf=None):

        # initialize self variables
        self.ID = ID
        self.ftpf = ftpf
        self.custom_ccd = custom_ccd
        self.transit = transit
        self.variable = variable
        self.neighbor = neighbor
        self.targets = 1

    def GenerateLightCurve(self, mag, roll=1., background_level=0., ccd_args=[], neighbor_magdiff=1., photnoise_conversion=.000625, ncadences=1000, apsize=7):
        '''
        Creates a light curve for given detector, star, and transiting exoplanet parameters
        Motion from a real K2 target is applied to the PSF
        '''

        self.ncadences = ncadences
        self.t = np.linspace(0, 90, self.ncadences) # simulation lasts 90 days, with n cadences
        self.apsize = apsize # number of pixels to a side for aperture
        self.background_level = background_level

        # calculate PSF amplitude for given Kp Mag
        self.A = self.PSFAmplitude(mag)

        # read in K2 motion vectors for provided K2 target (EPIC ID #)
        if self.ftpf is None:

            # access target information
            client=k2plr.API()
            star=client.k2_star(self.ID)
            tpf=star.get_target_pixel_files(fetch = True)[0]
            ftpf=os.path.join(KPLR_ROOT, 'data', 'k2', 'target_pixel_files', '%d' % self.ID, tpf._filename)
        else:
            ftpf=self.ftpf
        with pyfits.open(ftpf) as f:

            # read motion vectors in x and y
            self.xpos=f[1].data['pos_corr1']
            self.ypos=f[1].data['pos_corr2']

        # throw out outliers
        for i in range(len(self.xpos)):
            if abs(self.xpos[i]) >= 50 or abs(self.ypos[i]) >= 50:
                self.xpos[i] = 0
                self.ypos[i] = 0
            if np.isnan(self.xpos[i]):
                self.xpos[i] = 0
            if np.isnan(self.ypos[i]):
                self.ypos[i] = 0

        # crop to desired length and multiply by roll coefficient
        self.xpos = self.xpos[0:self.ncadences] * roll
        self.ypos = self.ypos[0:self.ncadences] * roll

        # create self.inter-pixel sensitivity variation matrix
        # random normal distribution centered at 0.975
        self.inter = np.zeros((self.apsize, self.apsize))
        for i in range(self.apsize):
            for j in range(self.apsize):
                self.inter[i][j] = (0.975 + 0.01 * np.random.randn())

        # assign PSF model parameters to be passed into PixelFlux function
        if not self.custom_ccd:

            # cx,cy: intra-pixel variation polynomial coefficients in x,y
            self.cx = [1.0, 0.0, -0.3]
            self.cy = [1.0, 0.0, -0.3]

            # x0,y0: center of PSF, half of aperture size plus random deviation
            x0 = (self.apsize / 2.0) + 0.2 * np.random.randn()
            y0 = (self.apsize / 2.0) + 0.2 * np.random.randn()

            # sx,sy: standard deviation of Gaussian in x,y
            # rho: rotation angle between x and y dimensions of Gaussian
            sinx = np.linspace(0, 5*np.pi, self.ncadences) #hack
            sinvals = 2. + np.sin(sinx)
            sx = [0.5 + 0.05 * np.random.randn()]
            sy = [0.5 + 0.05 * np.random.randn()]
            rho = [0.05 + 0.02 * np.random.randn()]
            psf_args = [np.concatenate([[self.A], np.array([x0]), np.array([y0]), sx, sy, rho])]

        ccd_args = [self.cx, self.cy, self.apsize, self.A, background_level, self.inter, photnoise_conversion]
        self.ccd_args = ccd_args

        # initialize pixel flux light curve, target light curve, and isolated noise in each pixel
        self.fpix = np.zeros((self.ncadences, self.apsize, self.apsize))
        self.target = np.zeros((self.ncadences, self.apsize, self.apsize))
        self.ferr = np.zeros((self.ncadences, self.apsize, self.apsize))

        '''
        Here is where the light curves are created
        PSF function calculates flux in each pixel
        Iterate through cadences (c), and x and y dimensions on the detector (i,j)
        '''

        for c in tqdm(range(self.ncadences)):

            self.fpix[c], self.target[c], self.ferr[c] = PSF(psf_args, ccd_args, self.xpos[c], self.ypos[c], self.targets)

        # add transit and variability
        if self.transit:
            self.fpix, self.flux = self.AddTransit(self.fpix)
        if self.variable:
            self.fpix, self.flux = self.AddVariability(self.fpix)
        if self.neighbor:
            self.fpix, self.flux = self.AddNeighbor(self.fpix)

        if not self.transit and not self.variable:
            # create flux light curve
            self.flux = np.sum(self.fpix.reshape((self.ncadences), -1), axis=1)

        return self.fpix, self.flux, self.ferr

    def Detrend(self, fpix):
        '''
        Runs 2nd order PLD with a Gaussian Proccess on a given light curve
        '''

        # Set empty transit mask if no transit provided
        if not self.transit:
            self.trninds = np.array([])

        # Check background level and define aperture
        if self.background_level != 0:
            aperture = self.Aperture(fpix)
        else:
            aperture = np.ones((self.apsize, self.apsize))

        # Run 2nd order PLD with a Gaussian Process
        flux, rawflux = PLD(fpix, self.trninds, self.ferr, self.t, aperture)

        return flux, rawflux

    def PSFAmplitude(self, mag):
        '''
        Returns the amplitude of the PSF for a star of a given magnitude.
        '''

        # mag/flux relation constants
        a,b,c = 1.65e+07, 0.93, -7.35

        return a * np.exp(-b * (mag+c))


    def AddTransit(self, fpix, depth=.001, per=15, dur=.5, t0=5.):
        '''
        Injects a transit into light curve
        '''

        self.transit = True

        # Transit information
        self.depth=depth
        self.per=per # period (days)
        self.dur=dur # duration (days)
        self.t0=t0 # initial transit time (days)

        # Create transit light curve
        if self.depth == 0:
            self.trn = np.ones((self.ncadences))
        else:
            self.trn = Transit(self.t, t0=self.t0, per=self.per, dur=self.dur, depth=self.depth)

        # Define transit mask
        self.trninds = np.where(self.trn>1.0)
        self.M=lambda x: np.delete(x, self.trninds, axis=0)

        # Add transit to light curve
        self.fpix_trn = np.zeros((self.ncadences, self.apsize, self.apsize))
        for i,c in enumerate(fpix):
            self.fpix_trn[i] = c * self.trn[i]

        # Create flux light curve
        self.flux_trn = np.sum(self.fpix_trn.reshape((self.ncadences), -1), axis=1)

        return self.fpix_trn, self.flux_trn

    def AddVariability(self, fpix, var_amp=0.0005, freq=0.25, custom_variability=[]):
        '''
        Add a sinusoidal variability model to the given light curve.
        '''

        self.variable = True

        # Check for custom variability
        if len(custom_variability) != 0:
            V = custom_variability
        else:
            V = 1 + var_amp * np.sin(freq*self.t)

        # Add variability to light curve
        V_fpix = [f * V[i] for i,f in enumerate(fpix)]

        # Create flux light curve
        V_flux = np.sum(np.array(V_fpix).reshape((self.ncadences), -1), axis=1)

        return V_fpix, V_flux

    def AddNeighbor(self, fpix, magdiff=1., dist=2.5):
        '''
        Add a neighbor star with given difference in magnitude and distance at a randomized location
        '''

        # initialize arrays
        n_fpix = np.zeros((self.ncadences, self.apsize, self.apsize))
        neighbor = np.zeros((self.ncadences, self.apsize, self.apsize))
        n_ferr = np.zeros((self.ncadences, self.apsize, self.apsize))

        # set neighbor params
        x_offset = dist * np.random.randn()
        y_offset = np.sqrt(dist**2 - x_offset**2) * random.choice((-1, 1))
        nx0 = (self.apsize / 2.0) + x_offset
        ny0 = (self.apsize / 2.0) + y_offset
        sx = [0.5 + 0.05 * np.random.randn()]
        sy = [0.5 + 0.05 * np.random.randn()]
        rho = [0.05 + 0.02 * np.random.randn()]

        neighbor_args = [np.concatenate([[self.A], [nx0], [ny0], sx, sy, rho])]

        # calculate comparison factor for neighbor, based on provided difference in magnitude
        self.r = 10 ** (magdiff / 2.5)

        # create neighbor pixel-level light curve
        for c in tqdm(range(self.ncadences)):

            # iterate through cadences, calculate pixel flux values
            n_fpix[c], neighbor[c], n_ferr[c] = PSF(neighbor_args, self.ccd_args, self.xpos[c], self.ypos[c], self.targets)

            # divide by magdiff factor
            n_fpix[c] /= self.r
            neighbor[c] /= self.r

        # add neighbor to light curve
        fpix += n_fpix
        self.n_fpix = n_fpix

        # calculate flux light curve
        n_flux = np.sum(np.array(n_fpix).reshape((self.ncadences), -1), axis=1)

        self.neighbor = True
        self.targets += 1

        return fpix, n_flux

    def Aperture(self, fpix):
        '''
        Create an aperture including all pixels containing target flux
        '''

        aperture = np.zeros((self.ncadences, self.apsize, self.apsize))

        # Identify pixels with target flux for each cadence
        for c,f in enumerate(self.target):
            for i in range(self.apsize):
                for j in range(self.apsize):
                    if f[i][j] < 1.:
                        aperture[c][i][j] = 0
                    else:
                        aperture[c][i][j] = 1

        # Create single aperture
        finalap = np.zeros((self.apsize, self.apsize))

        # Sum apertures to weight pixels
        for i in range(self.apsize):
            for ap in aperture:
                finalap[i] += ap[i]

        # Normalize to 1
        finalap /= np.max(finalap)

        # Set excluded pixels to NaN
        for i in range(self.apsize):
            for j in range(self.apsize):
                if finalap[i][j] == 0:
                    finalap[i][j] = np.nan

        return finalap

    def DisplayDetector(self):
        '''
        Returns matrix for CCD pixel sensitivity
        '''

        # Define detector dimensions
        xdim = np.linspace(0, self.apsize, 100)
        ydim = np.linspace(0, self.apsize, 100)

        # Pixel resolution
        res = int(1000 / self.apsize)

        pixel_sens = np.zeros((res,res))

        # Calculate sensitivity function with detector parameters for individual pixel
        for i in range(res):
            for j in range(res):
                pixel_sens[i][j] = np.sum([c * (i-res/2) ** m for m, c in enumerate(self.cx)], axis = 0) + \
                np.sum([c * (j-res/2) ** m for m, c in enumerate(self.cy)], axis = 0)

        # Tile to create detector
        intra = np.tile(pixel_sens, (self.apsize, self.apsize))
        self.detector = np.zeros((res*self.apsize,res*self.apsize))

        # Multiply by inter-pixel sensitivity variables
        for i in range(self.apsize):
            for j in range(self.apsize):
                self.detector[i*res:(i+1)*res][j*res:(j+1)*res] = intra[i*res:(i+1)*res][j*res:(j+1)*res] * self.inter[i][j]

        # Display detector
        pl.imshow(self.detector, origin='lower', cmap='gray')
        pl.show()

        return self.detector


    def FitPSF(self, fpix):
        '''

        '''

        PF = fitting.PSFFit(fpix, self.ferr, self.xpos, self.ypos, self.ccd_args)

        A = 200000
        sx = 0.5
        sy = 0.5
        rho = 0.05

        guess = np.concatenate([[A], [self.apsize/2], [self.apsize/2], [sx], [sy], [rho]])

        if self.neighbor:

            A2 = A / self.r
            sx2 = sx
            sy2 = sy
            rho2 = rho

            guess_with_neighbor = [[A, A2], [self.apsize/2, self.apsize/2], [self.apsize/2, self.apsize/2], [sx, sx2], [sy, sy2], [rho, rho2]]

            guess = guess_with_neighbor

        ans_set = []
        print("Finding solutions...")
        for i in tqdm(range(self.ncadences)):
            ans_cadence = PF.FindSolution(guess, i, self.targets)
            ans_set.append(ans_cadence)

        import pdb; pdb.set_trace()
        print("Creating PSF...")
        fit_fpix = []
        for ind, ans in tqdm(enumerate(ans_set)):

            n = self.targets

            A = ans[0:n]
            x0 = ans[n:2*n]
            y0 = ans[2*n:3*n]
            sx = ans[3*n:4*n]
            sy = ans[4*n:5*n]
            rho = ans[5*n:6*n]

            cadence, _, _ = PSF(A[0],x0[0],y0[0],sx[0],sy[0],rho[0], self.ccd_args, self.xpos[ind], self.ypos[ind], targets) \
                            + PSF(A[1],x0[1],y0[1],sx[1],sy[1],rho[1], self.ccd_args, self.xpos[ind], self.ypos[ind], targets)
            fit_fpix.append(cadence)

        return fit_fpix

    def FindFit(self):

        self.fit = pf.PSFFit(self.fpix,self.ferr)

        amp = [345000.0,(352000.0 / 2)]
        x0 = [2.6,3.7]
        y0 = [2.3,4.2]
        sx = [.4]
        sy = [.6]
        rho = [0.01]
        background = [1000]
        index = 200
        guess = np.concatenate([amp,x0,y0,sx,sy,rho])
        answer = self.fit.FindSolution(guess, index=index)
        invariant_vals = np.zeros((len(answer)))
        self.n_fpix = np.zeros((len(self.fpix),5,5))
        self.subtracted_fpix = np.zeros((len(self.fpix),5,5))


        for i,v in enumerate(answer):
            if i == 0:
                invariant_vals[i] = 0
            elif i == 2:
                invariant_vals[i] = v - self.xpos[index]
            elif i == 4:
                invariant_vals[i] = v - self.ypos[index]
            else:
                invariant_vals[i] = v

        print("Subtracting neighbor...")
        for cadence in tqdm(range(len(self.fpix))):
            n_vals = np.zeros((len(invariant_vals)))
            for i,v in enumerate(invariant_vals):
                if i == 2:
                    n_vals[i] = v + self.xpos[cadence]
                elif i == 4:
                    n_vals[i] = v + self.ypos[cadence]
                else:
                    n_vals[i] = v

            neighbor_cad = self.fit.PSF(n_vals)
            self.n_fpix[cadence] = neighbor_cad
            self.subtracted_fpix[cadence] = self.fpix[cadence] - neighbor_cad


        self.answerfit = self.fit.PSF(answer)
        self.neighborfit = self.fit.PSF(invariant_vals)
        self.subtraction = self.answerfit - self.neighborfit
        self.residual = self.fpix[200] - self.answerfit
        self.subtracted_flux = self.aft.FirstOrderPLD(self.subtracted_fpix)[0]
        return self.subtraction
