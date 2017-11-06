#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Simulate K2 target
------------------
Generate a simulated K2 target with motion vectors
from a real K2 observation of a given EPIC ID #.
Optionally includes synthetic transit injection.
!! (include sinusoidal stellar variability functionality)
'''

import numpy as np
import matplotlib.pyplot as pl
import everest
from everest.math import SavGol
from .skopemath import PSF, PLD
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

    def __init__(self, ID=205998445, custom_ccd=False, transit=False, variable=False, ftpf=None):
        '''

        '''

        # initialize self variables
        self.ID = ID
        self.ftpf = ftpf
        self.custom_ccd = custom_ccd
        self.transit = transit
        self.variable = variable

    def GenerateLightCurve(self, mag, roll=1., background_level=0., neighbor=False, ccd_args=[], neighbor_magdiff=1., photnoise_conversion=.000625, ncadences=1000, apsize=7):
        '''

        '''

        self.ncadences = ncadences
        self.t = np.linspace(0, 90, self.ncadences) # simulation lasts 90 days, with n cadences
        self.apsize = apsize # number of pixels to a side for aperture

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
            sx = [0.5 + 0.05 * np.random.randn()]
            sy = [0.5 + 0.05 * np.random.randn()]
            rho = [0.05 + 0.02 * np.random.randn()]

            ccd_args = [self.cx, self.cy, [self.A], x0, y0, sx, sy, rho]

        # calculate comparison factor for neighbor, based on provided difference in magnitude
        r = 10 ** (neighbor_magdiff / 2.5)

        psfargs = [self.apsize, self.A, background_level, self.inter, photnoise_conversion]

        # initialize pixel flux light curve, target light curve, and isolated noise in each pixel
        self.fpix = np.zeros((self.ncadences, self.apsize, self.apsize))
        self.target = np.zeros((self.ncadences, self.apsize, self.apsize))
        self.ferr = np.zeros((self.ncadences, self.apsize, self.apsize))

        '''
        here is where the light curves are created
        calculates flux in each pixel
        iterate through cadences (c), and x and y dimensions on the detector (i,j)
        '''

        for c in tqdm(range(self.ncadences)):
            self.fpix[c], self.ferr[c] = PSF(ccd_args, psfargs, self.xpos[c], self.ypos[c])

        # add transit and variability
        if self.transit:
            self.fpix, self.flux = self.AddTransit(self.fpix)

        if self.variable:
            self.fpix, self.flux = self.AddVariability(self.fpix)
        else:
            # create flux light curve
            self.flux = np.sum(self.fpix.reshape((self.ncadences),-1),axis=1)

        return self.fpix, self.flux, self.ferr

    def Detrend(self,fpix):
        '''

        '''

        if not self.transit:
            self.trninds = np.array([])

        return PLD(fpix,self.trninds)

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

        # transit information
        self.depth=depth
        self.per=per # period (days)
        self.dur=dur # duration (days)
        self.t0=t0 # initial transit time (days)

        # add transit to light curve
        if self.depth == 0:
            self.trn = np.ones((self.ncadences))
        else:
            self.trn = Transit(self.t, t0=self.t0, per=self.per, dur=self.dur, depth=self.depth)

        # define transit mask
        self.trninds = np.where(self.trn>1.0)
        self.M=lambda x: np.delete(x, self.trninds, axis=0)

        self.fpix_trn = np.zeros((self.ncadences, self.apsize, self.apsize))
        for i,c in enumerate(fpix):
            self.fpix_trn[i] = c * self.trn[i]

        self.flux_trn = np.sum(self.fpix_trn.reshape((self.ncadences), -1), axis=1)
        self.transit = True

        return self.fpix_trn, self.flux_trn

    def AddVariability(self, fpix, var_amp=0.0005, freq=0.25):
        '''
        Add a sinusoidal variability model to the given light curve.
        '''

        V = 1 + var_amp * np.sin(freq*self.t)
        V_fpix = [f * V[i] for i,f in enumerate(fpix)]

        V_flux = np.sum(np.array(V_fpix).reshape((self.ncadences), -1), axis=1)

        return V_fpix, V_flux

    def DisplayDetector(self):
        '''
        Returns matrix for CCD pixel sensitivity
        '''

        xdim = np.linspace(0, self.apsize, 100)
        ydim = np.linspace(0, self.apsize, 100)

        res = int(1000 / self.apsize)

        pixel_sens = np.zeros((res,res))

        for i in range(res):
            for j in range(res):
                pixel_sens[i][j] = np.sum([c * (i-res/2) ** m for m, c in enumerate(self.cx)], axis = 0) + \
                np.sum([c * (j-res/2) ** m for m, c in enumerate(self.cy)], axis = 0)

        intra = np.tile(pixel_sens, (self.apsize, self.apsize))
        self.detector = np.zeros((res*self.apsize,res*self.apsize))

        for i in range(self.apsize):
            for j in range(self.apsize):
                self.detector[i*res:(i+1)*res][j*res:(j+1)*res] = intra[i*res:(i+1)*res][j*res:(j+1)*res] * self.inter[i][j]

        pl.imshow(self.detector, origin='lower', cmap='gray')
        pl.show()

        return self.detector
