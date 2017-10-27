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
from .intrapix import PixelFlux
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

    def __init__(self, ID, transit = False, custom_ccd = False, depth = .01, per = 15, dur = .5, t0 = 5., variability = False, ncadences = 1000, apsize = 7, ftpf = None):
        '''

        '''

        # initialize self variables
        self.ID = ID
        self.ftpf = ftpf
        self.ncadences = ncadences
        self.custom_ccd = custom_ccd

        # transit information
        self.transit = transit
        self.depth = depth
        self.per = per # period (days)
        self.dur = dur # duration (days)
        self.t0 = t0 # initial transit time (days)

        # set aperture size (number of pixels to a side)
        self.apsize = apsize


    def GeneratePSF(self, mag, roll = 1., background_level = 0., neighbor = False, ccd_args = [], neighbor_magdiff = 1., photnoise_conversion = .000625):
        '''

        '''

        # calculate PSF amplitude for given Kp Mag
        self.A = self.PSFAmplitude(mag)

        # read in K2 motion vectors for provided K2 target (EPIC ID #)
        if self.ftpf is None:

            # access target information
            client = k2plr.API()
            star = client.k2_star(self.ID)
            tpf = star.get_target_pixel_files(fetch = True)[0]
            ftpf = os.path.join(KPLR_ROOT, 'data', 'k2', 'target_pixel_files', '%d' % self.ID, tpf._filename)
        else:
            ftpf = self.ftpf
        with pyfits.open(ftpf) as f:

            # read motion vectors in x and y
            self.xpos = f[1].data['pos_corr1']
            self.ypos = f[1].data['pos_corr2']

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
        self.xpos = self.xpos[1000:1000+self.ncadences] * roll
        self.ypos = self.ypos[1000:1000+self.ncadences] * roll

        # create inter-pixel sensitivity variation matrix
        # random normal distribution centered at 0.975
        inter = np.zeros((self.apsize, self.apsize))
        for i in range(self.apsize):
            for j in range(self.apsize):
                inter[i][j] = (0.975 + 0.01 * np.random.randn())

        # create transit light curve and transit mask
        if self.transit:
            self.trn = self.AddTransit() # DEBUG: transit vals here
            self.trnvals = np.where(self.trn < 1)
        else:
            # if no transit included: trn is a flat normalized light curve, mask does nothing
            self.trn = np.ones((self.ncadences))
            self.trnvals = []
        self.M = lambda x: np.delete(x, self.naninds, axis = 0)

        # assign PSF model parameters to be passed into PixelFlux function
        if not self.custom_ccd:

            # cx,cy: intra-pixel variation polynomial coefficients in x,y
            cx = [1.0, 0.0, -0.3]
            cy = [1.0, 0.0, -0.3]

            # x0,y0: center of PSF, half of aperture size plus random deviation
            x0 = (self.apsize / 2.0) + 0.2 * np.random.randn()
            y0 = (self.apsize / 2.0) + 0.2 * np.random.randn()

            # sx,sy: standard deviation of Gaussian in x,y
            # rho: rotation angle between x and y dimensions of Gaussian
            sx = [0.5 + 0.05 * np.random.randn()]
            sy = [0.5 + 0.05 * np.random.randn()]
            rho = [0.05 + 0.02 * np.random.randn()]

        else:
            cx, cy, x0, y0, sx, sy, rho = ccd_args

        # calculate comparison factor for neighbor, based on provided difference in magnitude
        r = 10 ** (neighbor_magdiff / 2.5)

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
            for i in range(self.apsize):
                for j in range(self.apsize):

                    # contribution to pixel from target
                    target_pixval = self.trn[c] * PixelFlux(cx, cy, [self.A], [x0-i+self.xpos[c]], [y0-j+self.ypos[c]], sx, sy, rho)
                    self.target[c][i][j] = target_pixval

                    # add neighbor contribution to pixel flux value
                    if neighbor:
                        pixval = target_pixval+(1/r)*PixelFlux(cx, cy, [self.A], [neighborcoords[0]-i+self.xpos[c]], [neighborcoords[1]-j+self.ypos[c]], sx, sy, rho)
                        self.fpix[c][i][j] = pixval
                    else:
                        self.fpix[c][i][j] = target_pixval

                    # add background noise
                    noise = background_level * np.random.randn()
                    self.fpix[c][i][j] += noise
                    self.target[c][i][j] += noise

                    # ensure positive
                    while self.fpix[c][i][j] < 0:
                        self.fpix[c][i][j] = noise
                        self.target[c][i][j] = noise

                    # add photon noise
                    self.ferr[c][i][j] = np.sqrt(np.abs(self.fpix[c][i][j]) * photnoise_conversion)
                    randnum = np.random.randn()
                    self.fpix[c][i][j] += self.ferr[c][i][j] * randnum
                    self.target[c][i][j] += self.ferr[c][i][j] * randnum

                # multiply each cadence by inter-pixel sensitivity variation
                self.fpix[c] * inter
                self.target[c] * inter

        return self.fpix, self.target, self.ferr


    def PSFAmplitude(self, mag):
        '''
        Returns the amplitude of the PSF for a star of a given magnitude.
        '''
        x = mag
        a,b,c = 1.65e+07, 0.93, -7.35

        return a * np.exp(-b * (x+c))


    def AddTransit(self):
        '''

        '''

        # simulation lasts 90 days, with n cadences
        t = np.linspace(0, 90, self.ncadences)

        if self.depth == 0:
            trn = np.ones((self.ncadences))
        else:
            trn = Transit(t, t0 = self.t0, per = self.per, dur = self.dur, depth = self.depth)

        return trn


    def AddVariability(self):
        '''

        '''

        pass
