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
from matplotlib.widgets import Button
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
from datetime import datetime

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

        self.startTime = datetime.now()

    def GenerateLightCurve(self, mag, roll=1., background_level=0., ccd_args=[], neighbor_magdiff=1., photnoise_conversion=.000625, ncadences=1000, apsize=7):
        '''
        Creates a light curve for given detector, star, and transiting exoplanet parameters
        Motion from a real K2 target is applied to the PSF
        '''

        self.ncadences = ncadences
        self.t = np.linspace(0, 90, self.ncadences) # simulation lasts 90 days, with n cadences
        self.apsize = apsize # number of pixels to a side for aperture
        self.background_level = background_level
        self.aperture = np.ones((self.ncadences, self.apsize, self.apsize))

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
                self.inter[i][j] = (0.975 + 0.001 * np.random.randn())

        # assign PSF model parameters to be passed into PixelFlux function
        if not self.custom_ccd:

            # cx,cy: intra-pixel variation polynomial coefficients in x,y
            self.cx = [1.0, 0.0, -0.05]
            self.cy = [1.0, 0.0, -0.05]

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
            psf_args = np.concatenate([[self.A], np.array([x0]), np.array([y0]), sx, sy, rho])

        ccd_args = [self.cx, self.cy, self.apsize, background_level, self.inter, photnoise_conversion]
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

            self.fpix[c], self.target[c], self.ferr[c] = PSF(psf_args, ccd_args, self.xpos[c], self.ypos[c])

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
            self.aperture = self.Aperture(fpix)
        else:
            self.aperture = np.ones((self.apsize, self.apsize))

        # Run 2nd order PLD with a Gaussian Process
        flux, rawflux = PLD(fpix, self.trninds, self.ferr, self.t, self.aperture)

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
        self.depth = depth
        self.per = per # period (days)
        self.dur = dur # duration (days)
        self.t0 = t0 # initial transit time (days)

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

        neighbor_args = np.concatenate([[self.A], [nx0], [ny0], sx, sy, rho])

        # calculate comparison factor for neighbor, based on provided difference in magnitude
        self.r = 10 ** (magdiff / 2.5)

        # create neighbor pixel-level light curve
        for c in tqdm(range(self.ncadences)):

            # iterate through cadences, calculate pixel flux values
            n_fpix[c], neighbor[c], n_ferr[c] = PSF(neighbor_args, self.ccd_args, self.xpos[c], self.ypos[c])

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


    def FindFit(self, neighbor_coords=[]):

        # initialize fit
        self.fit = pf.PSFFit(self.fpix, self.ferr, self.xpos, self.ypos, self.ccd_args, self.apsize)

        pl.imshow(self.fpix[0], origin='lower', cmap='viridis')
        nx = input("Enter x pixel coordinate: ")
        ny = input("Enter y pixel coordinate: ")
        neighbor_coords = [int(nx), int(ny)]

        # set guess
        # *** HARDCODED FOR DEBUGGING ***
        amp = [250000.0, (250000.0 / self.r)]
        x0 = [self.apsize/2, int(neighbor_coords[0])]
        y0 = [self.apsize/2, int(neighbor_coords[1])]
        sx = [.5]
        sy = [.5]
        rho = [0.01]

        cadence = 0
        guess = np.concatenate([amp,x0,y0,sx,sy,rho])

        # run
        answer = self.fit.FindSolution(guess, self.ccd_args, cadence=cadence)

        invariant_vals = np.zeros((len(answer)))
        self.n_fpix = np.zeros((len(self.fpix),self.apsize,self.apsize))
        self.subtracted_fpix = np.zeros((len(self.fpix),self.apsize,self.apsize))

        # PSF fit independent of motion vectors
        for i,v in enumerate(answer):
            if i == 2:
                invariant_vals[i] = v - self.xpos[cadence]
            elif i == 3:
                invariant_vals[i] = v - self.ypos[cadence]
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

            neighbor_cad = self.fit.CalculatePSF(n_vals, cadence, neighbor=True)

            self.n_fpix[cadence] = neighbor_cad
            self.subtracted_fpix[cadence] = self.fpix[cadence] - neighbor_cad

        self.answerfit = self.fit.CalculatePSF(answer, cadence)
        self.neighborfit = self.fit.CalculatePSF(invariant_vals, cadence, neighbor=True)
        self.subtraction = self.answerfit - self.neighborfit
        self.residual = self.fpix[cadence] - self.answerfit
        # self.subtracted_flux = PLD(self.subtracted_fpix, self.trninds, self.ferr, self.t, self.aperture)[0]

        return self.subtraction

    def interactive(self):

        fig = pl.figure()
        ax = fig.add_subplot(111)
        mask = np.ones((self.apsize,self.apsize))

        ax.imshow(self.fpix[0], origin='lower', cmap='viridis')
        ax.annotate('Select center of neighbor PSF.',
                    xy = (0.05, 0.05),xycoords='axes fraction',
                    color='w', fontsize=12)

        coords = []
        def onclick(event):

            x, y = event.inaxes.transData.inverted().transform((event.x, event.y))
            if mask[int(y+.5)][int(x+.5)] == 0:
                mask[int(y+.5)][int(x+.5)] = 1
                pl.close()
            else:
                mask[int(y+.5)][int(x+.5)] = 0
                print('pixel loc=(%d, %d)' % (x+.5, y+.5))

                coords.append(int(x+.5))
                coords.append(int(y+.5))

                ax.clear()
                ax.imshow(self.fpix[0], origin='lower', cmap='viridis')
                ax.imshow(mask, origin='lower', alpha=0.5, cmap='Purples')
                ax.annotate('Click again to confirm.',
                            xy = (0.05, 0.05),xycoords='axes fraction',
                            color='w', fontsize=12)

                fig.canvas.draw()


        fig.canvas.mpl_connect('button_press_event', onclick)

        return coords


    def Plot(self):

        fig, ax = pl.subplots(1,3, sharey=True)
        fig.set_size_inches(17,5)

        meanfpix = np.mean(self.fpix,axis=0)
        ax[0].imshow(self.fpix[0],interpolation='nearest',origin='lower',cmap='viridis',vmin=np.min(self.answerfit),vmax=np.max(self.answerfit));
        ax[1].imshow(self.answerfit,interpolation='nearest',origin='lower',cmap='viridis',vmin=np.min(self.answerfit),vmax=np.max(self.answerfit));
        ax[2].imshow(self.subtraction,interpolation='nearest',origin='lower',cmap='viridis',vmin=np.min(self.answerfit),vmax=np.max(self.answerfit));
        ax[0].set_title('Data');
        ax[1].set_title('Model');
        ax[2].set_title('Neighbor Subtraction');
        ax[1].annotate(r'$\mathrm{Max\ Residual\ Percent}: %.4f $' % (np.max(np.abs(self.residual))/np.max(self.fpix[0])),
                        xy = (0.05, 0.05),xycoords='axes fraction',
                        color='w', fontsize=12);


        unsub_flux = PLD(self.fpix, self.trninds, self.ferr, self.t, self.aperture)[0]
        fig, ax = pl.subplots(2,1)
        # ns_depth = self.aft.RecoverTransit(self.subtracted_flux)
        ax[0].plot(self.t,np.mean(unsub_flux)*self.trn,'r')
        ax[0].plot(self.t,unsub_flux,'k.')
        ax[0].set_title('No Subtraction, 1st Order PLD')
        # ax[1].plot(self.t,np.mean(self.subtracted_flux)*self.trn,'r')
        # ax[1].plot(self.t,self.subtracted_flux,'k.')
        ax[1].set_title('Neighbor Subtraction, 1st Order PLD')

        '''
        ax[0].annotate(r'$\mathrm{Recovered\ Depth}: %.4f$' % (self.aft.RecoverTransit(unsub_flux)),
                        xy = (0.05, 0.05),xycoords='axes fraction',
                        color='k', fontsize=12);

        ax[1].annotate(r'$\mathrm{Recovered\ Depth}: %.4f$' % (ns_depth),
                        xy = (0.05, 0.05),xycoords='axes fraction',
                        color='k', fontsize=12);
        '''

        print("Run time:")
        print(datetime.now() - self.startTime)
        # print("RTD: %.4f,   Subtracted RTD: %.4f" % (self.aft.RecoverTransit(unsub_flux),ns_depth))
        pl.show()
