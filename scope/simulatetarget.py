#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Simulate Target
---------------
Generate a forward model of a telescope detector with sensitivity variation,
and simulate stellar targets with motion relative to the CCD.
'''

import numpy as np
import matplotlib.pyplot as plt
import random
from random import randint
import os
from tqdm import tqdm
import warnings

from astropy import units as u
from astropy.constants import G
from astropy.io import fits

import everest
from everest.mathutils import SavGol
from everest.config import KEPPRF_DIR
from everest.missions.k2 import CDPP

import lightkurve as lk

from scipy.ndimage import zoom
import starry
import warnings

from .scopemath import PSF, PLD, _calculate_PSF_amplitude
from .utils import ScopeError, ScopeWarning, _interpolate_nans
from .transit import TransitModel

__all__ = ['Target', 'generate_target']

class Target(object):
    """A simulated stellar object with a forward model of a telescope detector's sensitivity variation"""

    def __init__(self, fpix, flux, ferr, target, t, mag=12., roll=1., neighbor_magdiff=1.,
                 ncadences=1000, apsize=7, transit=False, variable=False, neighbor=False,
                 ccd_args=[], psf_args=[], xpos=None, ypos=None):

        # initialize self variables
        self.targets = 1
        self.apsize = apsize
        self.ncadences = ncadences
        self.neighbor_magdiff = neighbor_magdiff
        self.mag = mag
        self.roll = roll
        self.ccd_args = ccd_args
        self.psf_args = psf_args
        self.xpos = xpos
        self.ypos = ypos

        self.t = t
        self.fpix = fpix
        self.flux = flux
        self.ferr = ferr
        self.target = target

    @property
    def time(self):
        return self.t

    @property
    def targetpixelfile(self):
        return self.fpix

    @property
    def lightcurve(self):
        return self.flux

    @property
    def error(self):
        return self.ferr

    @property
    def target_flux(self):
        return self.target

    def detrend(self, fpix=[]):
        """
        Runs 2nd order PLD with a Gaussian Proccess on a given light curve.

        Parameters
        ----------
        `fpix` :
            Pixel-level light curve of dimemsions (apsize, apsize, ncadences). Automatically set to fpix
            generated in GenerateLightCurve() unless a different light curve is passed.
        """

        # check if fpix light curve was passed in
        if len(fpix) == 0:
            fpix = self.fpix

        # Set empty transit mask if no transit provided
        if not self.transit:
            self.trninds = np.array([])

        # define aperture
        self.aperture = self.create_aperture(fpix)

        # Run 2nd order PLD with a Gaussian Process
        self.flux, self.rawflux = PLD(fpix, self.ferr, self.trninds, self.t, self.aperture)

        self.detrended_cdpp = self.estimate_CDPP(self.flux)
        self.raw_cdpp = self.estimate_CDPP(self.rawflux)

        return self

    def add_transit(self, fpix=[], rprs=.01, period=15., t0=5., i=90, ecc=0, m_star=1.):
        """
        Injects a transit into light curve.

        Parameters
        ----------
        `fpix` :
            Pixel-level light curve of dimemsions (apsize, apsize, ncadences). Automatically set to
            fpix generated in GenerateLightCurve() unless a different light curve is passed.
        `rprs` :
            R_planet / R_star. Ratio of the planet's radius to the star's radius.
        `period` :
            Period of transit in days.
        `t0` :
            Initial transit time in days.
        """

        # Create a starry transit model
        model = TransitModel(self.t)
        self.transit_signal = model.create_starry_model(rprs=.01, period=15., t0=5.,
                                                        i=90, ecc=0., m_star=1.)

        # Define transit mask
        M = model.create_transit_mask(self.transit_signal)

        self.fpix, self.flux, self.ferr, self.target = calculate_pixel_values(ncadences=self.ncadences, apsize=self.apsize,
                                                                              psf_args=self.psf_args, ccd_args=self.ccd_args,
                                                                              xpos=self.xpos, ypos=self.ypos, signal=self.transit_signal)

        return self

    def add_variability(self, fpix=[], var_amp=0.0005, freq=0.25, custom_variability=[]):
        """
        Add a sinusoidal variability model to the given light curve.

        Parameters
        ----------
        `fpix` :
            Pixel-level light curve of dimemsions (apsize, apsize, ncadences). Automatically
            set to fpix generated in GenerateLightCurve() unless a different light curve is passed.
        `var_amp` :
            Amplitude of sin wave, which is multiplied by the light curve.
        `freq` :
            Frequency of sin wave in days.
        `custom_variability` :
            A custom 1-dimensional array of length ncadences can be passed into the AddVariability()
            function, which will be multiplied by the light curve.
        """

        # check if fpix light curve was passed in
        if len(fpix) == 0:
            fpix = self.fpix

        self.variable = True

        # Check for custom variability
        if len(custom_variability) != 0:
            V = custom_variability
        else:
            V = 1 + var_amp * np.sin(freq*self.t)

        # Add variability to light curve
        self.fpix, self.flux, self.ferr, self.target = calculate_pixel_values(ncadences=self.ncadences, apsize=self.apsize,
                                                                              psf_args=self.psf_args, ccd_args=self.ccd_args,
                                                                              xpos=self.xpos, ypos=self.ypos, signal=V)

        return self

    def add_neighbor(self, fpix=[], magdiff=1., dist=1.7):
        """
        Add a neighbor star with given difference in magnitude and distance at a
        randomized location.

        Parameters
        ----------
        `fpix` :
            Pixel-level light curve of dimemsions (apsize, apsize, ncadences). Automatically
            set to fpix generated in GenerateLightCurve() unless a different light curve is passed.
        `magdiff` :
            Difference in stellar magnitude between target and neighbor. Positive magdiff
            corresponds to higher values for the neighbor star's magnitude.
        `dist` :
            Distance (in pixels) between cetroid position of target and neighbor. The (x, y)
            coordinates of the neighbor are chosen arbitrarily to result in the given distance.
        """

        if len(fpix) == 0:
            fpix = self.fpix

        # initialize arrays
        n_fpix = np.zeros((self.ncadences, self.apsize, self.apsize))
        neighbor = np.zeros((self.ncadences, self.apsize, self.apsize))
        n_ferr = np.zeros((self.ncadences, self.apsize, self.apsize))

        # set neighbor params
        x_offset = dist * np.random.randn()
        y_offset = np.sqrt(np.abs(dist**2 - x_offset**2)) * random.choice((-1, 1))
        nx0 = (self.apsize / 2.0) + x_offset
        ny0 = (self.apsize / 2.0) + y_offset
        sx = [0.5 + 0.05 * np.random.randn()]
        sy = [0.5 + 0.05 * np.random.randn()]
        rho = [0.05 + 0.02 * np.random.randn()]

        # calculate comparison factor for neighbor, based on provided difference in magnitude
        self.r = 10 ** (magdiff / 2.5)

        neighbor_args = dict({'A':[self.A / self.r], 'x0':np.array([nx0]),
                              'y0':np.array([ny0]), 'sx':sx, 'sy':sy, 'rho':rho})

        # create neighbor pixel-level light curve
        for c in tqdm(range(self.ncadences)):

            # iterate through cadences, calculate pixel flux values
            n_fpix[c], neighbor[c], n_ferr[c] = PSF(neighbor_args, self.ccd_args,
                                                    self.xpos[c], self.ypos[c])

        # add neighbor to light curve
        fpix += n_fpix
        self.n_fpix = n_fpix

        # calculate flux light curve
        flux = np.sum(np.array(fpix).reshape((self.ncadences), -1), axis=1)

        self.neighbor = True
        self.targets += 1

        self.fpix = fpix
        self.flux = flux

        return self

    def create_aperture(self, fpix=[]):
        """
        Create an aperture including all pixels containing target flux.

        Parameters
        ----------
        `fpix` :
            Pixel-level light curve of dimemsions (apsize, apsize, ncadences). Automatically set to
            fpix generated in GenerateLightCurve() unless a different light curve is passed.
        """

        # check if fpix light curve was passed in
        if len(fpix) == 0:
            fpix = self.fpix

        aperture = np.zeros((self.ncadences, self.apsize, self.apsize))

        # Identify pixels with target flux for each cadence
        for c,f in enumerate(self.target):
            for i in range(self.apsize):
                for j in range(self.apsize):
                    if f[i][j] < 100.:
                        aperture[c][i][j] = 0
                    else:
                        aperture[c][i][j] = 1

        # Identify pixels with target flux for each cadence
        if self.targets > 1:
            for c,f in enumerate(self.n_fpix):
                for i in range(self.apsize):
                    for j in range(self.apsize):
                        if f[i][j] > (.5 * np.max(f)):
                            aperture[c][i][j] = 0

        # Create single aperture
        finalap = np.zeros((self.apsize, self.apsize))

        # Sum apertures to weight pixels
        for i in range(self.apsize):
            for ap in aperture:
                finalap[i] += ap[i]

        max_counts = np.max(finalap)

        # Normalize to 1
        self.weighted_aperture = finalap / max_counts

        # Set excluded pixels to NaN
        for i in range(self.apsize):
            for j in range(self.apsize):
                if finalap[i][j] == 0:
                    finalap[i][j] = np.nan
                else:
                    finalap[i][j] = 1.

        self.aperture = finalap

        return finalap

    def display_aperture(self):
        """Displays aperture overlaid over the first cadence target pixel file."""

        self.create_aperture()
        plt.imshow(self.fpix[0] * self.aperture, origin='lower',
                  cmap='viridis', interpolation='nearest')
        plt.show()

    def display_detector(self):
        """Returns matrix of dimensions (apsize, apsize) for CCD pixel sensitivity."""

        # read in ccd parameters
        cx, cy, apsize, background_level, inter, photnoise_conversion = self.ccd_args

        # Define detector dimensions
        xdim = np.linspace(0, ccd_args['apsize'], 100)
        ydim = np.linspace(0, ccd_args['apsize'], 100)

        # Pixel resolution
        res = int(1000 / ccd_args['apsize'])

        pixel_sens = np.zeros((res, res))

        # Calculate sensitivity function with detector parameters for individual pixel
        for i in range(res):
            for j in range(res):
                pixel_sens[i][j] = np.sum([c * (i-res/2) ** m for m, c in enumerate(ccd_args['cx'])], axis = 0) + \
                                   np.sum([c * (j-res/2) ** m for m, c in enumerate(ccd_args['cy'])], axis = 0)

        # Tile to create detector
        intra = np.tile(pixel_sens, (ccd_args['apsize'], ccd_args['apsize']))
        intra_norm = 1-(intra + np.max(intra))/np.min(intra)
        self.detector = np.zeros((res*ccd_args['apsize'], res*ccd_args['apsize']))

        # Multiply by inter-pixel sensitivity variables
        for i in range(self.apsize):
            for j in range(self.apsize):
                self.detector[i*res:(i+1)*res][j*res:(j+1)*res] = intra_norm[i*res:(i+1)*res][j*res:(j+1)*res] * inter[i][j]

        # Display detector
        plt.imshow(self.detector, origin='lower', cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()

    def estimate_CDPP(self, flux=[]):
        """
        Quick function to calculate and return Combined Differential Photometric Precision (CDPP) of a given light curve.
         If no light curve is passed, this funtion returns the CDPP of the light curve generated in GenerateLightCurve().

        Parameters
        ----------
        `flux` :
            1-dimensional flux light curve for which CDPP is calculated. If nothing is passed into FindCDPP(), it returns
            the CDPP of the light curve generated in GenerateLightCurve()

        Returns
        -------
        `cdpp` : float
            Combined Differential Photometric Precision (CDPP) of given `flux` light curve
        """

        # check if flux light curve was passed in
        if len(flux) == 0:
            flux = self.flux

        cdpp = CDPP(flux)

        return cdpp

    def to_lightkurve_lc(self, aperture_mask='all'):
        """
        Integration with the lightkurve package.

        Returns
        -------
        lc : lightkurve.KeplerLightCurve object
            A `KeplerLightCurve` object from the lightkurve package
        """

        # make sure the lightkurve package is installed
        try:
            from lightkurve import KeplerLightCurve
        except:
            raise ImportError('Could not import lightkurve.')

        # define `KeplerLightCurve` object
        self.lc = self.to_lightkurve_tpf().to_lightcurve(aperture_mask=aperture_mask)
        return self.lc

    def to_lightkurve_tpf(self, target_id="Simulated Target"):
        """
        Integration with the lightkurve package.

        Parameters
        ----------
        target_id : str
            Name of the simulated target. Defaults to "Simulated Target"

        Returns
        -------
        tpf : lightkurve.KeplerTargetPixelFile object
            A `KeplerTargetPixelFile` object from the lightkurve package
        """

        # make sure the lightkurve package is installed
        try:
            from lightkurve.targetpixelfile import KeplerTargetPixelFileFactory
        except:
            raise ImportError('Could not import lightkurve.')

        # instantiate a factory to build our tpf
        factory = KeplerTargetPixelFileFactory(self.ncadences, self.apsize, self.apsize,
                                               target_id=target_id)

        # one cadence at a time, add the flux matrices to the tpf
        for i, tpf in enumerate(self.targetpixelfile):
            factory.add_cadence(flux=tpf, frameno=i)

        # set factory values
        factory.time = self.time
        factory.pos_corr1 = self.xpos
        factory.pos_corr2 = self.ypos
        factory.flux_err = self.ferr

        # generate the tpf
        self.tpf = factory.get_tpf()

        return self.tpf

    def plot(self):
        """Simple plotting function to view first cadence tpf, and both raw and de-trended flux light curves."""

        # initialize subplots with 1:3 width ratio
        fig, ax = plt.subplots(1, 2, figsize=(12,3), gridspec_kw = {'width_ratios':[1, 3]})

        # Get aperture contour
        aperture = self.create_aperture()

        def PadWithZeros(vector, pad_width, iaxis, kwargs):
            vector[:pad_width[0]] = 0
            vector[-pad_width[1]:] = 0
            return vector
        ny, nx = self.fpix[0].shape
        contour = np.zeros((ny, nx))
        contour[np.where(aperture==1)] = 1
        contour = np.lib.pad(contour, 1, PadWithZeros)
        highres = zoom(contour, 100, order=0, mode='nearest')
        extent = np.array([-1, nx, -1, ny])


        # display first cadence tpf
        ax[0].imshow(self.fpix[0], origin='lower', cmap='viridis', interpolation='nearest')
        ax[0].contour(highres, levels=[0.5], extent=extent, origin='lower', colors='r', linewidths=2)

        ax[0].set_title('First Cadence tpf')
        ax[0].set_xlabel('x (pixels)')
        ax[0].set_ylabel('y (pixels)')

        # make sure CDPP is a number before printing it
        if np.isnan(self.estimate_CDPP(self.flux)):
            ax[1].plot(self.t, self.flux, 'r.', alpha=0.3, label='raw flux')
        else:
            ax[1].plot(self.t, self.flux, 'r.', alpha=0.3, label='raw flux (CDPP = %.i)'
                       % self.estimate_CDPP(self.flux))
        ax[1].set_xlim([self.t[0], self.t[-1]])
        ax[1].legend(loc=0)
        ax[1].set_xlabel('Time (days)')
        ax[1].set_ylabel('Flux (counts)')
        ax[1].set_title('Flux Light Curve')

        fig.tight_layout()
        plt.show()

def generate_target(mag=12., roll=1., coords=None, background_level=0.,
                    neighbor_magdiff=1., ncadences=1000, apsize=7, ID=205998445,
                    transit=False, variable=False, neighbor=False, tpf_path=None,
                    no_sensitivity_variation=False, signal=None, **kwargs):
    """
    Parameters
    ----------
     `mag` :
         Magnitude of primary target PSF.
     `roll` :
         Coefficient on K2 motion vectors of target. roll=1 corresponds to current K2 motion.
     `coords` : tuple
         Coordinates of the PSF centroid.
     `background_level` :
         Constant background signal in each pixel. Defaults to 0.
     `neighbor_magdiff` :
         Difference between magnitude of target and neighbor. Only accessed if neighbor initialized as
         `True` or if AddNeighbor() function is called.
     `photnoise_conversion` :
         Conversion factor for photon noise, defaults to 0.000625 for consistency with benchmark.
     `ncadences` :
         Number of cadences in simulated light curve.
     `apsize` :
         Dimension of aperture on each side.

     Returns
     -------
     `Target`: :class:`Target` object
        A simulated CCD observation
    """

    aperture = np.ones((ncadences, apsize, apsize))

    # calculate PSF amplitude for given Kp Mag
    A = _calculate_PSF_amplitude(mag)

    if tpf_path is None:
        # read in K2 motion vectors for provided K2 target (EPIC ID #)
        try:
            tpf = lk.search_targetpixelfile(ID)[0].download()
        except OSError:
            raise ScopeError('Unable to access internet. Please provide a path '
                             '(str) to desired file for motion using the `tpf` '
                             'keyword.')
    else:
        tpf = lk.open(tpf_path)

    xpos = tpf.pos_corr1
    ypos = tpf.pos_corr2
    t = tpf.time

    # If a transit is included, create the model
    if transit:
        model = TransitModel(t)
        signal = model.create_starry_model(**kwargs)

    # throw out outliers
    for i in range(len(xpos)):
        if abs(xpos[i]) >= 50 or abs(ypos[i]) >= 50:
            xpos[i] = 0
            ypos[i] = 0
        if np.isnan(xpos[i]):
            xpos[i] = 0
        if np.isnan(ypos[i]):
            ypos[i] = 0

    # crop to desired length and multiply by roll coefficient
    xpos = xpos[0:ncadences] * roll
    ypos = ypos[0:ncadences] * roll

    if no_sensitivity_variation:
        cx = [1., 0., 0.]
        cy = [1., 0., 0.]
        inter = np.ones((apsize, apsize))
    else:
        # create self.inter-pixel sensitivity variation matrix
        # random normal distribution centered at 0.975
        inter = np.zeros((apsize, apsize))
        for i in range(apsize):
            for j in range(apsize):
                inter[i][j] = (0.975 + 0.001 * np.random.randn())

        # cx,cy: intra-pixel variation polynomial coefficients in x,y
        cx = [1.0, 0.0, -0.05]
        cy = [1.0, 0.0, -0.05]

    if coords is None:
        # x0,y0: center of PSF, half of aperture size plus random deviation
        x0 = (apsize / 2.0) + 0.2 * np.random.randn()
        y0 = (apsize / 2.0) + 0.2 * np.random.randn()
    else:
        x0, y0 = coords

    # sx,sy: standard deviation of Gaussian in x,y
    # rho: rotation angle between x and y dimensions of Gaussian
    sx = [0.5]
    sy = [0.5]
    rho = [0.0]

    psf_args = dict({'A':A, 'x0':np.array([x0]), 'y0':np.array([y0]),
                     'sx':sx, 'sy':sy, 'rho':rho})

    ccd_args = dict({'cx':cx, 'cy':cy, 'apsize':apsize, 'background_level':background_level,
                     'inter':inter, 'photnoise_conversion':0.000625})

    fpix, flux, ferr, target = calculate_pixel_values(ncadences=ncadences, apsize=apsize,
                                                      psf_args=psf_args, ccd_args=ccd_args,
                                                      xpos=xpos, ypos=ypos, signal=signal)

    t = t[:ncadences]

    return Target(fpix, flux, ferr, target, t, mag=mag, roll=roll,
                  neighbor_magdiff=neighbor_magdiff, ncadences=ncadences,
                  apsize=apsize, ccd_args=ccd_args, psf_args=psf_args, xpos=xpos,
                  ypos=ypos)

def fetch_psf_params():
    pass

def fetch_ccd_params():
    pass

def calculate_pixel_values(ncadences, apsize, psf_args, ccd_args, xpos, ypos, signal=None):
    """Returns the Target Pixel File generated by the """
    if signal is None:
        signal_amplitude = np.ones(ncadences)
    else:
        signal_amplitude = signal

    # initialize pixel flux light curve, target light curve, and isolated noise in each pixel
    fpix = np.zeros((ncadences, apsize, apsize))
    target = np.zeros((ncadences, apsize, apsize))
    ferr = np.zeros((ncadences, apsize, apsize))

    base_amplitude = psf_args['A']

    # The PSF function calculates flux in each pixel
    # Iterate through cadences (c), and x and y dimensions on the detector (i,j)
    for c in tqdm(range(ncadences)):
        A = base_amplitude * signal_amplitude[c]
        fpix[c], target[c], ferr[c] = PSF(A, psf_args, ccd_args, xpos[c], ypos[c])

    flux = np.sum(fpix.reshape((ncadences), -1), axis=1)

    return fpix, flux, ferr, target
