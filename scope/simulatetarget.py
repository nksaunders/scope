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

from .scopemath import PSF, PLD
from .utils import *

__all__ = ['Target', 'generate_target']

class Target(object):
    """A simulated stellar object with a forward model of a telescope detector's sensitivity variation"""

    def __init__(self, fpix, flux, ferr, target, t, mag=12., roll=1., neighbor_magdiff=1.,
                 ncadences=1000, apsize=7, transit=False, variable=False, neighbor=False,
                 ccd_args=[], psf_args=[], xpos=None, ypos=None):

        # initialize self variables
        self.transit = transit
        self.variable = variable
        self.neighbor = neighbor
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

        # add transit and variability
        if transit:
            self.add_transit()
        if variable:
            self.add_variability()
        if neighbor:
            self.add_neighbor()

    @property
    def targetpixelfile(self):
        return self.fpix

    @property
    def lightcurve(self):
        return self.flux

    @property
    def time(self):
        return self.t

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

        # check if fpix light curve was passed in
        if len(fpix) == 0:
            fpix = self.fpix

        # instantiate a starry primary object (star)
        star = starry.kepler.Primary()
        self.stellar_r = star.r
        self.stellar_L = star.L

        # calculate separation
        a = (((G*m_star*u.solMass/(4*np.pi**2))*(period*u.day)**2)**(1/3)).to(u.AU).value

        # store values
        self.transit = True
        self.rprs = rprs
        self.period = period
        self.t0 = t0
        self.i = i
        self.ecc = ecc
        self.m_star = m_star
        self.a = a

        # quadradic limb darkening
        star[1] = 0.40
        star[2] = 0.26

        # instantiate a starry secondary object (planet)
        planet = starry.kepler.Secondary(lmax=5)

        # define its parameters
        planet.r = rprs * star.r
        planet.porb = period
        planet.tref = t0
        planet.inc = i
        planet.ecc = ecc
        planet.a = star.r*(a*u.AU).to(u.solRad).value

        # create a system and compute its lightcurve
        system = starry.kepler.System(star, planet)
        system.compute(self.t)
        self.trn = system.lightcurve

        # Define transit mask
        self.trninds = np.where(self.trn > 1.0)
        self.M = lambda x: np.delete(x, self.trninds, axis=0)

        self.fpix, self.flux, self.ferr, self.target = calculate_pixel_values(ncadences=self.ncadences, apsize=self.apsize,
                                                                              psf_args=self.psf_args, ccd_args=self.ccd_args,
                                                                              xpos=self.xpos, ypos=self.ypos, signal=self.trn)

        """# Add transit to light curve
        self.fpix_trn = np.zeros((self.ncadences, self.apsize, self.apsize))
        for i,c in enumerate(fpix):
            self.fpix_trn[i] = c * self.trn[i]

        # Create flux light curve
        self.flux_trn = np.sum(self.fpix_trn.reshape((self.ncadences), -1), axis=1)"""


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
        V_fpix = [f * V[i] for i,f in enumerate(fpix)]

        # Create flux light curve
        V_flux = np.sum(np.array(V_fpix).reshape((self.ncadences), -1), axis=1)

        self.fpix = V_fpix
        self.flux = V_flux

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

        neighbor_args = np.concatenate([[self.A / self.r], np.array([nx0]),
                                       np.array([ny0]), sx, sy, rho])

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
        if self.neighbor:
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

    def calculate_duration(self, rprs, period, i):
        """ """

        a = .001
        b = ((1 + rprs)**2 - ((1/a)*np.cos(i*u.deg))**2) / (1 - np.cos(i*u.deg)**2)

        dur = (period / np.pi) * np.arcsin(a * b**1/2).value
        return dur

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
        xdim = np.linspace(0, self.apsize, 100)
        ydim = np.linspace(0, self.apsize, 100)

        # Pixel resolution
        res = int(1000 / self.apsize)

        pixel_sens = np.zeros((res,res))

        # Calculate sensitivity function with detector parameters for individual pixel
        for i in range(res):
            for j in range(res):
                pixel_sens[i][j] = np.sum([c * (i-res/2) ** m for m, c in enumerate(cx)], axis = 0) + \
                np.sum([c * (j-res/2) ** m for m, c in enumerate(cy)], axis = 0)

        # Tile to create detector
        intra = np.tile(pixel_sens, (self.apsize, self.apsize))
        intra_norm = 1-(intra + np.max(intra))/np.min(intra)
        self.detector = np.zeros((res*self.apsize,res*self.apsize))

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

    def to_lightkurve_lc(self):
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
        self.lc = self.to_lightkurve_tpf().to_lightcurve()
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

        # plot raw and de-trend light curves
        self.detrend()

        # make sure CDPP is a number before printing it
        if np.isnan(self.estimate_CDPP(self.flux)):
            ax[1].plot(self.t, self.rawflux, 'r.', alpha=0.3, label='raw flux')
            ax[1].plot(self.t, self.flux, 'k.', label='de-trended')
        else:
            ax[1].plot(self.t, self.rawflux, 'r.', alpha=0.3, label='raw flux (CDPP = %.i)'
                       % self.estimate_CDPP(self.rawflux))
            ax[1].plot(self.t, self.flux, 'k.', label='de-trended (CDPP = %.i)'
                       % self.estimate_CDPP(self.flux))
        ax[1].set_xlim([self.t[0], self.t[-1]])
        ax[1].legend(loc=0)
        ax[1].set_xlabel('Time (days)')
        ax[1].set_ylabel('Flux (counts)')
        ax[1].set_title('Flux Light Curve')

        fig.tight_layout()
        plt.show()

def generate_target(mag=12., roll=1., background_level=0., ccd_args=[], neighbor_magdiff=1.,
                    photnoise_conversion=.000625, ncadences=1000, apsize=7, ID=205998445,
                    custom_ccd=False, transit=False, variable=False, neighbor=False, tpf_path=None,
                    no_variation=False, signal=None):
    """

    Parameters
    ----------
     `mag` :
         Magnitude of primary target PSF.
     `roll` :
         Coefficient on K2 motion vectors of target. roll=1 corresponds to current K2 motion.
     `background_level` :
         Constant background signal in each pixel. Defaults to 0.
     `ccd_args` :
         Autogenerated if nothing passed, otherwise takes the following arguments:
         `cx` : sensitivity variation coefficients in `x`
         `cy` : sensitivity variation coefficients in `y`
         `apsize` : see below
         `background_level` : see above
         `inter` : matrix (apsize x apsize) of stochastic inter-pixel sensitivity variation
         `photnoise_conversion`: see below
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

    # TODO: need to define signal if variability or transit are True

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

    # create self.inter-pixel sensitivity variation matrix
    # random normal distribution centered at 0.975
    inter = np.zeros((apsize, apsize))
    for i in range(apsize):
        for j in range(apsize):
            inter[i][j] = (0.975 + 0.001 * np.random.randn())

    # assign PSF model parameters to be passed into PixelFlux function
    if not custom_ccd:

        # cx,cy: intra-pixel variation polynomial coefficients in x,y
        cx = [1.0, 0.0, -0.05]
        cy = [1.0, 0.0, -0.05]

        # x0,y0: center of PSF, half of aperture size plus random deviation
        x0 = (apsize / 2.0) + 0.2 * np.random.randn()
        y0 = (apsize / 2.0) + 0.2 * np.random.randn()

        # sx,sy: standard deviation of Gaussian in x,y
        # rho: rotation angle between x and y dimensions of Gaussian
        sx = [0.5 + 0.05 * np.random.randn()]
        sy = [0.5 + 0.05 * np.random.randn()]
        rho = [0.05 + 0.02 * np.random.randn()]
        # TODO: This should be a dictionary
        psf_args = np.concatenate([[A], np.array([x0]), np.array([y0]), sx, sy, rho])

    if no_variation:
        cx = [1., 0., 0.]
        cy = [1., 0., 0.]
        inter = np.ones((apsize, apsize))

    ccd_args = [cx, cy, apsize, background_level, inter, photnoise_conversion]
    ccd_args = ccd_args

    fpix, flux, ferr, target = calculate_pixel_values(ncadences=ncadences, apsize=apsize,
                                              psf_args=psf_args, ccd_args=ccd_args,
                                              xpos=xpos, ypos=ypos, signal=signal)

    return Target(fpix, flux, ferr, target, t, mag=mag, roll=roll,
                  neighbor_magdiff=neighbor_magdiff, ncadences=ncadences,
                  apsize=apsize, transit=transit, variable=variable,
                  neighbor=neighbor, ccd_args=ccd_args, psf_args=psf_args, xpos=xpos,
                  ypos=ypos)


def calculate_pixel_values(ncadences, apsize, psf_args, ccd_args, xpos, ypos, signal=None):
    """ """
    if signal is None:
        signal_amplitude = np.ones(ncadences)
    else:
        signal_amplitude = signal

    # initialize pixel flux light curve, target light curve, and isolated noise in each pixel
    fpix = np.zeros((ncadences, apsize, apsize))
    target = np.zeros((ncadences, apsize, apsize))
    ferr = np.zeros((ncadences, apsize, apsize))

    '''
    Here is where the light curves are created
    PSF function calculates flux in each pixel
    Iterate through cadences (c), and x and y dimensions on the detector (i,j)
    '''

    for c in tqdm(range(ncadences)):

        psf_args[0] *= signal_amplitude[c]
        fpix[c], target[c], ferr[c] = PSF(psf_args, ccd_args, xpos[c], ypos[c])

    flux = np.sum(fpix.reshape((ncadences), -1), axis=1)

    return fpix, flux, ferr, target


def _calculate_PSF_amplitude(mag):
    """
    Returns the amplitude of the PSF for a star of a given magnitude.

    Parameters
    ----------
    `mag`: float
        Input magnitude.

    Returns
    -------
    amp : float
        Corresponding PSF applitude.
    """

    # mag/flux relation constants
    a,b,c = 1.65e+07, 0.93, -7.35
    amp = a * np.exp(-b * (mag+c))
    return amp


def _interpolate_nans(y):
    """Helper to handle indices and logical indices of NaNs.

    Parameters
    ----------
    `y` :
        1d numpy array with possible NaNs

    Returns
    -------
    `nans` :
        logical indices of NaNs
    `index` :
        a function, with signature indices= index(logical_indices),
        to convert logical indices of NaNs to 'equivalent' indices

    Example
    -------
    >>> # linear interpolation of NaNs
    >>> nans, x= nan_helper(y)
    >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    nans = np.isnan(y)
    x = lambda z: z.nonzero()[0]

    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    return y
