import starry
import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.constants import G

class TransitModel(object):

    def __init__(self, time):

        self.time = time

    def create_starry_model(self, rprs=.01, period=15., t0=5., i=90, ecc=0., m_star=1.):
        """ """
        # instantiate a starry primary object (star)
        star = starry.kepler.Primary()

        # calculate separation
        a = self._calculate_separation(m_star, period)

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
        planet.a = star.r*(a*u.AU).to(u.solRad).value # in units of stellar radius

        # create a system and compute its lightcurve
        system = starry.kepler.System(star, planet)
        system.compute(self.time)

        # return the light curve
        return system.lightcurve

    def create_transit_mask(self, transit_model):

        # Define transit mask
        trninds = np.where(transit_model < 1.0)
        M = lambda x: np.delete(x, trninds, axis=0)

        return M


    def _calculate_duration(self, rprs, period, i):
        """ """

        a = .001
        b = ((1 + rprs)**2 - ((1/a)*np.cos(i*u.deg))**2) / (1 - np.cos(i*u.deg)**2)

        dur = (period / np.pi) * np.arcsin(a * b**1/2).value
        return dur

    def _calculate_separation(self, m_star, period):
        """ """

        a = (((G*m_star*u.solMass/(4*np.pi**2))*(period*u.day)**2)**(1/3))

        return a.to(u.AU).value
