#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Batch
-----
Perform a batch run to generate multiple simulated observations.
'''

import numpy as np
import matplotlib.pyplot as pl
import scope
from scope.scopemath import PLD
from tqdm import tqdm
import itertools
import os
from os.path import abspath, join
import warnings

from scope import PACKAGEDIR
from .utils import *

__all__ = ['Batch', 'run_batch']

class Batch(object):

    def __init__(self, n, mags, motion_mags, directory):
        """A batch object to generate multiple simulated light curves."""

        self.n = n
        self.mags = mags
        self.motion_mags = motion_mags
        self.directory = directory

    def Simulate(self, offline, **kwargs):

        for iter in range(self.n):
            for m, mot in zip(self.mags, self.motion_mags):
                print("Generating target: mag = %.2f, motion_mag = %.2f..." % (m, mot))

                if offline:
                    ftpf = abspath(join(PACKAGEDIR, os.pardir, '.kplr', 'data', 'k2',
                                        'target_pixel_files', '205998445',
                                        'ktwo205998445-c03_lpd-targ.fits.gz'))
                    sK2 = scope.generate_target(mag=m, roll=mot, ftpf=ftpf, **kwargs)
                else:
                    sK2 = scope.generate_target(mag=m, roll=mot, **kwargs)

                if not os.path.isdir(self.directory):
                    os.makedirs(self.directory)

                # check to see if file exists, skip if it's already there
                if os.path.isfile(os.path.join(self.directory, '%2dmag%.2fmotion%.2f.npz' % (iter, m, mot))):
                    warnings.warn("Mag = %.2f, m_mag = %.2f already exists!" % (m, mot))

                # create missing lc
                else:
                    tpf, flux, ferr = sK2.targetpixelfile, sK2.lightcurve, sK2.error
                    np.savez(os.path.join(self.directory, '%2dmag%.2fmotion%.2f' % (iter, m, mot)),
                             tpf=tpf, flux=flux, ferr=ferr)


def run_batch(n, mags, motion_mags, directory, offline=False, **kwargs):
    '''
    Performs a batch simulation run to easily generate multiple targets at the
    same time.

    Parameters
    ----------
    `n` : int
        Number of targets to be generated for each combination of magnitude and
        motion.
    `mags` : list
        List of magnitudes for generated targets. For a single magnitude, use a
        list with a single entry (for example `[10]`).
    `motion_mags` : list
        List of motion intensity of simulated targets. For a single motion
        magnitude, use a list with a single entry (for example `[3]`).
    `directory` : path, str
        The directory where simulated targets will be saved.
    `offline` : boolean
        If the user does not have access to the internet, this script can still
        be run using motion vectors from a stored target. Default is `False`.
    `kwargs` :
        Keyword arguments for the `scope.generate_target` function.
    '''

    b = Batch(n, mags, motion_mags, directory)
    b.Simulate(offline, **kwargs)
