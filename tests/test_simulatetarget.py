#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
test_simulatetarget.py
----------------------
'''

from __future__ import division, print_function, absolute_import, \
                       unicode_literals

import sys, os
from os.path import abspath, join
import scope
from scope import PACKAGEDIR

# set file directory to local tpf
ftpf = abspath(join(PACKAGEDIR, os.pardir, '.kplr', 'data', 'k2',
                    'target_pixel_files', '205998445',
                    'ktwo205998445-c03_lpd-targ.fits.gz'))

def test_generate():
    # test generating a single cadence target
    star = scope.generate_target(ftpf=ftpf, ncadences=2)

    # make sure properties look right
    assert(len(star.lightcurve) == 2)
    assert(len(star.targetpixelfile) == 2)
    assert(len(star.error) == 2)

def test_roll():
    star = scope.generate_target(roll=5, ftpf=ftpf, ncadences=1)
    assert(star.roll == 5)

def test_detrend():
    pass

if __name__ == '__main__':
    test_all()
