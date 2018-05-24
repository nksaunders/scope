#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
test_scripts.py
---------------
Test all of the scripts.
'''

from __future__ import division, print_function, absolute_import, \
                       unicode_literals

import sys, os
sys.path.append("..")
import scope


def test_all():
    '''
    Test all scripts in the `scripts/` directory.

    '''
    print('Testing scope...')
    star = scope.Target()
    star.GenerateLightCurve(ncadences=1)

    print('Success!')


if __name__ == '__main__':
    test_all()
