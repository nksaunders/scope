#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
test_main.py
------------
Test scope script.
'''

from __future__ import division, print_function, absolute_import, \
                       unicode_literals

import sys, os
sys.path.append("..")
import scope


def test_all():

    print('Testing scope...')
    star = scope.Target()
    star.GenerateLightCurve(ncadences=1)

    print('Success!')


if __name__ == '__main__':
    test_all()
