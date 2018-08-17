#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
from setuptools import setup, find_packages
import sys
if sys.version_info[0] < 3:
  import __builtin__ as builtins
else:
  import builtins
builtins.__SKOPE_SETUP__ = True
import scope

long_description = \
"""
Synthetic K2 Objects for PLD Experimentation
Generates a forward model of the Kepler detector
including inter- and intra-pixel sensitivity variation,
and models synthetic stellar PSFs traversing the CCD.
"""

# Setup!
setup(name = 'tele-scope',
      version = scope.__version__,
      description = 'Simulated CCD Observations for Photometric Experimentation',
      long_description = long_description,
      classifiers = [
                      # 'Development Status :: 5 - Production/Stable',
                      'License :: OSI Approved :: MIT License',
                      'Programming Language :: Python',
                      'Programming Language :: Python :: 3',
                      'Topic :: Scientific/Engineering :: Astronomy',
                    ],
      url = 'https://github.com/nksaunders/scope',
      author = 'Nicholas Saunders',
      author_email = 'nks1994@uw.edu',
      license = 'MIT',
      packages = ['scope'],
      install_requires = [
                          'everest-pipeline',
                          'astroML',
                         ],
      include_package_data = True,
      zip_safe = False,
      test_suite='nose.collector',
      tests_require=['nose']
      )
