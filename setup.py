#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
from setuptools import setup, find_packages

# Hackishly inject a constant into builtins to enable importing of the
# module in "setup" mode. Stolen from `kplr`
import sys
if sys.version_info[0] < 3:
  import __builtin__ as builtins
else:
  import builtins
builtins.__SKOPE_SETUP__ = True
import skope

long_description = \
"""
Synthetic K2 Objects for PLD Experimentation
Generates a forward model of the Kepler detector
including inter- and intra-pixel sensitivity variation,
and models synthetic stellar PSFs traversing the CCD.
"""

# Setup!
setup(name = 'skope',
      version = skope.__version__,
      description = 'Synthetic K2 Objects for PLD Experimentation',
      long_description = long_description,
      classifiers = [
                      # 'Development Status :: 5 - Production/Stable',
                      'License :: OSI Approved :: MIT License',
                      'Programming Language :: Python',
                      'Programming Language :: Python :: 3',
                      'Topic :: Scientific/Engineering :: Astronomy',
                    ],
      url = 'http://github.com/nksaunders/SKOPE',
      author = 'Nicholas Saunders',
      author_email = 'nks1994@uw.edu',
      license = 'MIT',
      packages = ['skope'],
      install_requires = [
                          'everest',
                         ],
      include_package_data = True,
      zip_safe = False,
      )
