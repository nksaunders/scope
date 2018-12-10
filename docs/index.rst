.. scope documentation master file, created by
   sphinx-quickstart on Wed May 23 14:00:17 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: https://nksaunders.github.io/images/scope_logo.png

Welcome to scope!
=================
Simulated CCD Observations for Photometric Experimentation

scope creates a forward model of telescope detectors with pixel sensitivity variation, and synthetic stellar targets with motion relative to the CCD. This model allows the creation of light curves to test de-trending methods for existing and future telescopes. The primary application of this package is the simulation of the Kepler Space Telescope detector to prepare for increased instrumental noise in its final campaigns of observation.

This package includes methods to change magnitude of motion and sensitivity properties of the CCD, inject synthetic transiting exoplanet targets and stellar variability, and test PLD de-trending.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   simulatetarget
   scopemath
   batch

   Source Code on Github <https://github.com/nksaunders/scope>

   Examples <https://nksaunders.github.io/files/Example.html>

Index
=====

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
