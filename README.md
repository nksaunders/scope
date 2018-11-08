<p align="center">
  <img src="https://nksaunders.github.io/images/scope_logo.png" width="200">
</p>

<p align="center">
  <a href="https://travis-ci.org/nksaunders/scope"><img src="https://travis-ci.org/nksaunders/scope.svg?branch=master"/></a>
  <a href="https://nksaunders.github.io/scope"><img src="https://img.shields.io/badge/read-the_docs-blue.svg?style=flat"/></a>   <a href="https://github.com/nksaunders/scope/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-brightgreen.svg"/></a>
</p>

**S**imulated **C**CD **O**bservations for **P**hotometric **E**xperimentation

**scope** creates a forward model of telescope detectors with pixel sensitivity variation, and synthetic stellar targets with motion relative to the CCD. This model allows the creation of light curves to test de-trending methods for existing and future telescopes. The primary application of this package is the simulation of the *Kepler* Space Telescope detector to prepare for increased instrumental noise in its final campaigns of observation.

This package includes methods to change magnitude of motion and sensitivity properties of the CCD, inject synthetic transiting exoplanet targets and stellar variability, and test PLD de-trending.

<p align="center">
  <img src="https://nksaunders.github.io/images/sample_output.png">
</p>

For examples of usage, see [the sample notebook](https://nksaunders.github.io/files/Example.html).

To install **scope**, run
<pre><code>pip install tele-scope</code></pre>

Note that **scope** depends on the **EVEREST** pipeline ([Luger et. al 2016](https://rodluger.github.io/everest/pipeline.html)). **EVEREST** can be installed with
<pre><code>pip install everest-pipeline</code></pre>
