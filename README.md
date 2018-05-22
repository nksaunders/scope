# scope
**S**imulated **C**CD **O**bservations for **P**hotometric **E**xperimentation

**scope** creates a forward model of telescope detectors with pixel sensitivity variation, and synthetic stellar targets with motion relative to the CCD. This model allows the creation of light curves to test de-trending methods for existing and future telescopes. The primary application of this package is the simulation of the *Kepler* Space Telescope detector to prepare for increased instrumental noise in its final campaigns of observation.

This package includes methods to change magnitude of motion and sensitivity properties of the CCD, inject synthetic transiting exoplanet targets and stellar variability, and test PLD de-trending.

For examples of usage, see [the sample notebook](https://nksaunders.github.io/files/Example.html).

To install **scope**, clone and navigate to the directory and run
<pre><code>python setup.py install</code></pre>

Note that **scope** depends on the **EVEREST** pipeline (Luger et. al). **EVEREST** can be installed with
<pre><code>pip install everest-pipeline</code></pre>
