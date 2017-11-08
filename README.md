# skope
**S**ynthetic ***K**2* **O**bjects for **P**LD **E**xperimentation

**skope** creates a forward model of the *Kepler* detector with pixel sensitivity variation, and synthetic *K2* targets. This model allows the creation of light curves to test de-trending methods. As *Kepler* runs out of fuel, telescope motion will increase in magnitude and become less predictable. **skope** simulates targets traversing the CCD with high motion, allowing the characterization of instrumental noise for high-motion cases.

This package includes methods to change magnitude of motion and sensitivity properties of the CCD, inject synthetic transiting exoplanet targets and stellar variability, and test PLD de-trending. For examples of usage, see Notebooks/Examples.

Note that **skope** depends on the **EVEREST** pipeline (Luger et. al). **EVEREST** can be installed with:
<pre><code>pip install everest-pipeline</code></pre>
