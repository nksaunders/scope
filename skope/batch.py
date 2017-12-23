import numpy as np
import matplotlib.pyplot as pl
import skope

star = skope.Target()

mags = [10, 11, 12, 13, 14]
roll = [1., 2., 5., 10., 20.]

for m in mags:
    for r in roll:

        fpix, flux, ferr = star.GenerateLightCurve(m, roll=r, apsize=9)
        np.savez(('stars/mag%iroll%i'%(m,r)), fpix=fpix, ferr=ferr)
