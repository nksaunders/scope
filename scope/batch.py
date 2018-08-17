import numpy as np
import matplotlib.pyplot as pl
import scope
from scope.scopemath import PLD
from tqdm import tqdm
import itertools
from everest.pool import Pool
from everest.missions.k2 import CDPP
from everest.config import EVEREST_SRC
import os
import os.path

# Number of targets to simulate
niter = 5

# Magnitude and motion arrays
mags = np.arange(10., 16., .5)
m_mags = np.arange(0., 21., 1)

def Simulate(arg):

    iter, mag, m_mag = arg
    print("Running mag = %.2f, m_mag = %.2f..." % (mag, m_mag))
    sK2 = scope.Target(variable=True, ftpf = os.path.expanduser('/usr/lusers/nks1994/scope/.kplr/data/k2/target_pixel_files/205998445/ktwo205998445-c03_lpd-targ.fits.gz'))

    # check to see if file exists, skip if it's already there
    if os.path.isfile('/usr/lusers/nks1994/hyaktest/everest_data/scope_batch3/%2dmag%.2fmotion%.2f.npz' % (iter, mag, m_mag)):
        print("Mag = %.2f, m_mag = %.2f already exists!" % (mag, m_mag))

    # create missing lc
    else:
        fpix, flux, ferr = sK2.GenerateLightCurve(mag=mag, roll=m_mag, background_level=20, ncadences=1000, apsize=13)
        np.savez('/usr/lusers/nks1994/hyaktest/everest_data/scope_batch2/%2dmag%.2fmotion%.2f' % (iter, mag, m_mag), fpix=fpix, flux=flux, ferr=ferr)

def Benchmark():
    '''

    '''

    # astroML format for consistent plotting style
    from astroML.plotting import setup_text_plots
    setup_text_plots(fontsize=10, usetex=True)

    # Compare zero-motion synthetic data to original Kepler raw CDPP
    print("Plotting Figure 1...")
    _, kepler_kp, kepler_cdpp6 = np.loadtxt(os.path.join(EVEREST_SRC, 'missions', 'k2', 'tables', 'kepler.cdpp'), unpack = True)
    fig, ax = pl.subplots(1)
    ax.plot(kepler_kp, kepler_cdpp6, 'y.', alpha = 0.01, zorder = -1)
    ax.set_rasterization_zorder(-1)
    bins = np.arange(7.5,18.5,0.5)
    by = np.zeros_like(bins) * np.nan
    for b, bin in enumerate(bins):
        i = np.where((kepler_cdpp6 > -np.inf) & (kepler_cdpp6 < np.inf) & (kepler_kp >= bin - 0.5) & (kepler_kp < bin + 0.5))[0]
        if len(i) > 10:
            by[b] = np.median(kepler_cdpp6[i])
    ax.plot(bins, by, 'yo', label = 'Kepler', markeredgecolor = 'k')
    for iter in range(niter):
        cdpp = np.zeros_like(mags)
        for i, mag in enumerate(mags):
            # flux = np.load('batch/plot_run7/%2dmag%.2fmotion%.2f.npz' % (iter, mag, 0.))['flux']

            # load in fpix
            fpix = np.load('batch/plot_run7/%2dmag%.2fmotion%.2f.npz' % (iter, mag, 0.))['fpix']
            # crop out extra pixels
            crop = np.array([f[6:7,6:7] for f in fpix])
            # sum into flux
            flux = np.sum(crop.reshape((len(crop)), -1), axis=1)

            # calculate CDPP
            cdpp[i] = CDPP(flux)
            # import pdb; pdb.set_trace()

        if iter == 0:
            ax.plot(mags, cdpp, 'b.', label = 'Synthetic (0x motion)')
        else:
            ax.plot(mags, cdpp, 'b.')
    ax.set_xlabel('Kepler Magnitude')
    ax.set_ylabel('CDPP [ppm]')
    ax.set_ylim(-10, 500)
    ax.set_xlim(8, 18)
    ax.legend(loc = 'best')
    '''
    # Compare 1x motion to K2 raw CDPP from campaign 3
    print("Plotting Figure 2...")
    _, kp, cdpp6r, _, _, _, _, _, _ = np.loadtxt(os.path.join(EVEREST_SRC, 'missions', 'k2', 'tables', 'c03_nPLD.cdpp'), unpack = True, skiprows = 2)
    fig, ax = pl.subplots(1, figsize=(7.5,5))
    ax.plot(kp, cdpp6r, 'r.', alpha = 0.05, zorder = -1)
    ax.set_rasterization_zorder(-1)
    bins = np.arange(7.5,18.5,0.5)
    by = np.zeros_like(bins) * np.nan
    for b, bin in enumerate(bins):
        i = np.where((cdpp6r > -np.inf) & (cdpp6r < np.inf) & (kp >= bin - 0.5) & (kp < bin + 0.5))[0]
        if len(i) > 10:
            by[b] = np.median(cdpp6r[i])
    ax.plot(bins, by, 'ro', label = 'Raw K2', markeredgecolor = 'k')
    for iter in range(niter):
        cdpp = np.zeros_like(mags)
        for i, mag in enumerate(mags):
            # flux = np.load('batch/plot_run7/%2dmag%.2fmotion%.2f.npz' % (iter, mag, 1.))['flux']
            # perform aperture masking
            fpix = np.load('batch/plot_run7/%2dmag%.2fmotion%.2f.npz' % (iter, mag, 1.))['fpix']
            # crop out extra pixels
            crop = np.array([f[2:10,2:10] for f in fpix])
            # sum into flux
            flux = np.sum(crop.reshape((len(crop)), -1), axis=1)
            cdpp[i] = CDPP(flux)
        if iter == 0:
            ax.plot(mags, cdpp, 'b.', label = 'Synthetic (1x motion)')
        else:
            ax.plot(mags, cdpp, 'b.')
    ax.set_xlabel('Kepler Magnitude')
    ax.set_ylabel('CDPP [ppm]')
    ax.set_ylim(-30, 1500)
    ax.set_xlim(8, 18)
    ax.legend(loc = 'best')

    # Plot several different motion vectors
    print("Plotting Figure 3...")
    _, kp, cdpp6r, _, _, _, _, _, _ = np.loadtxt(os.path.join(EVEREST_SRC, 'missions', 'k2', 'tables', 'c03_nPLD.cdpp'), unpack = True, skiprows = 2)
    fig, ax = pl.subplots(1, figsize = (6, 7))
    ax.plot(kp, cdpp6r, 'r.', alpha = 0.05, zorder = -1)
    ax.set_rasterization_zorder(-1)
    bins = np.arange(7.5,18.5,0.5)
    by = np.zeros_like(bins) * np.nan
    for b, bin in enumerate(bins):
        i = np.where((cdpp6r > -np.inf) & (cdpp6r < np.inf) & (kp >= bin - 0.5) & (kp < bin + 0.5))[0]
        if len(i) > 10:
            by[b] = np.median(cdpp6r[i])
    ax.plot(bins, by, 'ro', label = 'Raw K2', markeredgecolor = 'k')
    for m_mag, color in zip([1, 2, 5, 10, 20], ['b', 'g', 'y', 'orange', 'k']):
        cdpp = [[] for mag in mags]
        for i, mag in enumerate(mags):
            for iter in range(niter):
                flux = np.load('batch/plot_run7/%2dmag%.2fmotion%.2f.npz' % (iter, mag, m_mag))['flux']
                cdpp[i].append(CDPP(flux))
        cdpp = np.nanmean(np.array(cdpp), axis = 1)
        ax.plot(mags, cdpp, '.', color = color, label = 'Synthetic (%dx motion)' % m_mag)
        ax.plot(mags, cdpp, '-', color = color)
    ax.set_xlabel('Kepler Magnitude')
    ax.set_ylabel('CDPP [ppm]')
    ax.set_ylim(-30, 2500)
    ax.set_xlim(8, 18)
    ax.legend(loc = 'best')
    '''
    pl.show()

if __name__ == '__main__':

    Benchmark()
    '''
    # Run!
    combs = list(itertools.product(range(niter), mags, m_mags))
    with Pool() as pool:
        pool.map(Simulate, combs)
    '''
