#!/usr/bin/env python
"""
Lightcurve analysis of data from photometry pipeline
2021-06-29, nmosko@lowell.edu
       
Parameters:
    photfile (str): name of photometry*.dat file
    fit_order (int): order of Fourier fit to data

Optional Arguments:
    -make_plots: Write summary plots to file
    -fourier_step (float): period step size in Fourier period scan [hr]
    -min_period (float): minimum period to search [hr]
    -max_period (float): maximum period to search [hr]

Usage:
    pp_lightcurve.py photometry.dat 5 -make_plots -fourier_step 0.01 -min_period 0.1 -max_period 4

"""

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, Column
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit as cf


##########################
# Output plots to file
##########################
def make_plots(best_period,phase,fit,periods,chisq,power,frequency):
    '''
    Generate a plot of the unfolded lightcurve and the period analysis results
    '''

    # if specified to ignore SourceExtractor flag then set to 0 for all frames
    if ignore_flag:
            dat['[8]'] = 0

    # Raw photometry plotted as an unfolded lightcurve
    plt.errorbar((dat['julian_date'][dat['[8]'] == 0]-jd0)*24.,dat['mag'][dat['[8]'] == 0],\
                 yerr=dat['sig'][dat['[8]'] == 0],fmt='o',markersize=0.5,linestyle=' ',elinewidth=0.5)
    
#    plt.plot(np.linspace(min(time_hr),max(time_hr),10000), fit+np.mean(dat['mag'][dat['[8]'] == 0]),'k-')


    plt.ylabel('Apparent Magnitude')
    plt.xlabel('Time (hr)')
    tit = photfile.replace('photometry_','').replace('.dat','').replace('_','')
    plt.title(tit,fontsize=8)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(tit+'_lightcurve.png',format='png',dpi=300)
    plt.clf()


    # Stack plot of periodogram and Fourier analysis, and folded lightcurve
    plt.figure(figsize=(7,5))
    plt.subplot(3,1,1)
    plt.title(photfile.replace('photometry_','').replace('.dat',''),fontsize=8)
    
    # LS periodogram
    plt.plot(1./frequency*24., power)
    plt.ylabel('L-S Power')
    plt.xlabel('Period (hr)')

    # Fourier analysis
    plt.subplot(3,1,2)
    plt.plot(periods,chisq)
    plt.ylabel(r'Fourier fit $\chi _\nu ^2$')
    plt.xlabel('Period (hr)')

    # Phase folded lightcurve
    plt.subplot(3,1,3)
    # number of lightcurve cycles in observing window
    ncycle = math.ceil((dat['julian_date'][-1]-jd0) / (best_period/24.))
    for i in range(ncycle):
        plt.errorbar(((dat['julian_date'][dat['[8]'] == 0]-jd0)*24.-i*best_period)/best_period,
                     dat['norm_mag'][dat['[8]'] == 0],
                     yerr=dat['in_sig'][dat['[8]'] == 0],fmt='o',markersize=1.5,linestyle=' ')


    # Best fit Fourier function
    phase = np.linspace(min(time_hr),max(time_hr),10000)/best_period
    #phase = np.linspace(0,ncycle,10000)
    plt.plot(phase, fit,'k-')
    #plt.plot(phase, fit,'bo')

    plt.axis([-0.1,1.1,1.5*max(dat['norm_mag'][dat['[8]'] == 0]),1.5*min(dat['norm_mag'][dat['[8]'] == 0])])
    plt.ylabel('Diff. Mag.')
    plt.xlabel('Phase (Period = '+str('{:1.4f}'.format(best_period))+' hr)')

    plt.subplots_adjust(hspace=0.5)
    plt.savefig(tit+'_period.png',format='png',dpi=300)
    plt.clf()
    
##########################
# Lightcurve analysis
##########################
def pp_lightcurve():
    '''
    Run Lomb-Scargle periodogram and Fourier series fitting
    '''

    # Define Fourier function for fitting data
    def fourier(t, *a):
    
        # first order term
        foufunc = a[0] * np.sin(2*np.pi/period*t) + a[1] * np.cos(2*np.pi/period*t)
        
        # construct higher order terms based on number of input parameters *a
        for deg in range(1, int(len(a)/2)):
            foufunc += a[2*deg] * np.sin(2*np.pi*(deg+1)/period*t) + a[2*deg+1] * np.cos(2*np.pi*(deg+1)/period*t)

        return foufunc

    # Lomb-Scargle (LS) periodogram
    #
    ls = LombScargle(dat['julian_date'],dat['norm_mag'],dat['sig'])
    
    # Expected number of peaks in lightcurve
    peaks = 2

    # Compute LS power spectrum
    frequency, power = ls.autopower(minimum_frequency=min_freq,\
                                    maximum_frequency=max_freq,\
                                    samples_per_peak=100)
    # LS results
    best_freq = frequency[np.argmax(power)]
    ls_period = 1./best_freq*24.*peaks
    print('LS Periodgram:')
    print('   Best period = '+str(ls_period)+' hr')


    #Run chi-sq analysis
    #
    # Range of periods to scan through
    periods = np.arange(min_per,max_per,fourier_step) # [hr]
    chisq = np.array([])
    
    # Loop through range of periods
    for period in periods:
    
        # Fit for a given period
        popt, pcov = cf(fourier, time_hr, dat['norm_mag'],[1.0] * (fit_order * 2), sigma=dat['in_sig'])
        
        # degrees of freedom
        dof = len(time_hr)-len(popt)
        
        # residual and chi-squared from fit
        residual = dat['norm_mag'] - fourier(time_hr.data, *popt)
        chisq = np.append(chisq,np.sum((residual/dat['in_sig'])**2)/dof)
        
        # keep fit parameters if lowest chi-sq
        if np.sum((residual/dat['in_sig'])**2)/dof == min(chisq):
            best_popt = popt
            best_period = period
            # store high resolution fit
            fit = fourier(np.linspace(min(time_hr),max(time_hr),10000), *best_popt)

    # compute phases for best period
    phase = (time_hr % best_period) / best_period

    # Best fit solution parameters
    print('Fourier analysis:')
    print('   Fourier series order = '+str(fit_order))
    print('   Min. chi-sq = '+str(min(chisq)))
    print('   Best period = '+str(best_period)+' hr')
    
    # Write best fit parameters to file
    f = open(photfile.replace('.dat','_fitpar.txt'),'w')
    f.write(photfile+'\n')
    f.write('Initial_JD '+str(jd0)+'\n')
    f.write('Best_fit_period_hr '+str(best_period)+'\n')
    f.write('Fit_order '+str(fit_order)+'\n')
    f.writelines('%s ' % par for par in best_popt)
    f.close()

    if do_plot:
        make_plots(best_period,phase,fit,periods,chisq,power,frequency)


##########################
# Main
##########################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Lightcurve analysis of PP photometry')
    parser.add_argument('file', help='.dat file to process',nargs=1)
    parser.add_argument('fit_order', help='Order of fourier fit',type=int)
    parser.add_argument('-no_plots', default=True, action='store_false', help='Dont produce plots')
    parser.add_argument('-fourier_step', help='Period step size in Fourier analysis [hr]',default=0.001,type=float)
    parser.add_argument('-min_period', help='Minimum period to search [hr]',default=0,type=float)
    parser.add_argument('-max_period', help='Maximum period to search [hr]',default=0,type=float)
    parser.add_argument('-ignore_flag', help='Ignore SourceExtractor flag?',default=False,action='store_true')
    args = parser.parse_args()
    file = sorted(args.file)
    fit_order = args.fit_order
    do_plot = args.no_plots
    fourier_step = args.fourier_step
    min_period = args.min_period
    max_period = args.max_period
    ignore_flag = args.ignore_flag

    if len(file) > 1:
        print('Only 1 photometry file allowed. Exiting.')
        sys.exit()
    else:
        photfile = file[0]
    
    # Read photfile into astropy table
    dat = Table.read(photfile,format='ascii.commented_header')

    # First time stamp
    jd0 = min(dat['julian_date'])
    
    # relative time in hours
    time_hr = (dat['julian_date']-jd0)*24.

    # Correct for LDT shutter delay of 2.05s = 2.37e-5
    if 'DCTLMI' in dat['[9]']:
        dat['julian_date'] = dat['julian_date'] + 2.3726851851851847e-5

    # Add normalized magnitudes to table, exlude rows with SExtractor flag !=0
    dat['norm_mag'] = Column(dat['mag'] - dat['mag'][dat['[8]'] == 0].mean())

    # Max and min periods/frequencies to bound lightcurve search
    # Maximum period = 2 x minimum exposure time
    if min_period == 0:
        min_per = 2*min(dat['[5]'])/60./60. # [hr]
    else:
        min_per = min_period
    max_freq = 1. / min_per * 24. # [1/day]
    # Minimum period = 2 x length of observing sequence
    if max_period == 0:
        max_per = 2 * (max(dat['julian_date']) - min(dat['julian_date'])) * 24. # [hr]
    else:
        max_per = max_period
    min_freq = 1. / max_per * 24. # [1/day]
        
    pp_lightcurve()

