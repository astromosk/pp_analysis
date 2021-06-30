#!/usr/bin/env python
"""
Spectro-photometric analysis of data from photometry pipeline
2021-06-30, nmosko@lowell.edu
       
Parameters:
    photfiles (str): name of *.dat files

Optional Arguments:
    -ref_filt: Reference filter used to monitor lightcurve variations (str)
                If not specified, filter with most number of exposure and highest S/N
                is chosen as reference.
    -facility: Telescope/instrument string for labeling plots
    -date: Date of observation string for labeling plots
    -target: Object designation string for labeling plots

Usage:
    pp_colors.py *.dat
    
"""
import argparse
import os
import sys
import wget

import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

from astropy.time import Time
from astropy.table import Table, Column
from astropy.io import ascii
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit as cf


##########################
# Fourier series function definition
##########################
def fourier(t, *a):

    # first order term
    foufunc = a[0] * np.sin(2*np.pi/period*t) + a[1] * np.cos(2*np.pi/period*t)
    
    # construct higher order terms based on number of input parameters *a
    for deg in range(1, int(len(a)/2)):
        foufunc += a[2*deg] * np.sin(2*np.pi*(deg+1)/period*t) + a[2*deg+1] * np.cos(2*np.pi*(deg+1)/period*t)

    return foufunc


##########################
# Filter specs
##########################
def lookup_filter(filterName):
    """
    Parse filter name, return wavelength, solar mag, plot color
    """
    
    #  dictionary of supported filters: wavelengths (micron), solar mag, and plot color
    filters = {
              'B': [0.4353, 5.47, 'blue'],
              'V': [0.5477, 4.82, 'darkgreen'],
              'R': [0.6349, 4.46, 'tomato'],
              'I': [0.78, 4.12, 'black'],
              'u': [0.3551, 6.55, 'blue'],
              'g': [0.4686, 5.12, 'darkgreen'],
              'r': [0.6166, 4.68, 'tomato'],
              'i': [0.7480, 4.57, 'black'],
              'z': [0.8932, 4.54, 'magenta'],
              'SDSS-U': [0.3551, 6.55, 'blue'],
              'SDSS-G': [0.4686, 5.12, 'darkgreen'],
              'SDSS-R': [0.6166, 4.68, 'tomato'],
              'SDSS-I': [0.7480, 4.57, 'black'],
              'SDSS-Z': [0.8932, 4.54, 'magenta'],
              }

    if filterName in filters:
        return filters[filterName]
    else:
        print('Filter '+filterName+' not recognized. Exiting.')
        sys.exit()


##########################
# Read in photometry data table
##########################
def read_phot_table(file):
    """
    read into Table a photometry pipeline output file
    """

    # check for file existence
    if os.path.exists(file):
        # check file can be read as a standard PP output file
        try:
            t = Table.read(file,format='ascii.commented_header')
        except:
            print('File '+file+' not in expected format. Exiting')
            sys.exit()
    else:
        print('File '+file+' does not exist. Exiting.')
        sys.exit()
    
    return t


##########################
# Determine polynomial fit to lightcurve
##########################
def lc_polyfit(ref_mag_table,jd0,poly_n=0):
    """
    Polynomial fit to lightcurve variations in referenece filter
    """
    # setup output plot
    fig2 = plt.figure()
    plt.gca().invert_yaxis()
    time_day = ref_mag_table['julian_date']-jd0
    # plot data
    plt.errorbar(time_day,ref_mag_table['mag'],ref_mag_table['sig'],
                 marker='o',ecolor='0.7',ls='None',markersize=4)

    # If polynomial order specified, fit and plot
    if poly_n != 0:
        fit = Polynomial.fit(ref_mag_table['julian_date'], ref_mag_table['mag'], poly_n, w=1/ref_mag_table['sig']**2)
        
        plt.plot(time_day,fit(ref_mag_table['julian_date']), label='Order '+str(poly_n) + ' polynomial',linewidth=0.5)

    # If polynomial order not specified try order 1-3 to find best fit
    else:
        # Try linear fit
        fit1 = Polynomial.fit(ref_mag_table['julian_date'], ref_mag_table['mag'], 1, w=1/ref_mag_table['sig']**2)
        resid1 = ref_mag_table['mag'] - fit1(ref_mag_table['julian_date'])
        dof = len(ref_mag_table['julian_date']) - 2
        chi1 = sum((resid1/(ref_mag_table['sig']))**2)/dof

        # Try quadratic fit, if enough data points
        if len(ref_mag_table['julian_date']) > 2:
            fit2 = Polynomial.fit(ref_mag_table['julian_date'], ref_mag_table['mag'], 2, w=1/ref_mag_table['sig']**2)
            resid2 = ref_mag_table['mag'] - fit2(ref_mag_table['julian_date'])
            dof = len(ref_mag_table['julian_date']) - 3
            chi2 = sum((resid2/(ref_mag_table['sig']))**2)/dof
        else: chi2 = 1000.

        # Try cubic fit, if enough data points
        if len(ref_mag_table['julian_date']) > 3:
            fit3 = Polynomial.fit(ref_mag_table['julian_date'], ref_mag_table['mag'], 3, w=1/ref_mag_table['sig']**2)
            resid3 = ref_mag_table['mag'] - fit3(ref_mag_table['julian_date'])
            dof = len(ref_mag_table['julian_date']) - 4
            chi3 = sum((resid3/(ref_mag_table['sig']))**2)/dof
        else: chi3 = 1000.
    
        # Determine and overplot best fit: linear, quadratic or cubic
        labels = ['linear','quadratic','cubic']
        widths = [0.5,0.5,0.5]
        if chi1 < chi2 and chi1 < chi3:
            fit = fit1
            labels[0] = r'linear $\longleftarrow best\ fit$'
            widths[0] = 2
        if chi2 < chi1 and chi2 < chi3:
            fit = fit2
            labels[1] = r'quadratic $\longleftarrow best\ fit$'
            widths[1] = 2
        if chi3 < chi1 and chi3 < chi2:
            fit = fit3
            labels[2] = r'linear $\longleftarrow best\ fit$'
            widths[2] = 2
        plt.plot(time_day,fit1(ref_mag_table['julian_date']),label=labels[0],lw=widths[0])
        plt.plot(time_day,fit2(ref_mag_table['julian_date']), label=labels[1],lw=widths[1])
        plt.plot(time_day,fit3(ref_mag_table['julian_date']), label=labels[2],lw=widths[2])

    # Annotatte plot
    plt.title('Reference filter: '+ref_mag_table['[7]'][0])
    plt.xlabel('Julian Date - '+str(jd0))
    plt.ylabel('Apparent Magnitude')
    plt.legend(loc='lower center')
    fig2.savefig('lightcurve.png',format='png',dpi=300)

    return fit.coef

##########################
# Conert magnitude to reflectance
##########################
def mag_to_ref(avg_mags,avg_colors,ref_filt,ref_mag_err):
    """
    Convert measured magnitudes to normalized (@ 0.55 micron) reflectance
    """

    norm_mag = np.array([])
    norm_mag_err = np.array([])
    
    for filt in avg_mags['filter']:
        if filt == ref_filt:
            norm_mag = np.append(norm_mag,1.)
            norm_mag_err = np.append(norm_mag_err,ref_mag_err)
        else:
            for i in range(len(avg_colors['color_name'])):
                col = avg_colors['color_name'][i]
                if filt in col:
                    norm_mag = np.append(norm_mag,1.-avg_colors['color'][i])
                    norm_mag_err = np.append(norm_mag_err,avg_colors['error'][i])

    flux = 10**(-(norm_mag - avg_mags['solar_mag'])/2.5)
    norm = np.interp([0.55],avg_mags['wavelength'],flux)
    reflectance = flux / norm

    ref_error = 2.30259*flux*norm_mag_err/2.5/norm
    
    return reflectance, ref_error


##########################
# Determine best taxonomic fit
##########################
def fit_taxonomy(wav,ref,ref_err):
    """
    Determine best-fit taxonomy from Bus-Demeo templates
    """

    # retrieve taxonomic template data
    url = 'http://www2.lowell.edu/users/nmosko/busdemeo-meanspectra.csv'
    tax_file = wget.download(url)
    print('')

    tax_table = Table.read(tax_file,data_start=2,header_start=1)
    
    taxa = np.array([])
    rms = np.array([])
    
    smass_wav = tax_table['Wavelength'].data
    
    #find best fit taxonomy
    for i in range(len(tax_table.colnames)):

        col_name = tax_table.colnames[i]
        
        if 'Mean' in col_name:
            rebin_smass_ref = np.interp(wav,smass_wav,tax_table[col_name])
            taxa = np.append(taxa,col_name.split('_')[0])
            rootmeansq = np.sqrt(np.average((ref-rebin_smass_ref)**2))

            # force to taxonmic type, uncomment to activate
            #if 'C_Mean' == col_name: rootmeansq = 0.

            rms = np.append(rms,rootmeansq)
            if rootmeansq == min(rms):
                best_rms = rootmeansq
                best_taxon = col_name.split('_')[0]
                best_fit_ref = tax_table[col_name].data
                best_fit_sig = tax_table[col_name.split('_')[0]+'_Sigma'].data
                

    # redfine sigma values if undefined for taxonomy
    if (best_taxon == 'O') or (best_taxon == 'Cg') or (best_taxon == 'R'):
        best_fit_sig[:] = 0.1

    # plot results to reflectance.png
    fig3 = plt.figure()
    plt.errorbar(wav,ref,ref_err,marker='o',label='data',color='0',markerfacecolor='b')
    plt.fill(np.append(smass_wav,smass_wav[::-1]),
             np.append(best_fit_ref+best_fit_sig,best_fit_ref[::-1]-best_fit_sig[::-1]),
             '0.75',label=best_taxon+'-type')
    plt.xlim(0.4,0.95)
    if best_taxon == 'D':
        plt.ylim(0.7,1.8)
    else:
        plt.ylim(0.5,1.5)
        
    plt.figtext(0.15,0.83,'Best fit taxonomy: '+best_taxon+'-type')
    plt.xlabel('Wavelength (microns)')
    plt.ylabel('Normalized Reflectance')
    plt.title(ref_plot_title)
    plt.legend(loc='lower center')

    fig3.savefig('reflectance.png',format='png',dpi=300)

    os.remove(tax_file)

    return taxa, rms

##########################
# Main processing script
##########################
def pp_colors(filenames):
              
    # Initialize results Tables
    avg_mags = Table(names=('phot_file','filter','wavelength', \
                            'average_mag','mag_err','solar_mag','num_obs'), \
                            dtype=(object,object,float,float,float,float,int))
                        
    color_summary = Table(names=('ref_filter','ref_mag','ref_err', \
                                'filter2','f2_mag','f2_err', \
                                'f2_jd','color_name','color','color_err'), \
                                dtype=(object,float,float,object,float,float,float, \
                                        object,float,float))

    avg_colors = Table(names=('color_name','color','error'), \
                        dtype=(object,float,float))
                          
    taxon_results = Table()
    
    # Setup magnitudes vs. time plot
    fig1 = plt.figure()
    plt.gca().invert_yaxis()

    # Loop through each input file to summarize results in Table avg_mags
    # and start to populate color_summary table
    #
    print('Reading photometry files & plotting time series photometry...')
    for i in range(len(filenames)):

        # Create empty row in avg_mag Table
        avg_mags.add_row(['','',0,0,0,0,0])
        
        # Photometry file name
        file = filenames[i]
        avg_mags['phot_file'][i] = file
        
        # mag_table read from photometry file
        mag_table = read_phot_table(file)
              
        # Retrieve filter information
        filt_name = mag_table['[7]'][0]
        filter_info = lookup_filter(filt_name)
        avg_mags['filter'][i] = filt_name
        avg_mags['wavelength'][i] = filter_info[0]
        avg_mags['solar_mag'][i] = filter_info[1]
              
        # Plot mags vs time
        plt.errorbar(mag_table['julian_date'],mag_table['mag'],mag_table['sig'],
                     color=filter_info[2],ecolor='0.7',linestyle='--',
                     marker='x',label=filt_name)
              
        # Compute weighted average mag, error, number of observations
        avmag = np.round(np.average(mag_table['mag'],weights=1/mag_table['sig']**2),4)
        averr = np.round(np.average(mag_table['sig']),4)
        avg_mags['average_mag'][i], avg_mags['mag_err'][i] = avmag, averr
        avg_mags['num_obs'][i] = len(mag_table['mag'])
    
        # Put all magnitudes in color_summary table
        for i in range(len(mag_table['mag'])):
            color_summary.add_row(['',0,0, \
                                mag_table['[7]'][i],mag_table['mag'][i], \
                                mag_table['sig'][i],mag_table['julian_date'][i],'',0,0])

    # Annotate and write time series plot of mags vs time
    plt.title('Time series photometry')
    plt.xlabel('Julian Date')
    plt.ylabel('Apparent Magnitude')
    plt.legend()
    fig1.savefig('timeSeries.png',format='png',dpi=300)

    # Determine reference filter for tracking lightcurve based on
    # band with most observations + highest S/N or user specification
    #
    print('Determine reference filter...')
    if reference_filter == '':
        mask = avg_mags['num_obs'] == max(avg_mags['num_obs'])
        if sum(mask) > 1:
            print('More than one filter has '+str(max(avg_mags['num_obs']))+' exposures:')
            print('   '+avg_mags[mask]['filter'])
            print('Using filter with lowest error as reference:')
    
            mask = avg_mags['mag_err'] == min(avg_mags['mag_err'])
            print('   '+avg_mags[mask]['filter'])

            if sum(mask) > 1:
                print('Cant determine reference filter.')
                print('Multiple filters have equal number of exposures.')
                print('Multiple filters have the same average error.')
                print('Exiting.')
                sys.exit()
        else:
            print('   Using filter with most observations as reference: '+avg_mags[mask]['filter'][0])

    else:
        if reference_filter in avg_mags['filter']:
            print('   Using user-specified filter as reference: '+reference_filter)
            mask = avg_mags['filter'] == reference_filter
        else:
            print('Filter '+str(reference_filter)+' not recognized in dataset.')
            print('Exiting.')
            sys.exit()

    # Record reference filter in color_summary Table
    ref_filt = avg_mags[mask]['filter']
    color_summary['ref_filter'] = ref_filt
    color_summary['color_name'] = color_summary['ref_filter']+'-'+color_summary['filter2']

    # Remove rows in color_summary Table where ref_filter == filter
    mask2 = color_summary['ref_filter'] != color_summary['filter2']
    color_summary = color_summary[mask2]

    # Calculate reference magnitudes based on lightcurve fit
    #
    # Lightcurve correction based on stored Fourier fit
    if lc_fit_file != '':
        # read in fit file
        # compute fourier series
        # compute reference mags at f2_jd in color_summary
        a = 1
    # Lightcurve correction based on polynomial fit
    else:
        # Reference magnitudes and first time step
        ref_mag_table = read_phot_table(avg_mags[mask]['phot_file'][0])
        jd0 = ref_mag_table['julian_date'][0]
        
        # Best fit polynomial coefficients and function
        coeff = lc_polyfit(ref_mag_table,jd0,poly_n)
        best_fit_poly = Polynomial(coeff)
        
        # Reference magnitide calculated at times of other exposures
        ref_mag_calc = np.round(best_fit_poly(color_summary['f2_jd']-jd0),4)
        print(best_fit_poly(color_summary['f2_jd']-jd0))

    # Lightcurve corrected reference magnitudes
    color_summary['ref_mag'] = ref_mag_calc

    # error on computed reference filter magnitude is standard error on mean of all measurements
    # this method may not be ideal because it doesnt deal with lightcurve variability
    #color_summary['ref_err'] = np.round(np.std(ref_mag_table['mag'])/len(ref_mag_table),4)
    #
    # error on computed reference filter magnitude is mean error of all measurements
    #color_summary['ref_err'] = np.round(np.average(ref_mag_table['sig']),4)
    #
    # error on computed reference filter magnitude is standard deviation on difference between
    # computed magnitudes (ref_mag_calc) and measured mags (ref_mag_table)
    ref_mag_residuals = ref_mag_table['mag'] - best_fit_poly(ref_mag_table['julian_date']-jd0)
    color_summary['ref_err'] = np.round(np.std(ref_mag_residuals),4)



    # compute color relative to computed reference magnitude for each frame
    color_summary['color'] = np.round(color_summary['ref_mag'] - color_summary['f2_mag'],4)

    # error on color is error of magnitudes addedd in quadrature
    color_summary['color_err'] = np.round(np.sqrt(color_summary['ref_err']**2 +
                                           color_summary['f2_err']**2),4)


    #compute average values for each color across all images
    #
    unique_colors = np.unique(color_summary['color_name'])
    for i in range(len(unique_colors)):
        mask3 = color_summary['color_name'] == unique_colors[i]
        num_col = len(color_summary[mask3]['color'])
        
        # weighted average color across all images
        weight_vals = 1/color_summary[mask3]['color_err']**2
        avg_col = np.round(np.average(color_summary[mask3]['color'], \
        weights=weight_vals),4)
        
        # average color error = standard deviation of all color values if N > 1
        #   alternative could be:
        #   average error = standard deviation of colors / sqrt(N)
        # else error is reported as just the error for the single color
        if num_col > 1:
            avg_col_err = np.round(np.std(color_summary[mask3]['color']),4)
            #avg_col_err = np.sqrt(1/np.sum(1/color_summary[mask3]['color_err']**2))
            #avg_col_err = np.round(np.std(color_summary[mask3]['color'])/np.sqrt(num_col),4)
        else:
            avg_col_err = np.round(color_summary[mask3]['color_err'],4)

        avg_colors.add_row([unique_colors[i],avg_col,avg_col_err])

    # compute normalized reflectance and associated errors
    avg_mags.sort('wavelength')
    ref, ref_err = mag_to_ref(avg_mags,avg_colors,ref_filt,color_summary['ref_err'][0])
    
    # find best fit SMASS taxonomy
    print('Find taxonomic type...')
    wav = avg_mags['wavelength']
    taxa, rms = fit_taxonomy(wav,ref,ref_err)
    taxon_results.add_column(Column(taxa,name='taxon'))
    taxon_results.add_column(Column(rms,name='rms'))
    taxon_results.sort('rms')
    print('   Best fit type: '+taxon_results['taxon'][0])

    # write results to file
    with open('resultSummary.txt', mode='w') as f:
        
        f.write('AVERAGE MAGNITUDES:\n')
        avg_mags.write(f,format='ascii.fixed_width')

    with open('resultSummary.txt', mode='a') as f:

        f.write('\n')
        f.write('AVERAGE COLORS:\n')
        f.seek(0, os.SEEK_END)
        avg_colors.write(f,format='ascii.fixed_width')
        
        f.write('\n')
        f.write('TAXONOMIC FITS (sorted by RMS):\n')
        taxon_results.write(f,format='ascii.fixed_width')

        f.write('\n')
        f.write('SUMMARY OF COLOR DATA:\n')
        color_summary.write(f,format='ascii.fixed_width')
    print('Results written to resultSummary.txt')

##########################
# Main
##########################
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Photometry pipeline data files.')
    parser.add_argument('files', help='*.dat files to process',nargs='+')
    parser.add_argument('-lc_fit', help='Lightcurve fit parameter file',default='')
    parser.add_argument('-poly_n', help='Polynomial order to fit lightcurve',default=0, type=int)
    parser.add_argument('-ref_filt', help='Reference filter, options: <B|V|R|I|u|g|r|i|z|SDSS-U|SDSS_G|SDSS-R|SDSS-I|SDSS-Z>',default='')
    parser.add_argument('-facility', help='Telescope/instrument (str)',
                        default='')
    parser.add_argument('-date', help='Date of observations (str)',default='')
    parser.add_argument('-target', help='Object designation (str)',default='---')

    args = parser.parse_args()
    filenames = sorted(args.files)
    lc_fit_file = args.lc_fit
    poly_n = args.poly_n
    facility = args.facility
    date_obs = args.date
    target = args.target
    reference_filter = args.ref_filt

    # check for sufficient number of photometry files
    if len(filenames) < 3:
        print('Only '+str(len(filenames))+' photometry files found.')
        print('At least 3 required. Exiting.')
        sys.exit()

    # check for lightcurve fit parameter file if specified
    if lc_fit_file != '':
        if not os.path.exists(lc_fit_file):
            print('Lightcurve fit parmeter file not found: '+lc_fit_file)
            print('Exiting')
            sys.exit()

    # retrive facility and date info from data file
    dat = read_phot_table(filenames[0])
    if facility == '':
        facility = dat['[9]'][0]
    if date_obs == '':
        jd_obs = Time(dat['julian_date'][0],format='jd')
        date_obs = jd_obs.iso.split()[0]
        
    # construct plot title string
    ref_plot_title = facility+';  '+date_obs+';  '+target

    pp_colors(filenames)
