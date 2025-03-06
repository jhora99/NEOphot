#!/usr/bin/env python
# coding: utf-8

# # NEOphot_funcs: Routines used by NEOphot
#     to extract NEO photometry from Spitzer data
# 
# ## Joseph Hora
# ### Center for Astrophysics | Harvard & Smithsonian
#  2024/07/15

# Import all necessary packages and functions

import numpy as np
import numpy.ma as ma
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.text import Text

plt.rcParams.update({'figure.max_open_warning': 0})

import glob
import re
import os
import sys
import shutil

# from astropy.utils.data import download_file
from astropy.io import fits
from matplotlib.colors import LogNorm
from astropy.io.fits import getval
from astropy.io.fits import getheader
from astropy.io.fits import getdata
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.utils.exceptions import AstropyWarning
from astropy import units as u
from astropy.coordinates import SkyCoord

# photometry functions
from photutils.centroids import centroid_com
from image_registration import chi2_shift

# For mosaic construction
from reproject import reproject_interp
from reproject import reproject_adaptive
from reproject import reproject_exact
from reproject.mosaicking import reproject_and_coadd
from reproject.mosaicking import find_optimal_celestial_wcs
from astropy.stats import sigma_clip
from astropy.nddata import Cutout2D
from skimage.util.shape import view_as_windows

# Photometry
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from photutils import datasets
from photutils import DAOStarFinder,aperture_photometry
from photutils import CircularAperture
from photutils import CircularAnnulus
from photutils.centroids import centroid_sources
from photutils.utils import calc_total_error

import requests
from scipy import stats
from tempfile import mkstemp
import warnings
from datetime import datetime



# Define new header using the wcs, then add all header lines from the ref_hdr that are
# not already present in the new header (to avoid copying WCS header keywords and values)

def copy_headerlines(ref_hdr, wcs):
    header = wcs.to_header()
    for fitskeywd in ref_hdr:
        found = False
        for wcsfitskeywd in header:
            if wcsfitskeywd == fitskeywd:
                found = True
        if not found:                      # If the keyword is not already in header, add it
            if fitskeywd == "COMMENT":                # The comment and history keywords have 
                commenttext = ref_hdr[fitskeywd]      # multi-line values, so must add them
                for line in commenttext:              # line-by-line  
                    header.add_comment(line)
            elif fitskeywd == "HISTORY":
                commenttext = ref_hdr[fitskeywd]
                for line in commenttext:
                    header.add_history(line)
            elif fitskeywd == '':
                commenttext = header.append(('', ref_hdr.comments['']), end=True)
            elif not (fitskeywd[0:2] == 'A_' or fitskeywd[0:2] == 'B_' 
                      or fitskeywd[0:2] == 'C_' or fitskeywd[0:3] == 'AP_' 
                      or fitskeywd[0:3] == 'BP_'):
                header.append((fitskeywd, ref_hdr[fitskeywd], ref_hdr.comments[fitskeywd]),
                              end=True)
    return header


# Subtract the sky mosaic from the input filelist, which have already been projected to the 
# same WCS as the sky mosaic. Frames that do not contain the NEO are skipped, and the list of 
# subtracted NEO frames is returned

def subframes(filelist, mosaicname, wcs_out, shape_out, objectname, versionnum, versiondate, useall=False):
    """
    
    """
    sublist = []
    mosaic = getdata(mosaicname)

    for image in filelist:
        hdu_list = fits.open(image)
        header = hdu_list[0].header
        if (header['NEOFRAME'] == 1) or useall:
            sourceframe = hdu_list[0].data
            diffimage = sourceframe - mosaic

            sline = 'SKY IMAGE CREATED BY NEOPhot ver. ' + str("%8.3f " % versionnum)+ versiondate
            header.add_history(sline)
            header.add_history('NON-SIDEREAL TRACKING USING' + objectname + ' RATES AT OBS. TIME')
            header.add_history('STARS ARE TRAILED BASED ON NON-SIDEREAL RATES OVER THE FRAME TIME')
            header.add_history('THIS FRAME CONSTRUCTED BY PROJECTING THE BCD FRAME TO THE WCS OF')
            header.add_history('THE SKY FRAME AND THEN SUBTRACTING THE SKY FRAME')

            hdum = fits.PrimaryHDU(diffimage, header)
            hdul = fits.HDUList([hdum])
            outname = image[:image.rfind('.fits')] + '_sub.fits'
            hdul.writeto(outname, overwrite=True)
            sublist.append(outname)
        hdu_list.close()
    return sublist


def calculate_darkframe(framelist, darkfilename, versionnum, versiondate):
    image_concat = []
    for image in framelist:
        sourceframe = getdata(image)
        sourceframe = sourceframe - get_median_limited(sourceframe, -0.5, 0.9)
        sourceframe = np.where(sourceframe>0.1, np.nan, sourceframe)
        image_concat.append(sourceframe)
        np.seterr(invalid='ignore')

    stacked_array = np.ma.stack(image_concat)
    med_image = np.nanmedian(stacked_array, axis=0)
    header = getheader(framelist[0])
    sline = 'DARK IMAGE CREATED BY NEOPhot ver. ' + str("%8.3f " % versionnum)+ versiondate
    header.add_history(sline)
    header.add_history('THIS IS THE DARK FRAME CONSTRUCTED FROM MEDIAN OF ALL FRAMES IN AOR')
    header.add_history('HEADER DATA ABOVE IS FROM FIRST LONG FRAME IN AOR SET')
    hdum = fits.PrimaryHDU(med_image, header)
    hdul = fits.HDUList([hdum])
    hdul.writeto(darkfilename, overwrite=True)
    hdul.close()
    
    return med_image

# find median of image subject to the min and max limits given

def get_median_limited(image, minlimit, maxlimit):
    maskimage = np.where(image>maxlimit, np.nan, image)
    maskimage = np.where(maskimage<minlimit, np.nan, maskimage)
    median_estimate = np.nanmedian(maskimage)
    return median_estimate

# Put together the sky mosaic (in stationary frame, but all of the observations are
# performed at the non-siderial rate of the NEO, so they will be trailed).

def make_skymosaic(framelist, mradius, mshift, outname, outname_coadd, refnum, minstd, 
                   wcs_out, shape_out, darkframe, sub_dark, use_mask, BCDsfx, 
                   versionnum, versiondate, writemsk=False):
    refimage = framelist[refnum]
    ref_hdr = getheader(refimage)
    refRApos = ref_hdr['RA_REF']
    refDEpos = ref_hdr['DEC_REF']
    objectname = ref_hdr['OBJECT']
    image_concat = []
    fprint_concat = []
    projlist = []
    
    for image in framelist:
        hdu_list = fits.open(image)
        sourceframe = hdu_list[0].data
        if sub_dark:
            sourceframe = sourceframe - darkframe
        header = hdu_list[0].header
        rapos = header['RA_REF']
        depos = header['DEC_REF']
        wtmp = WCS(header)
        neoxpos, neoypos = wtmp.wcs_world2pix(rapos, depos, 1)
        if (neoxpos>0) and (neoxpos<sourceframe.shape[0]) and (neoypos>0) and (neoypos<sourceframe.shape[1]):
            onframe = True
            header['NEOFRAME'] = (1, 'FRAME CONTAINS NEO')
        else:
            onframe = False
            header['NEOFRAME'] = (0, 'FRAME DOES NOT CONTAIN NEO')
        if use_mask:
            maskname = image[:image.rfind(BCDsfx)]+'bimsk.fits'        # mask bad pixels
            maskdata = getdata(maskname)
            if writemsk:
                hdum = fits.PrimaryHDU(maskdata, header=header)      # debug: write mask out
                hdul = fits.HDUList([hdum])
                hdul.writeto(image[:image.rfind(BCDsfx)]+'msktst.fits', overwrite=True)
            sourceframe = np.where(maskdata!=18432, sourceframe, np.nan)
        sourceframe = sourceframe - get_median_limited(sourceframe, -1.0, 2.0)
        hdu_list[0].data = sourceframe
        array, footprint = reproject_exact(hdu_list, wcs_out, shape_out=shape_out, parallel=True)
        newheader = copy_headerlines(header, wcs_out)
        hdum = fits.PrimaryHDU(array, newheader)        # write out reprojected BCD
        hdul = fits.HDUList([hdum])
        outfile = image[:image.rfind(BCDsfx)]+'rpro.fits'
        hdul.writeto(outfile, overwrite=True)
        projlist.append(outfile)
        # Mask out the region around the NEO in reprojected array
        neoxpos, neoypos = wcs_out.wcs_world2pix(rapos, depos, 1)
        if onframe and (mradius>0) and (neoxpos>0) and (neoypos>0):
            for i in range(int(neoxpos-mradius-mshift[0]), int(neoxpos+mradius-mshift[0])):
                for j in range(int(neoypos-mradius-mshift[1]), int(neoypos+mradius-mshift[1])):
                    if math.sqrt((i-neoxpos-mshift[0])**2 + (j-neoypos-mshift[1])**2)<=mradius:
                        array[j][i] = np.nan
        image_concat.append(array)
        np.seterr(invalid='ignore')
        farray = np.divide(array, array)        # make array with 1=good, nan for bad pixels
        fprint_concat.append(farray)
        hdu_list.close()

    stacked_array = np.ma.stack(image_concat)
    stacked_fprint = np.ma.stack(fprint_concat)
    coadd_image = np.nansum(stacked_fprint, axis=0)
    filtered_data = sigma_clip(stacked_array, axis=0, sigma=2.5, masked=False, copy=False)
                                #cenfunc=mean)#, masked=False, copy=False)
    med_image = np.nanmedian(filtered_data, axis=0)
    
    window_shape = (5, 5)
    B = view_as_windows(med_image, window_shape)
    Bstd = np.nanstd(B, axis=(2,3))
    Bstdpd = np.pad(Bstd, 2)
    std_image = np.where(Bstdpd>minstd, Bstdpd, minstd)

    newcutlist = []
    fprint_concat = []
    for imdata in image_concat:
        imdata = imdata - np.nanmedian(imdata)
        imdata = np.where(abs(imdata - med_image)>std_image*3, np.nan, imdata)
        newcutlist.append(imdata)
        goodpix = np.divide(imdata, imdata)
        fprint_concat.append(goodpix)
    stacked_array = np.ma.stack(newcutlist)
    med_image = np.nanmean(stacked_array, axis=0)
    stacked_coadds = np.ma.stack(fprint_concat)
    coadd_image = np.nansum(stacked_coadds, axis=0)
    
    med_image = med_image - get_median_limited(med_image, -1.0, 1.0)
    med_image = np.where(coadd_image<1, np.nan, med_image)   

    header = copy_headerlines(ref_hdr, wcs_out)

    header.append(('NMOSFRAM', len(framelist),'Number of frames used to make mosaic'))
    header.add_history('SKY IMAGE CREATED BY NEOPhot ver. ' + str("%8.3f " % versionnum)+ versiondate)
    header.add_history('NON-SIDEREAL TRACKING USING ' + objectname + ' RATES AT OBS. TIME')
    header.add_history('STARS ARE TRAILED BASED ON NON-SIDEREAL RATES OVER THE FRAME TIME')
    
    hdum = fits.PrimaryHDU(med_image, header=header)
    hdul = fits.HDUList([hdum])
    hdul.writeto(outname, overwrite=True)

    stdheader = header.copy()
    outname_std = outname[:outname.rfind('.fits')] + '_std.fits'
    stdheader.add_history('THIS IS THE STANDARD DEVIATION IMAGE USED IN MOSAIC CONSTRUCTION')
    hdum = fits.PrimaryHDU(std_image, header=stdheader)
    hdul = fits.HDUList([hdum])
    hdul.writeto(outname_std, overwrite=True)

    header.add_history('THIS IS THE COADD IMAGE, VALUES REPRESENT THE NUMBER OF VALID')
    header.add_history('IMAGE PIXELS AT THAT SKY LOCATION USED IN THE MOSAIC, 0 to NFRAMES')
    hdum = fits.PrimaryHDU(coadd_image.data, header=header)
    hdul = fits.HDUList([hdum])

    hdul.writeto(outname_coadd, overwrite=True)

    return projlist


def make_NEOmosaic(subframelist, outname, refnum, minstd, recenter, xshift, yshift,
                   logfile, moswl, objectname, versionnum, versiondate, debug=False):
    cutlist = []
    NEOlist = []
    fprint_concat = []
    moscen = int(moswl/2)+1
    size = (moswl, moswl)
    cenwl = 19                  # size of centroiding recentering window; make this an odd number
    cencen = int(cenwl/2)+1
    cenwinsize = (cenwl, cenwl)
    refRApos = getval(subframelist[refnum],'RA_REF')
    refDEpos = getval(subframelist[refnum],'DEC_REF')
    ch = getval(subframelist[refnum],'CHNLNUM')
    ref_hdr = getheader(subframelist[refnum])

    for image in subframelist:
        header = getheader(image)
        RApos = header['RA_REF']
        DEpos = header['DEC_REF']
        wtmp = WCS(header)
        imdata = getdata(image)
        xpos, ypos = wtmp.wcs_world2pix(RApos, DEpos, 1)
        position = ((xpos+xshift), (ypos+yshift))
        imcut = Cutout2D(imdata, position=position, size=cenwinsize, wcs=wtmp)
        cutframe = imcut.data
        if (not np.nanmean(cutframe)==0.0) and (not np.nanmean(cutframe)==np.nan):
            xref1, yref1 = centroid_sources(cutframe, cencen, cencen, box_size = cenwl)
            xwoff = xref1[0]-cencen
            ywoff = yref1[0]-cencen
            if recenter:
                position = ((xpos+xwoff+xshift), (ypos+ywoff+yshift))
                print("recenter shift: ",str("%5.3f, " % xwoff), str("%5.3f" % ywoff))
            imcut = Cutout2D(imdata, position=position, size=size, wcs=wtmp)
            cutframe = imcut.data
            if (cutframe.shape[0]==moswl) and (cutframe.shape[1]==moswl):
                cutframe = cutframe - get_median_limited(cutframe, -1.0, 1.0)
                cutframe = np.where(cutframe>(-minstd*3), cutframe, np.nan)
                cutlist.append(cutframe)
                NEOlist.append(image)
                np.seterr(invalid='ignore')
                goodpix = np.divide(cutframe, cutframe)
                fprint_concat.append(goodpix)
                if debug:
                    print('Relative offset of NEO (arcsec): ',
                          str("%7.3f" % ((RApos-refRApos)*3600*np.cos(DEpos/57.29578))),
                          str("%7.3f" % ((DEpos-refDEpos)*3600)), ' median value: ',
                          str("%8.5f" % np.nanmedian(cutframe)))
                if debug:
                    tmphdum =fits.PrimaryHDU(cutframe, header=ref_hdr)   # debug: write out subframes
                    hdul = fits.HDUList([tmphdum])
                    hdul.writeto(impath+str("I%1i" % ch) + 'cutim' +
                                 str("%02i" % subframelist.index(image)+'.fits'), overwrite=True)
    print(len(cutlist), ' frames to be coadded')
    logfile.write(str('\n%3i ' % (len(cutlist)))+'frames to be coadded for '+outname+'\n')
    stacked_array = np.ma.stack(cutlist)
    stacked_coadds = np.ma.stack(fprint_concat)
    filtered_data = sigma_clip(stacked_array, axis=0, sigma=3)#, maxiters=None,
                                #cenfunc=mean)#, masked=False, copy=False)
    med_image = np.nanmedian(filtered_data, axis=0)
    coadd_image = np.nansum(stacked_coadds, axis=0)

    med_image = med_image - get_median_limited(med_image, -1.0, 1.0)
  
    wsize = 5
    window_shape = (wsize, wsize)
    B = view_as_windows(med_image, window_shape)
    Bstd = np.nanstd(B, axis=(2,3))
    Bstdpd = np.pad(Bstd, int(wsize/2))
    std_image = np.where(Bstdpd>minstd, Bstdpd, minstd)
    
    newcutlist = []
    fprint_concat = []
    for imdata in cutlist:
        imdata = imdata - np.nanmedian(imdata)
        imdata = np.where(abs(imdata - med_image)>std_image*5, np.nan, imdata)
        newcutlist.append(imdata)
        goodpix = np.divide(imdata, imdata)
        fprint_concat.append(goodpix)
    stacked_array = np.ma.stack(newcutlist)
    med_image = np.nanmean(stacked_array, axis=0)
    stacked_coadds = np.ma.stack(fprint_concat)
    coadd_image = np.nansum(stacked_coadds, axis=0)

    refnum = int(len(NEOlist)/2)+1
    refimage = NEOlist[refnum]
    ref_hdr = getheader(refimage)
    refimdata = getdata(refimage)
    refRApos = ref_hdr['RA_REF']
    refDEpos = ref_hdr['DEC_REF']
    print(refRApos, refDEpos)
    w = WCS(ref_hdr)
    xpos, ypos = wtmp.wcs_world2pix(refRApos, refDEpos, 1)
    position = (xpos, ypos)
    refcut = Cutout2D(refimdata, position=position, size=size, wcs=w)
    
    ref_hdr.update(refcut.wcs.to_header())    
    
    ref_hdr.append(('NMOSFRAM', len(cutlist),'Number of frames used to make mosaic'))
    sline = 'NEO IMAGE CREATED BY NEOPhot ver. ' + str("%8.3f " % versionnum)+ versiondate
    ref_hdr.add_history(sline)
    ref_hdr.add_history('NON-SIDEREAL TRACKING USING' + objectname + ' RATES AT OBS. TIME')
    ref_hdr.add_history('STARS ARE TRAILED BASED ON NON-SIDEREAL RATES OVER THE FRAME TIME')
    ref_hdr.add_history('NEO IMAGE CONSTRUCTED FROM SKY-SUBTRACTED BCD FRAMES. THIS IMAGE')
    ref_hdr.add_history('IS A CUTOUT CENTERED ON THE NEO POSITION')

    hdum = fits.PrimaryHDU(med_image.data, header=ref_hdr)
    hdul = fits.HDUList([hdum])
    hdul.writeto(outname, overwrite=True)

    coadd_hdr = ref_hdr.copy()
    coadd_hdr.add_history('THIS IS THE COADD IMAGE, VALUES REPRESENT THE NUMBER OF VALID')
    coadd_hdr.add_history('IMAGE PIXELS AT THAT SKY LOCATION USED IN THE MOSAIC, 0 to NFRAMES')

    outname_coadd = outname[:outname.rfind('.fits')] + '_coadd.fits'
    hdum = fits.PrimaryHDU(coadd_image.data, header=coadd_hdr)
    hdul = fits.HDUList([hdum])
    hdul.writeto(outname_coadd, overwrite=True)
    
    std_hdr = ref_hdr.copy()
    std_hdr.add_history('THIS IS THE STANDARD DEVIATION IMAGE, THE VALUES ARE THE STANDARD')
    std_hdr.add_history('DEVIATION OF A 5X5 PIXEL REGION AROUND EACH PIXEL. THIS WAS USED')
    std_hdr.add_history('TO SET THE REJECTION CUTOFF SIGMA IN THE NEO MOSAIC CONSTRUCTION')
    outname_coadd = outname[:outname.rfind('.fits')] + '_std.fits'
    hdum = fits.PrimaryHDU(std_image, header=std_hdr)
    hdul = fits.HDUList([hdum])
    hdul.writeto(outname_coadd, overwrite=True)
    
    return med_image, NEOlist
 


# perform photometry on the NEO mosaic

def get_mosaicphot(imagename, moswl):
    image_data = getdata(imagename)
    header = getheader(imagename)
    wmos = WCS(header)
    ch = header['CHNLNUM']
    moscen = int(moswl/2)+1
    # xref, yref = centroid_sources(image_data, pts[0][0],  pts[0][1], box_size = 9)
    xref1, yref1 = centroid_sources(image_data, moscen, moscen, box_size = 21)
    positions = (xref1[0], yref1[0])
    print("Positions: ",positions)
    xref, yref = centroid_sources(image_data, xref1[0], yref1[0], box_size = 7)
    positions = (xref[0], yref[0])
    rapos, decpos = wmos.wcs_pix2world(xref, yref, 1)

    apertures = CircularAperture(positions, r=8)
    annulus_apertures = CircularAnnulus(positions, r_in=12, r_out=16)
    #    apers = [apertures, annulus_apertures]
    #    phot_table = aperture_photometry(image_data, apers)

    mask = annulus_apertures.to_mask(method='center')

    bkg_median = []
    annulus_data = mask.multiply(image_data)
    annulus_data_1d = annulus_data[mask.data > 0]
    _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
    bkg_median.append(median_sigclip)
    bkg_median = np.array(bkg_median)
    effective_gain = header['GAIN']/header['FLUXCONV']*header['NMOSFRAM']
    bkg_error = np.full_like(image_data, header['ZODY_EST'])
    terror = calc_total_error(image_data, bkg_error, effective_gain)#/ma.sqrt(header['NMOSFRAM'])  
    phot = aperture_photometry(image_data, apertures, error=terror)
    phot['annulus_median'] = bkg_median
    phot['aper_bkg'] = bkg_median * apertures.area
    phot['aper_sum_bkgsub'] = phot['aperture_sum'] - phot['aper_bkg']


    for col in phot.colnames:
        phot[col].info.format = '%.8g'  # for consistent table output
    print(phot)

    return ADU2uJy(ch, phot['aper_sum_bkgsub'][0]), ADU2uJy(ch,phot['aperture_sum_err'][0]), rapos, decpos


# Perform photometry on individual BCD frames

def get_BCDphot(imagelist, firstboxsize, offset, do_mult_mos, impath, objectname, mosnum, logfile):
    printinfo = False
    xoffsets = []
    yoffsets = []
    raoffsets= []
    deoffsets= []
    photlist = []
    times = []
    cuttimes = []
    cutouts = []
    goodimagelist = []
    cutoutsize = (17,17)
    # Make initial pass through frames, determine median X, Y offset to recenter on the target
    for image in imagelist:                             
        image_data = getdata(image)
        header = getheader(image)
        RApos = header['RA_REF']
        DEpos = header['DEC_REF']
        itime = header['FRAMTIME']
        otime = header['MJD_OBS']
        if imagelist.index(image) == 0:
            time_ref = header['MJD_OBS']
        ch = header['CHNLNUM']
        wtmp = WCS(header)
        xposi, yposi = wtmp.wcs_world2pix(RApos, DEpos, 1)
        xref1, yref1 = centroid_sources(image_data, (xposi+offset[0]), (yposi+offset[1]), box_size = firstboxsize)
        positions = (xref1[0], yref1[0])
        if printinfo:
            print("ref pos:", xposi, yposi, positions)
        if positions_OK(positions, image_data):
            xref, yref = centroid_sources(image_data, xref1[0], yref1[0], box_size = 11)
            positions = (xref[0], yref[0])
            if positions_OK(positions,image_data):           # If bad positions from centroid, do not add to good list
                raoff, decoff = wtmp.wcs_pix2world(xref[0], yref[0],1)
                raoffsets.append(raoff)
                deoffsets.append(decoff)
                times.append((otime - time_ref)*24*3600)
                goodimagelist.append(image)
                xoffsets.append(xref[0]-xposi)
                yoffsets.append(yref[0]-yposi)
                if printinfo:
                    print("Frame: ", image[(image.rfind('\\')+1):], " Offset: ",
                          str("%6.2f, " % (xref[0]-xposi)), str("%6.2f" % (yref[0]-yposi)))

    xshift = np.median(xoffsets)
    yshift = np.median(yoffsets)
    cutouts=[]
    explist = []
    
    # Do cutouts on the median X and Y offsets
    for image in goodimagelist:
        image_data = getdata(image)
        header = getheader(image)
        RApos = header['RA_REF']
        DEpos = header['DEC_REF']
        expid = header['EXPID']
        wtmp = WCS(header)
        xposi, yposi = wtmp.wcs_world2pix(RApos, DEpos, 1)
        xref1 = xposi + xshift
        yref1 = yposi + yshift
        positions = (xref1, yref1)
        cutoutframe = Cutout2D(image_data, position=positions, size=cutoutsize)
        cutouts.append(cutoutframe.data)        
        explist.append(expid)
 
    # Plot thumbnails of regions around the NEO to show possible issues
    columns = 7                                     # Number of cutouts across page
    rows = int((len(cutouts)-1)/columns) + 1        # Calculate the number of rows necessary to
    fig = plt.figure(figsize=(16, rows*2.5))        #  display all cutouts
    ax = []
    for i in range(len(cutouts)):
        img = np.real(cutouts[i])
        ax.append(fig.add_subplot(rows, columns, i+1) )
        ax[-1].tick_params(axis='x', labelsize=10)
        ax[-1].tick_params(axis='y', labelsize=10)
        titlestr = str("%2i," % explist[i]) + str("T:%5.1f" % times[i])
        ax[-1].set_title(titlestr)  # set title
        imlabel = str("(%4.2f," % xoffsets[i])+str("%4.2f)" % yoffsets[i])
        plt.imshow(img, cmap='gray',origin='lower')    #  norm=LogNorm(), 
        ax[-1].text(3, 15, imlabel, color="lime")
    if do_mult_mos:
        plt.savefig(impath + objectname + str("_BCDcut_I%1i_" % ch) + str("%1i.pdf" % mosnum))
    else:
        plt.savefig(impath + objectname + str("_BCDcut_I%1i.pdf" % ch))
    plt.show()
    
    print(str("Ch%1i median shift: " % ch)+str("%6.2f, " % xshift)+str("%6.2f " % yshift))
    logfile.write(str("Ch%1i median shift: " % ch)+str("%6.2f, " % xshift)+str("%6.2f\n" % yshift))

    fxoff = []
    fyoff = []
    fluxes = []
    goodtimes = []
    for image in goodimagelist:                   # perform photometry on all good frames in list
        image_data = getdata(image)
        header = getheader(image)
        RApos = header['RA_REF']
        DEpos = header['DEC_REF']

        wtmp = WCS(header)        #  UTC at Spitzer, add 1/2 FRAMTIME to get time in middle 
        UT = header['MJD_OBS'] + (itime/2)/(3600*24)                          # of exposure
        xposi, yposi = wtmp.wcs_world2pix(RApos, DEpos, 1)
        xposi = xposi + xshift
        yposi = yposi + yshift
        xref1, yref1 = centroid_sources(image_data, xposi, yposi, box_size = 7)
        raref, decref = wtmp.wcs_pix2world(xref1, yref1, 1)
        positions = (xref1[0], yref1[0])
        if positions_OK(positions,image_data):
            apertures = CircularAperture(positions, r=8)
            annulus_apertures = CircularAnnulus(positions, r_in=12, r_out=16)
            #    apers = [apertures, annulus_apertures]
            #    phot_table = aperture_photometry(image_data, apers)

            mask = annulus_apertures.to_mask(method='center')

            bkg_median = []
            annulus_data = mask.multiply(image_data)
            annulus_data_1d = annulus_data[mask.data > 0]
            _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
            bkg_median.append(median_sigclip)
            bkg_median = np.array(bkg_median)
            effective_gain = header['GAIN']/header['FLUXCONV']*header['FRAMTIME']
            bkg_error = np.full_like(image_data, header['ZODY_EST'])
            terror = calc_total_error(image_data, bkg_error, effective_gain)  
            phot = aperture_photometry(image_data, apertures, error=terror)
            phot['annulus_median'] = bkg_median
            phot['aper_bkg'] = bkg_median * apertures.area
            phot['aper_sum_bkgsub'] = phot['aperture_sum'] - phot['aper_bkg']
            flux = ADU2uJy(ch, phot['aper_sum_bkgsub'][0])
            fluxerr = ADU2uJy(ch, phot['aperture_sum_err'][0])


            for col in phot.colnames:
                phot[col].info.format = '%.8g'  # for consistent table output
            if printinfo:
                print(str("UT: %14.7f " % UT),str("%11.7f" % raref),str(", %11.7f" % decref),
                      ' Flux: ', flux)
            i = goodimagelist.index(image)
            time = times[i]
            photline = (image, UT, raref[0], decref[0], flux, phot['aperture_sum_err'][0], time)
            if not np.isnan(flux):
                goodtimes.append(time)
                photlist.append(photline)
                fluxes.append(flux)
                fxoff.append(xref1[0]-xposi)
                fyoff.append(yref1[0]-yposi)
            
    nmeas = len(fluxes)
    badtimes = []
    badflux = []
    medfluxes = []
    errest = []

    # package results, calculate medians and standard deviations to return to program
    
    for i,flux in zip(range(nmeas), fluxes):
        istr = max(0, i-3)
        iend = min(istr+7, nmeas)
        nearfluxes = []
        for j in range(istr,iend):
            nearfluxes.append(fluxes[j])
#             print(j,fluxes[j])
        medfluxes.append(np.nanmedian(nearfluxes))
    for i,flux,time in zip(range(nmeas), fluxes, goodtimes):
        istr = max(0, i-3)
        iend = min(istr+7, nmeas)
        nearfluxes = []
        nearmeds = []
        for j in range(istr,iend):
            nearfluxes.append(fluxes[j])
            nearmeds.append(medfluxes[j])
        ph_std = np.nanstd(nearmeds)
        errest.append(ph_std)
        ph_mean = np.nanmedian(nearmeds)
        cutoff = ph_std * 5.0
        if abs(flux-ph_mean)>cutoff or math.sqrt(fxoff[i]**2+fyoff[i]**2)>7:
#             print(flux, ph_mean, fxoff[i], fyoff[i])
            badtimes.append(time)
            badflux.append(flux)
    xreloff = fxoff #- xshift
    yreloff = fyoff #- yshift

    return photlist, goodtimes, fluxes, badtimes, badflux, xreloff, yreloff, xshift, yshift, errest


# Evaluate BCD photometry - plot photometry and x,y offsets and allow user to select points
# to exclude from dataset

def eval_phot(photlist, times, fluxes, badtimes, badflux, xreloff, yreloff, objectname,
              ch, xshift, yshift, do_mult_mos, impath, mosnum):            
    ch = getval(photlist[0][0],'CHNLNUM')
    matplotlib.use('Qt5Agg')
    plt.ion()
    npts = len(fluxes)
    errest = []
    badypt = []

    badxpt = []
    for i in range(len(badtimes)):
        idx = times.index(badtimes[i])
        badypt.append(yreloff[idx])
        badxpt.append(xreloff[idx])
    for i in range(npts):
        istr = max(0, i-3)
        iend = min(istr+7, npts)
        nearfluxes = []
        for j in range(istr,iend):
            nearfluxes.append(photlist[j][4])
        ph_std = np.nanstd(nearfluxes)
        errest.append(ph_std)

    # make plot of BCD photometry, showing red Xs where data points have been excluded
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[14,5])
    ax1.plot(times,yreloff, linestyle="None",marker='o',markerfacecolor='blue', markersize=6)
    ax1.plot(times,xreloff, linestyle="None",marker='o',markerfacecolor='green', markersize=6)
    ax1.grid()
    ax2.errorbar(times, fluxes, yerr=errest, linestyle="None", label="BCD",
                 marker='o',markerfacecolor='blue', markersize=6)  
    ax2.plot(times, fluxes, linestyle="None",marker='o',markerfacecolor='blue', markersize=6)  
    ax2.grid()
    ax2.set_title(str('Ch%1i: Left mouse button to toggle, right button to exit' % ch))
    if len(badtimes)>0:
        tempts, = ax2.plot(badtimes, badflux, linestyle="None",marker='x',markerfacecolor='red',
                           markersize=11, markeredgecolor = 'red', mew=3)    
        temptsx, = ax1.plot(badtimes,badypt, linestyle="None",marker='x',markerfacecolor='red',
                            markersize=11, markeredgecolor = 'red', mew=3)
        temptsy, = ax1.plot(badtimes,badxpt, linestyle="None",marker='x',markerfacecolor='red',
                            markersize=11, markeredgecolor = 'red', mew=3)
        rangeflist= fluxes.copy()
        for ii in range(len(times)):
            if times[ii] in badtimes:
                rangeflist[ii] = np.nan
        rangeflist = [x for x in rangeflist if not np.isnan(x)]
        if min(rangeflist)<0:
            ax2.set_ylim(min(rangeflist)*1.1,max(rangeflist)*1.1)
        else:
            ax2.set_ylim(min(rangeflist)*0.9,max(rangeflist)*1.1)
            
    plt.show()

    pts = plt.ginput(1, timeout=-1, mouse_stop=3)
    while len(pts)>0:                             # Allow user to click on points. Left click for changing points,
        td = 1E+12                                # right mouse button stops interactive input
        idx = -1
        if len(badtimes)>0:
            tempts.remove()
            temptsx.remove()
            temptsy.remove()
        if pts[0][0]>times[-1]:                    # If click was > largest time, remove all points above
            for i in range(len(times)):            # the flux value of the click
                if fluxes[i]>pts[0][1]:
                    badtimes.append(times[i])
                    badflux.append(fluxes[i])
                    badypt.append(yreloff[i])
                    badxpt.append(xreloff[i])
                    print("added bad time ",times[i])
        else:
            if pts[0][0] < 0:                      # If click was below 0 time, change all bad points to good
                badtimes = []
                badflux = []
                badypt = []
                badxpt = []
            else:                                  # Otherwise, toggle bad/good value for nearest point
                for i in range(len(times)):
                    tdist = np.sqrt(pow((pts[0][0] - times[i]),2)) #+ pow((pts[0][1] - fluxes[i]),2))
                    if tdist<td:
                        td = tdist
                        idx = i
                if idx>-1:
                    print(idx, times[idx], fluxes[idx])
                if idx>-1:
                    if times[idx] in badtimes:
                        bindex = badtimes.index(times[idx])
                        del badtimes[bindex]
                        del badflux[bindex]
                        del badypt[bindex]
                        del badxpt[bindex]
                        print("removed bad index", bindex)
                    else:
                        badtimes.append(times[idx])
                        badflux.append(fluxes[idx])
                        badypt.append(yreloff[idx])
                        badxpt.append(xreloff[idx])
                        print("added bad time ",times[idx])
        if len(badtimes)>0:
            tempts, = ax2.plot(badtimes, badflux, linestyle="None",marker='x',markerfacecolor='red',
                               markersize=11, markeredgecolor = 'red', mew=3)
            temptsx, = ax1.plot(badtimes,badypt, linestyle="None",marker='x',markerfacecolor='red',
                                markersize=11, markeredgecolor = 'red', mew=3)
            temptsy, = ax1.plot(badtimes,badxpt, linestyle="None",marker='x',markerfacecolor='red',
                                markersize=11, markeredgecolor = 'red', mew=3)

        fig.canvas.draw()
        fig.canvas.flush_events()
        pts = plt.ginput(1, timeout=-1, mouse_stop=3)

    plt.close()
    # Remove rejected points from list and return list of good photometry

    goodlist = photlist.copy()
    for i in range(len(photlist)):
        if photlist[i][6] in badtimes:
            goodlist.remove(photlist[i])
            
    # recalculate error estimate with only good points
    npts = len(goodlist)
    errest = []
    goodtimes = []
    goodfluxes = []
    gxreloff = []
    gyreloff = []
    for i in range(npts):
        istr = max(0, i-2)
        iend = min(istr+5, npts)
        goodfluxes.append(goodlist[i][4])
        goodtimes.append(goodlist[i][6])
        gxreloff.append(xreloff[i])
        gyreloff.append(yreloff[i])
        nearfluxes = []
        for j in range(istr,iend):
            nearfluxes.append(goodlist[j][4])
        ph_std = np.nanstd(nearfluxes)
        errest.append(ph_std)
        
    get_ipython().run_line_magic('matplotlib', 'inline')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[16,5])

    ax1.plot(times,yreloff, linestyle="None", label="y offsets", marker='o',
             markerfacecolor='blue', markersize=6)
    ax1.plot(times,xreloff, linestyle="None", label="x offsets", marker='o',
             markerfacecolor='green', markersize=6)
    ax1.set_ylabel("Offset (pixels)")
    ax1.legend()
    ax2.errorbar(goodtimes, goodfluxes, yerr=errest, linestyle="None", label="BCD",
                 marker='o',markerfacecolor='blue', markersize=6)  
    ax2.set_ylabel("Source flux (uJy)")
    ax1.set_xlabel("Time (s)")
    maxy = min(np.std(goodfluxes)*5+np.median(goodfluxes), max(goodfluxes)+2*np.std(goodfluxes))
    miny = max(max(np.median(goodfluxes)-5*np.std(goodfluxes), 
                   min(goodfluxes)-2*np.std(goodfluxes)),0)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim(miny, maxy)
    if len(badtimes)>0:
        tempts, = ax2.plot(badtimes, badflux, label="rejected", linestyle="None",
                           marker='x',markerfacecolor='red', markeredgecolor = 'red',
                           markersize=12, mew=3)    
        temptsx, = ax1.plot(badtimes,badypt, linestyle="None",marker='x',markerfacecolor='red',
                            markersize=12, markeredgecolor = 'red', mew=3, label="rejected")
        temptsy, = ax1.plot(badtimes,badxpt, linestyle="None",marker='x',markerfacecolor='red', 
                            markersize=12,markeredgecolor = 'red', mew=3, label="rejected")
                                
    plt.legend()
    
    if do_mult_mos:
        ax2.set_title(objectname + str("  Ch%1i" % ch) + str("_%1i" % mosnum))
        plt.savefig(impath + objectname + str("_I%1i_" % ch) + str("%1i.pdf" % mosnum))
    else:
        ax2.set_title(objectname + str("  Ch%1i" % ch))
        plt.savefig(impath + objectname + str("_I%1i.pdf" % ch))
    plt.show()
    
    return goodlist, errest, gxreloff, gyreloff, badtimes, badflux


# Make the query text to send to HORIZONS to get the NEO position at each BCD time

def getHorizonsQueryText(filelist,itime, SPK):
    fname = "HORIZONS.txt"
    with open(fname, "w") as output_file:
        output_file.write("!$$SOF\n")
        output_file.write(str("COMMAND= 'DES=%s'\n" % SPK))
        output_file.write("CENTER= '500@-79'\n")           # Spitzer Space Telescope
        output_file.write("MAKE_EPHEM= 'YES'\n")
        output_file.write("TABLE_TYPE= 'OBSERVER'\n")
        output_file.write("TLIST=\n")
        for fname in filelist:                           # JD at center of integration
            fileJD = getval(fname,'MJD_OBS') + 2400000.5 + itime/(24*3600*2) 
            output_file.write(str("  '%15.7f'\n" % fileJD))
        output_file.write("CAL_FORMAT= 'JD'\n")
        output_file.write("TIME_DIGITS= 'FRACSEC'\n")
        output_file.write("ANG_FORMAT= 'DEG'\n")
        output_file.write("OUT_UNITS= 'KM-S'\n")
        output_file.write("RANGE_UNITS= 'AU'\n")
        output_file.write("APPARENT= 'AIRLESS'\n")
        output_file.write("SUPPRESS_RANGE_RATE= 'NO'\n")
        output_file.write("SKIP_DAYLT= 'NO'\n")
        output_file.write("EXTRA_PREC= 'YES'\n")
        output_file.write("R_T_S_ONLY= 'NO'\n")
        output_file.write("REF_SYSTEM= 'J2000'\n")
        output_file.write("CSV_FORMAT= 'NO'\n")
        output_file.write("OBJ_DATA= 'YES'\n")
        output_file.write("QUANTITIES= '1,2,9,20,23,24'\n")
        output_file.write("!$$EOF")
    return



# These convert from ADU units (actually MJy/sr units in BCDs that have been reprojected to 0.6"/pix)
# to micro-Jansky (uJy) for photometry performed with the aperture and sky annulus radii defined here

# Calibration was done using the A-type stars J1812095+6329423, HD184837, and HD165459
# on 2022/09/05 using 20210905_IRACmosCAL jupyter notebook
# calibration derivation saved in 20220905calstarphot.xlsx

def ADU2uJy(ch,ADUval):          #calibration from ADU to uJy in ch1 and ch2
    if ch==1:
        uJyval = 9.326365 * ADUval
    elif ch==2:
        uJyval = 9.635281 * ADUval
    else:
        print("error - unknown channel")
        uJyval = 0
    return uJyval

def uJy2mag(ch, uJyval):         # convert uJy to mag for ch1 or ch2
    if ch==1:
        magval = -2.5 * ma.log10(uJyval * 1E-6/280.9)
    elif ch==2:
        magval = -2.5 * ma.log10(uJyval * 1E-6/179.9)
    else:
        print("error - unknown channel")
        magval = 0
    return magval
    


# Write the BCD photometry results to a .csv file

def write_BCDphot_output(outfile, BCDphot, errest, objectname, SPK, ch, overwrite=True, debug=False):
    goodfluxes = []
    raoffsets = []
    decoffsets = []
    writehdr = False
    if overwrite:
        if os.path.exists(outfile):
            try:
                os.remove(outfile)
            except:
                print("Warning: error while deleting file ", outfile)
    if not os.path.exists(outfile):
        headstr = "#Name,NAIFID,Channel,MJD_OBS(center),flux (uJy),fluxerr (uJy),mag,magerr,AORKEY,EXPID,DATE_OBS(start)"
        headstr = headstr + ",ra,dec,ra_diff (arcsec), dec_diff (arcsec), separation (arcsec)"
        if debug:
            print(headstr)
        writehdr = True
    with open(outfile, 'a') as f:
        if writehdr:
            f.write(headstr + '\n')
        for i in range(len(BCDphot)):
            hdr = getheader(BCDphot[i][0])
            time = hdr['MJD_OBS'] + 2400000.5 + hdr['FRAMTIME']/(24*3600*2.0)
            flux = BCDphot[i][4]
            fluxerr = errest[i]
            mag = uJy2mag(ch,flux)
            magerr = fluxerr / flux
            goodfluxes.append(BCDphot[i][4])
            outstr = objectname + "," + SPK + str(",%1i," % ch)
            outstr = outstr + str("%15.7f," % time) + str("%7.2f," % flux)
            outstr = outstr + str("%5.2f," % fluxerr) + str("%6.3f," % mag) + str("%5.3f," % magerr)
            outstr = outstr + str("%i," % hdr['AORKEY']) + str("%4i," % hdr['EXPID'])
            outstr = outstr + hdr['DATE_OBS']+ str(",%13.7f," % BCDphot[i][2])
            outstr = outstr + str("%13.7f" % BCDphot[i][3])
            horiz_ra = hdr['RA_REF']
            horiz_dec = hdr['DEC_REF']
            horiz_pos = SkyCoord(horiz_ra, horiz_dec, unit="deg")
            Spitz_pos = SkyCoord(BCDphot[i][2], BCDphot[i][3], unit="deg")
            ra_diff = (BCDphot[i][2]-horiz_ra)*3600*ma.cos(np.deg2rad(BCDphot[i][3]))  #offset in arcsec from 
            de_diff = (BCDphot[i][3]- horiz_dec)*3600                                  #Horizons to Spitzer position
            raoffsets.append(ra_diff)
            decoffsets.append(de_diff)
            sep = horiz_pos.separation(Spitz_pos)    # separation between Horizons position and Spitzer position
            outstr = outstr + (str(", %5.3f" % ra_diff) + str(", %5.3f" % de_diff)
                               + str(", %5.3f" % sep.arcsec))
            if debug:
                print(outstr)
            f.write(outstr + '\n')
    return goodfluxes, raoffsets, decoffsets

# Make the plots of X and Y pixel offsets for the BCDs for both channels

def plotRA_DEC(BCDphot1,BCDphot2, do_mult_mos, impath, objectname, mosnum):
    params={'legend.fontsize':'12','axes.labelsize':'18',
        'axes.titlesize':'18','xtick.labelsize':'18',
        'ytick.labelsize':'18','lines.linewidth':2,'axes.linewidth':2,'animation.html': 'html5'}
    plt.rcParams.update(params)
    times = []
    timesch1 = []
    timesch2 = []
    ra = []
    dec = []
    rach1 = []
    rach2 = []
    dech1 = []
    dech2 = []
    for i in range(len(BCDphot1)):
        hdr = getheader(BCDphot1[i][0])
        otime = (hdr['MJD_OBS'] + 2400000.5 + hdr['FRAMTIME']/(24*3600*2.0))
        times.append(otime)
        timesch1.append(otime)
        ra.append(BCDphot1[i][2])
        dec.append(BCDphot1[i][3])
        rach1.append(BCDphot1[i][2])
        dech1.append(BCDphot1[i][3])
    for i in range(len(BCDphot2)):
        hdr = getheader(BCDphot2[i][0])
        otime = hdr['MJD_OBS'] + 2400000.5 + hdr['FRAMTIME']/(24*3600*2.0)
        times.append(otime)
        timesch2.append(otime)
        ra.append(BCDphot2[i][2])
        dec.append(BCDphot2[i][3])       
        rach2.append(BCDphot2[i][2])
        dech2.append(BCDphot2[i][3])       
    ra = np.asarray(ra)
    dec = np.asarray(dec)
    timesch1 = (np.asarray(timesch1)- float(min(times)))*24*3600
    timesch2 = (np.asarray(timesch2)- float(min(times)))*24*3600
    times = (np.asarray(times)- float(min(times)))*24*3600
    ra_slope, ra_intercept, ra_r, ra_p, ra_std_err = stats.linregress(times, ra)
    de_slope, de_intercept, de_r, de_p, de_std_err = stats.linregress(times, dec)
    ra = (ra - (times*ra_slope + ra_intercept))*3600*ma.cos(dec[0]/57.29578)
    dec = (dec - (times*de_slope + de_intercept))*3600
    stdra=np.std(ra)
    stddec=np.std(dec)
    print("stderr: ",stdra, ",", stddec, " arcsec")
    rach1 = (rach1 - (timesch1*ra_slope + ra_intercept))*3600*ma.cos(dech1[0]/57.29578)
    dech1 = (dech1 - (timesch1*de_slope + de_intercept))*3600
    rach2 = (rach2 - (timesch2*ra_slope + ra_intercept))*3600*ma.cos(dech2[0]/57.29578)
    dech2 = (dech2 - (timesch2*de_slope + de_intercept))*3600

    fig = plt.figure(figsize=[15,6])
    ax = fig.add_subplot(111)
    l1, = ax.plot(timesch1, rach1, linestyle="None", label="ch1ra offsets", marker='o',
                  markerfacecolor='blue', markersize=9)
    l2, = ax.plot(timesch1,dech1, linestyle="None", label="ch1dec offsets", marker='o',
                  markerfacecolor='red', markersize=9)
    l3, = ax.plot(timesch2, rach2, linestyle="None", label="ch2ra offsets", marker='o',
                  markerfacecolor='green', markersize=9)
    l4, = ax.plot(timesch2,dech2, linestyle="None", label="ch2dec offsets", marker='o',
                  markerfacecolor='orange', markersize=9)
    ax.legend()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("position offset (arcsec)")
    minlimit = min(min(ra),min(min(dec),-0.5))
    maxlimit = max(max(ra),max(max(dec),0.5))
    ax.set_ylim(minlimit, maxlimit)
    plt.grid()
    if do_mult_mos:
        ax.set_title(objectname + str("_%1i" % mosnum))
        plt.savefig(impath + objectname + str("_poserr_%1i.pdf" % mosnum))
    else:
        ax.set_title(objectname)
        plt.savefig(impath + objectname + str("_poserr.pdf"))
    plt.show()
    
    return stdra, stddec


# Make the plots of X and Y pixel offsets for the BCDs for Channel 2 only

def plotRA_DECch2(BCDphot2, do_mult_mos, impath, objectname, mosnum):
    params={'legend.fontsize':'12','axes.labelsize':'18',
        'axes.titlesize':'18','xtick.labelsize':'18',
        'ytick.labelsize':'18','lines.linewidth':2,'axes.linewidth':2,'animation.html': 'html5'}
    plt.rcParams.update(params)
    times = []
    ra = []
    dec = []
    for i in range(len(BCDphot2)):
        hdr = getheader(BCDphot2[i][0])
        otime = hdr['MJD_OBS'] + 2400000.5 + hdr['FRAMTIME']/(24*3600*2.0)
        times.append(otime)
        ra.append(BCDphot2[i][2])
        dec.append(BCDphot2[i][3])       
    ra = np.asarray(ra)
    dec = np.asarray(dec)
    times = (np.asarray(times)- float(min(times)))*24*3600
    ra_slope, ra_intercept, ra_r, ra_p, ra_std_err = stats.linregress(times, ra)
    de_slope, de_intercept, de_r, de_p, de_std_err = stats.linregress(times, dec)
    ra_rel = (ra - (times*ra_slope + ra_intercept))*3600*math.cos(np.deg2rad(dec[0]))
    dec_rel = (dec - (times*de_slope + de_intercept))*3600
    stdra=np.std(ra_rel)
    stddec=np.std(dec_rel)
    print("stderr: ",stdra, ",", stddec, " arcsec")
    rach2 = (ra - (times*ra_slope + ra_intercept))*3600*math.cos(np.deg2rad(dec[0]))
    dech2 = (dec - (times*de_slope + de_intercept))*3600

    fig = plt.figure(figsize=[15,6])
    ax = fig.add_subplot(111)
    l3, = ax.plot(times, rach2, linestyle="None", label="ch2ra offsets", marker='o',
                  markerfacecolor='green', markersize=9)
    l4, = ax.plot(times,dech2, linestyle="None", label="ch2dec offsets", marker='o',
                  markerfacecolor='orange', markersize=9)
    ax.legend()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("position offset (arcsec)")
    minlimit = min(min(ra_rel),min(min(dec_rel),-0.5))
    maxlimit = max(max(ra_rel),max(max(dec_rel),0.5))
    ax.set_ylim(minlimit, maxlimit)
    plt.grid()
    if do_mult_mos:
        ax.set_title(objectname + str("_%1i" % mosnum))
        plt.savefig(impath + objectname + str("_poserr_%1i.pdf" % mosnum))
    else:
        ax.set_title(objectname)
        plt.savefig(impath + objectname + str("_poserr.pdf"))
    plt.show()
    
    return stdra, stddec


# Plot the results of the BCD photometry and the mean value, plus the BCD photometry

def plot_phot(goodlist, ch, Iphot, Iphoterr, Bphot, Berr, badtimes, badflux, do_mult_mos, mosnum,
              objectname, impath, show=True):
    npts = len(goodlist)
    errest = []
    goodtimes = []
    goodfluxes = []
    for i in range(npts):
        istr = max(0, i-2)
        iend = min(istr+5, npts)
        goodfluxes.append(goodlist[i][4])
        goodtimes.append(goodlist[i][6])
        nearfluxes = []
        for j in range(istr,iend):
            nearfluxes.append(goodlist[j][4])
        ph_std = np.nanstd(nearfluxes)
        errest.append(ph_std)
        
    get_ipython().run_line_magic('matplotlib', 'inline')
    fig, (ax1) = plt.subplots(1, 1, figsize=[9,5])

    ax1.errorbar(goodtimes, goodfluxes, yerr=errest, linestyle="None", label="BCD",
                 marker='o',markerfacecolor='blue', markersize=6)  
    ax1.set_ylabel("Source flux (uJy)")
    maxy = min(np.std(goodfluxes)*5+np.median(goodfluxes), max(goodfluxes)+2*np.std(goodfluxes))
    miny = max(max(np.median(goodfluxes)-5*np.std(goodfluxes), 
                   min(goodfluxes)-2*np.std(goodfluxes)),0)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylim(miny, maxy)
    if len(badtimes)>0:
        tempts, = ax1.plot(badtimes, badflux, label="rejected", linestyle="None",
                           marker='x',markerfacecolor='red', markersize=9, mew=3)    
    ax1.errorbar(np.mean(goodtimes), Iphot, yerr=Iphoterr, label="Mosaic", marker='s',linestyle = 'none',
                 markerfacecolor='red', markersize=12, ecolor='red', markeredgecolor = 'red') 
    ax1.errorbar(np.mean(goodtimes)+10, Bphot, yerr=Berr, label="BCDmean", marker='D', linestyle = 'none',
                 markerfacecolor='green', markersize=12, ecolor='green', markeredgecolor = 'green') 
    plt.legend()
    
    if do_mult_mos:
        ax1.set_title(objectname + str("  Ch%1i" % ch) + str("_%1i" % mosnum))
        plt.savefig(impath+objectname+str("_BCDphot_I%1i_" % ch) + str("%1i.pdf" % mosnum))
    else:
        ax1.set_title(objectname + str("  Ch%1i" % ch))
        plt.savefig(impath+objectname+str("_BCDphot_I%1i.pdf" % ch))
    if show:
        plt.show()
    plt.close()


# Check to see if NEO position is within the data frame

def getNEOframes(imlist):
    NEOlist = []
    for image in imlist:
        raref = getval(image,'RA_REF')
        decref = getval(image,'DEC_REF')
        hdr = getheader(image)
        idat = getdata(image)
        w = WCS(hdr)
        x, y = w.wcs_world2pix(raref, decref, 1)
        if (x>0) and (x<idat.shape[0]) and (y>0) and (y<idat.shape[1]):
            NEOlist.append(image)
    return NEOlist


# Replace the NEO position given in the header with the new value
# save the original values in new keywords
def update_refpos(file, raref, decref):
    data, header = fits.getdata(file, header=True)
    header['RA_REF_ORIG'] = header['RA_REF']
    header['DEC_REF_ORIG'] = header['DEC_REF']
    header['RA_REF'] = float(raref)
    header['DEC_REF'] = float(decref)
    outfile = file[:file.rfind('.fits')]+'c.fits'
    fits.writeto(outfile, data, header, overwrite=True)
    return outfile

# plot a small central section of the sky coadd image to visualize the NEO masking
def plotskycoadds(impath, I2skycoadname, objectname, do_mult_mos, mosnum):
    fig = plt.figure(figsize=(5,4))
    img = getdata(I2skycoadname)
    cenx = int(img.shape[0]/2)
    ceny = int(img.shape[1]/2)
    plt.imshow(img[(cenx-125):(cenx+125),(ceny-125):(ceny+125)], cmap='gray',origin='lower', vmin=0 )
    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='y', labelsize=10)
    plt.title('SkyMosCoadds: '+objectname)
    if do_mult_mos:
        plt.savefig(impath + objectname + "_skycoadd" + str("_%1i" % mosnum) + ".pdf")
    else:
        plt.savefig(impath + objectname + "_skycoadd.pdf")
    plt.show()
    return

def plotSkymosaics(ch1imname, ch2imname, bothchan, std_base, impath, objectname, mosnum, do_mult_mos):
    if bothchan:
        ch1NEOimage = getdata(ch1imname)
        fig = plt.figure(figsize=(12,8))        #  display all cutouts
        ax = []
        ax.append(fig.add_subplot(1, 2, 1) )
        if do_mult_mos:
            ax[-1].set_title('Ch1: ' + objectname + str("_%1i" % mosnum))  # set title
        else:
            ax[-1].set_title('Ch1: ' + objectname)  # set title
        ax[-1].tick_params(axis='x', labelsize=10)
        ax[-1].tick_params(axis='y', labelsize=10)
        img = ch1NEOimage + std_base*3
        plt.imshow(img, cmap='gray',origin='lower',norm=LogNorm()) 
        ch2NEOimage = getdata(ch2imname)
        ax.append(fig.add_subplot(1, 2, 2) )
        ax[-1].tick_params(axis='x', labelsize=10)
        ax[-1].tick_params(axis='y', labelsize=10)
        img = ch2NEOimage + std_base*3
        plt.imshow(img, cmap='gray',origin='lower',norm=LogNorm()) 
        if do_mult_mos:
            ax[-1].set_title('Ch2: '+objectname + str("_%1i" % mosnum))  # set title
        else:
            ax[-1].set_title('Ch2: '+objectname)  # set title
        plt.show()
    else:
        ch2NEOimage = getdata(ch2imname)
        fig = plt.figure(figsize=(12,8))
        img = ch2NEOimage + std_base*3
        plt.imshow(img, cmap='gray',norm=LogNorm(),origin='lower')    #  norm=LogNorm(), 
        plt.tick_params(axis='x', labelsize=10)
        plt.tick_params(axis='y', labelsize=10)
        if do_mult_mos:
            plt.title('Ch2: '+objectname + str("_%1i" % mosnum))
        else:
            plt.title('Ch2: '+objectname)
        plt.show()
    return

# Plot the central part of the NEO mosaic
def plotNEOmosaics(ch1NEOimage, ch2NEOimage, bothchan, std_base, impath, objectname, mosnum, do_mult_mos):
    wsz = 20
    xsz = ch2NEOimage.shape[0]
    ysz = ch2NEOimage.shape[1]
    xcn = int(xsz/2)+1
    ycn = int(ysz/2)+1
    xll = xcn-wsz
    xul = xcn+wsz
    yll = ycn-wsz
    yul = ycn+wsz
    if bothchan:
        fig = plt.figure(figsize=(8, 4))        #  display all cutouts
        ax = []
        ax.append(fig.add_subplot(1, 2, 1) )
        if do_mult_mos:
            ax[-1].set_title('Ch1: ' + objectname + str("_%1i" % mosnum))  # set title
        else:
            ax[-1].set_title('Ch1: ' + objectname)  # set title
        ax[-1].tick_params(axis='x', labelsize=10)
        ax[-1].tick_params(axis='y', labelsize=10)
        img = ch1NEOimage + std_base*3

        plt.imshow(img[xll:xul,yll:yul], cmap='gray',origin='lower',norm=LogNorm()) 
        ax.append(fig.add_subplot(1, 2, 2) )
        ax[-1].tick_params(axis='x', labelsize=10)
        ax[-1].tick_params(axis='y', labelsize=10)
        img = ch2NEOimage + std_base*3
        plt.imshow(img[xll:xul,yll:yul], cmap='gray',origin='lower',norm=LogNorm()) 
        if do_mult_mos:
            ax[-1].set_title('Ch2: '+objectname + str("_%1i" % mosnum))  # set title
            plt.savefig(impath + objectname + "_NEOmos_" + str("%1i.pdf" % mosnum))
        else:
            ax[-1].set_title('Ch2: '+objectname)  # set title
            plt.savefig(impath + objectname + "_NEOmos.pdf")
        plt.show()
    else:
        fig = plt.figure(figsize=(5,4))
        img = ch2NEOimage + std_base*3
        plt.imshow(img[xll:xul,yll:yul], cmap='gray',norm=LogNorm(),origin='lower')    #  norm=LogNorm(), 
        plt.tick_params(axis='x', labelsize=10)
        plt.tick_params(axis='y', labelsize=10)
        if do_mult_mos:
            plt.title('Ch2: '+objectname + str("_%1i" % mosnum))
            plt.savefig(impath + objectname + "_NEOmos" + str("_%1i.pdf" % mosnum))
        else:
            plt.title('Ch2: '+objectname)
            plt.savefig(impath + objectname + "_NEOmos.pdf")
        plt.show()
    return

# plot image of skydarks constructed from the BCD frames

def plotdarkframes(bothchan, objectname, impath, darkframe1, darkframe2):
    if bothchan:
        fig = plt.figure(figsize=(14, 6))        #  display all cutouts
        ax = []
        ax.append(fig.add_subplot(1, 2, 1) )
        ax[-1].set_title('Darkframe Ch1: ' + objectname)  # set title
        ax[-1].tick_params(axis='x', labelsize=10)
        ax[-1].tick_params(axis='y', labelsize=10)
        lowval = -np.nanstd(darkframe1)*2
        hival = -lowval
        pos1 = plt.imshow(darkframe1, cmap='gray',origin='lower', vmin=lowval, vmax=hival)
        fig.colorbar(pos1, ax=ax[-1])
        ax.append(fig.add_subplot(1, 2, 2) )
        ax[-1].tick_params(axis='x', labelsize=10)
        ax[-1].tick_params(axis='y', labelsize=10)
        pos2 = plt.imshow(darkframe2, cmap='gray',origin='lower', vmin=lowval, vmax=hival) #,norm=LogNorm()) 
        ax[-1].set_title('Darkframe Ch2: ' + objectname)  # set title
        fig.colorbar(pos2, ax=ax[-1])
        plt.savefig(impath + objectname + "_darks.pdf")
        plt.show()
    else:
        fig = plt.figure(figsize=(5,4))
        lowval = -np.nanstd(darkframe2)*2
        hival = -lowval
        pos=plt.imshow(darkframe2, cmap='gray',origin='lower', vmin=lowval, vmax=hival) 
        fig.colorbar(pos)
        plt.tick_params(axis='x', labelsize=10)
        plt.tick_params(axis='y', labelsize=10)
        plt.title('Darkframe Ch2: '+objectname)
        plt.savefig(impath + objectname + "_darks.pdf")
        plt.show()
    return

# copy the results to another directory to collect them in one place
def copy_results():
    filenames = impath + r'*.csv'
    for file in glob.glob(filenames):
        shutil.copy2(file, resultsdir)
    filenames = impath + r'*.pdf'
    for file in glob.glob(filenames):
        shutil.copy2(file, resultsdir)
    filenames = impath + r'*.log'
    for file in glob.glob(filenames):
        shutil.copy2(file, resultsdir)
        
# delete the specified files 
def clean_datadir(filenames):
    filelist= glob.glob(filenames)
    if len(filelist)>0:
        for filepath in filelist:
            try:
                os.remove(filepath)
            except:
                print("Error while deleting file : ", filepath)

# Check if the position specified is within the array given
def positions_OK(pos, img):
    if np.isnan(pos[0]):
        return False
    elif np.isnan(pos[1]):
        return False
    elif (pos[0]>=0 and pos[0]<img.shape[1] and pos[1]>=0 and pos[1]<img.shape[0]):
        return True
    else:
        return False
    
# returns distance between NEO positions in frames in arcsec
def getNEOmotion(file1, file2):            
    c1 = SkyCoord(getval(file1,'RA_REF'), getval(file1,'DEC_REF'), unit="deg")
    c2 = SkyCoord(getval(file2,'RA_REF'), getval(file2,'DEC_REF'), unit="deg")
    return c1.separation(c2).value*3600

def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))

