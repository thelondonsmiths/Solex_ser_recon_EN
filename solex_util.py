"""
@author: Andrew Smith
contributors: Valerie Desnoux, Jean-Francois Pittet, Jean-Baptiste Butet, Pascal Berteau, Matt Considine
Version 6 November 2022

"""

import numpy as np
import matplotlib.figure
import matplotlib.pyplot
from astropy.io import fits
from scipy.interpolate import interp1d
import os
#import time
from scipy.signal import savgol_filter
import cv2
import sys
import math
from scipy.ndimage import gaussian_filter1d
from numpy.polynomial.polynomial import polyval
from video_reader import *
import tkinter as tk
import ctypes # Modification Jean-Francois: for reading the monitor size
import cv2

mylog = []


def clearlog():
    mylog.clear()


def logme(s):
    print(s)
    mylog.append(s + '\n')


# read video and return constructed image of sun using fit
def read_video_improved(file, fit, options):
    rdr = video_reader(file)
    ih, iw = rdr.ih, rdr.iw
    FrameMax = rdr.FrameCount
    disk_list = [np.zeros((ih, FrameMax), dtype='uint16')
                 for _ in options['shift']]

    if options['flag_display']:
        screen = tk.Tk()
        sw, sh = screen.winfo_screenwidth(), screen.winfo_screenheight()
        scaling = sh/ih * 0.8
        screen.destroy()
        cv2.namedWindow('disk', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('disk', int(FrameMax * scaling), int(ih * scaling))
        cv2.moveWindow('disk', 200, 0)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.moveWindow('image', 0, 0)
        cv2.resizeWindow('image', int(iw * scaling), int(ih * scaling))

    col_indeces = []

    for shift in options['shift']:
        ind_l = (np.asarray(fit)[:, 0] + np.ones(ih)*shift).astype(int)

        # CLEAN if fitting goes too far
        ind_l[ind_l < 0] = 0
        ind_l[ind_l > iw - 2] = iw - 2
        ind_r = (ind_l + np.ones(ih)).astype(int)
        col_indeces.append((ind_l, ind_r))

    left_weights = np.ones(ih) - np.asarray(fit)[:, 1]
    right_weights = np.ones(ih) - left_weights

    # lance la reconstruction du disk a partir des trames
    print('reader num frames:', rdr.FrameCount)
    while rdr.has_frames():
        img = rdr.next_frame()
        for i in range(len(options['shift'])):
            ind_l, ind_r = col_indeces[i]
            left_col = img[np.arange(ih), ind_l]
            right_col = img[np.arange(ih), ind_r]
            IntensiteRaie = left_col * left_weights + right_col * right_weights
            disk_list[i][:, rdr.FrameIndex] = IntensiteRaie

        if options['flag_display'] and rdr.FrameIndex % 10 == 0:
            # disk_list[1] is always shift = 0
            cv2.imshow('image', img)
            cv2.imshow('disk', disk_list[1])
            if cv2.waitKey(
                    1) == 27:                     # exit if Escape is hit
                cv2.destroyAllWindows()
                sys.exit()
    return disk_list, ih, iw, rdr.FrameCount


def make_header(rdr):
    # initialisation d'une entete fits (etait utilisé pour sauver les trames
    # individuelles)
    hdr = fits.Header()
    hdr['SIMPLE'] = 'T'
    hdr['BITPIX'] = 32
    hdr['NAXIS'] = 2
    hdr['NAXIS1'] = rdr.iw
    hdr['NAXIS2'] = rdr.ih
    hdr['BZERO'] = 0
    hdr['BSCALE'] = 1
    hdr['BIN1'] = 1
    hdr['BIN2'] = 1
    hdr['EXPTIME'] = 0
    return hdr

# compute mean and max image of video

def detect_bord(img, axis):
    blur = cv2.blur(img, ksize=(5,5))
    ymean = np.mean(blur, axis)
    threshhold = np.median(ymean) / 5
    where_sun = ymean > threshhold
    lb = np.argmax(where_sun)
    ub = img.shape[int(not axis)] - 1 - np.argmax(np.flip(where_sun)) # int(not axis) : get the other axis 1 -> 0 and 0 -> 1
    return lb, ub

def compute_mean_max(file):
    """IN : file path"
    OUT :numpy array
    """
    rdr = video_reader(file)
    logme('Width, Height : ' + str(rdr.Width) + ' ' + str(rdr.Height))
    logme('Number of frames : ' + str(rdr.FrameCount))
    my_data = np.zeros((rdr.ih, rdr.iw), dtype='uint64')
    max_data = np.zeros((rdr.ih, rdr.iw), dtype='uint16')
    while rdr.has_frames():
        img = rdr.next_frame()
        my_data += img
        max_data = np.maximum(max_data, img)
    return (my_data / rdr.FrameCount).astype('uint16'), max_data


def compute_mean_return_fit(file, options, hdr, iw, ih, basefich0):
    """
    ----------------------------------------------------------------------------
    Use the mean image to find the location of the spectral line of maximum darkness
    Apply a 3rd order polynomial fit to the datapoints, and return the fit, as well as the
    detected extent of the line in the y-direction.
    ----------------------------------------------------------------------------
    """
    flag_display = options['flag_display']
    # first compute mean image
    # rdr is the video_reader object
    mean_img, max_img = compute_mean_max(file)
    
    if options['save_fit']:
        DiskHDU = fits.PrimaryHDU(mean_img, header=hdr)
        DiskHDU.writeto(basefich0 + '_mean.fits', overwrite='True')

    # affiche image moyenne
    if flag_display:
        screen = tk.Tk()
        sw, sh = screen.winfo_screenwidth(), screen.winfo_screenheight()
        scaling = sh/ih * 0.8
        screen.destroy()
        cv2.namedWindow('Ser mean', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Ser mean', int(iw*scaling), int(ih*scaling))
        cv2.moveWindow('Ser mean', 100, 0)
        cv2.imshow('Ser mean', mean_img)
        if cv2.waitKey(2000) == 27:                     # exit if Escape is hit
            cv2.destroyAllWindows()
            sys.exit()

        cv2.destroyAllWindows()
    y1, y2 = detect_bord(max_img, axis=1) # use maximum image to detect borders
    y1 = min(max_img.shape[0]-1, y1+10)
    y2 = max(0, y2-10)
    logme('Vertical limits y1, y2 : ' + str(y1) + ' ' + str(y2))
    min_intensity = np.argmin(mean_img, axis = 1) # use mean image to detect spectral line
    p = np.flip(np.asarray(np.polyfit(np.arange(y1, y2), min_intensity[y1:y2], 3), dtype='d'))
    # remove outlier points and get new line fit
    delta = polyval(np.asarray(np.arange(y1,y2), dtype='d'), p) - min_intensity[y1:y2]
    stdv = np.std(delta)
    keep = np.abs(delta/stdv) < 3
    p = np.flip(np.asarray(np.polyfit(np.arange(y1, y2)[keep], min_intensity[y1:y2][keep], 3), dtype='d'))
    logme('Spectral line polynomial fit: ' + str(p))
    curve = polyval(np.asarray(np.arange(ih), dtype='d'), p)
    fit = [[math.floor(curve[y]), curve[y] - math.floor(curve[y]), y] for y in range(ih)]
    if not options['clahe_only']:
        fig = matplotlib.figure.Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mean_img, cmap=matplotlib.pyplot.cm.gray)
        s = (y2-y1)//20 + 1
        ax.plot(min_intensity[y1:y2:s], np.arange(y1, y2, s), 'rx', label='line detection')
        ax.plot(curve, np.arange(ih), label='polynomial fit')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_aspect(0.1)
        fig.tight_layout()
        fig.savefig(basefich0+'_spectral_line_data.png', dpi=400)
    return fit, y1, y2

'''
img: np array
borders: [minX, minY, maxX, maxY]
cirlce: (centreX, centreY, radius)
not_fake: true/false on if this was a user-requested image
'''

def correct_transversalium2(img, circle, borders, options, not_fake, basefich):
    if circle == (-1, -1, -1):
        print('ERROR : no circle fit so no transversalium correction')
        return img
    y_s = []
    y_mean = []
    y1 = math.ceil(max(circle[1] - circle[2], borders[1]))
    y2 = math.floor(min(circle[1] + circle[2], borders[3]))
    for y in range(y1, y2):
        dx = math.floor((circle[2]**2 - (y-circle[1])**2)**0.5)
        strip = img[y, math.ceil(max(circle[0] - dx, borders[0])) : math.floor(min(circle[0] + dx, borders[2]))]

        y_s.append(y)
        y_mean.append(np.mean(strip))


    #smoothed2 = savgol_filter(y_mean, min(301, len(y_mean) // 2 * 2 - 1), 3)
    smoothed = savgol_filter(y_mean, min(options['trans_strength'], len(y_mean) // 2 * 2 - 1), 3)
    #plt.plot(y_s, y_mean)
    #plt.plot(y_s, smoothed2)
    #plt.plot(y_s, smoothed)
    #plt.show()

    correction = np.divide(smoothed, y_mean)

    a = 0.05 # taper width
    N = correction.shape[0]

    # Tukey taper function
    def t(x):
        if 0 <= x < a*N/2:
            return 1/2 * (1-math.cos(2*math.pi*x/(a*N)))
        elif a*N/2 <= x <= N/2:
            return 1
        elif N/2 <= x <= N:
            return t(N - x)
        print('error: weird input for taper function: ' + str(x))
        return 1

    taper = np.array([t(x) for x in range(N)])
    
    correction_t = np.ones(N) + (correction - np.ones(N)) * taper

    #plt.plot(y_s, correction)
    #plt.plot(y_s, correction_t)
    #plt.show()

    c = np.ones(img.shape[0])
    c[y1:y2] = correction_t
    #c[c<1] = 1
    if not_fake and not options['clahe_only']:
        fig = matplotlib.figure.Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(c)
        ax.set_xlabel('y')
        ax.set_ylabel('transversalium correction factor')
        fig.savefig(basefich+'_transversalium_correction.png', dpi=300)
    ret = (img.T * c).T # multiply each row in image by correction factor
    ret[ret > 65535] = 65535 # prevent overflow
    return np.array(ret, dtype='uint16') 


def image_process(frame, cercle, options, header, basefich):
    flag_result_show = options['flag_display']
                
    # create a CLAHE object (Arguments are optional)
    # clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(5,5))
    clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(2,2))
    cl1 = clahe.apply(frame)
    
    # image leger seuils
    frame1=np.copy(frame)
    Seuil_bas=np.percentile(frame, 25)
    Seuil_haut=np.percentile(frame,99.9999)
    print('Seuil bas       :', np.floor(Seuil_bas))
    print('Seuil haut      :', np.floor(Seuil_haut))
    fc=(frame1-Seuil_bas)* (65535/(Seuil_haut-Seuil_bas))
    fc[fc<0]=0
    fc[fc>65535] = 65535
    frame_contrasted=np.array(fc, dtype='uint16')
    
    # image seuils serres 
    frame1=np.copy(frame)
    Seuil_bas=(Seuil_haut*0.25)
    Seuil_haut=np.percentile(frame1,99.9999)
    print('Seuil bas HC    :', np.floor(Seuil_bas))
    print('Seuil haut HC   :', np.floor(Seuil_haut))
    fc2=(frame1-Seuil_bas)* (65535/(Seuil_haut-Seuil_bas))
    fc2[fc2<0]=0
    fc2[fc2>65535] = 65535
    frame_contrasted2=np.array(fc2, dtype='uint16')
    
    # image seuils protus
    frame1=np.copy(frame)
    Seuil_bas=0
    Seuil_haut=np.percentile(frame1,99.9999)*0.18        
    print('Seuil bas protu :', np.floor(Seuil_bas))
    print('Seuil haut protu:', np.floor(Seuil_haut))
    fc2=(frame1-Seuil_bas)* (65535/(Seuil_haut-Seuil_bas))
    fc2[fc2<0]=0
    fc2[fc2>65535] = 65535
    frame_contrasted3=np.array(fc2, dtype='uint16')
    if not cercle == (-1, -1, -1) and options['disk_display']:
        x0=int(cercle[0])
        y0=int(cercle[1])
        r=int(cercle[2]) + options['delta_radius']
        if r > 0:
            frame_contrasted3=cv2.circle(frame_contrasted3, (x0,y0),r,80,-1)            
    Seuil_bas=np.percentile(cl1, 10)
    Seuil_haut=np.percentile(cl1,99.9999)*1.05
    cc=(cl1-Seuil_bas)*(65535/(Seuil_haut-Seuil_bas))
    cc[cc<0]=0
    cc[cc>65535] = 65535
    cc=np.array(cc, dtype='uint16')

    # handle rotations
    cc = np.rot90(cc, options['img_rotate']//90, axes=(0,1))
    frame_contrasted = np.rot90(frame_contrasted, options['img_rotate']//90, axes=(0,1))
    frame_contrasted2 = np.rot90(frame_contrasted2, options['img_rotate']//90, axes=(0,1))
    frame_contrasted3 = np.rot90(frame_contrasted3, options['img_rotate']//90, axes=(0,1))
    frame = np.rot90(frame, options['img_rotate']//90, axes=(0,1))
    
    # sauvegarde en png de clahe
    cv2.imwrite(basefich+'_clahe.png',cc)   # Modification Jean-Francois: placed before the IF for clear reading
    if not options['clahe_only']:
        # sauvegarde en png pour appliquer une colormap par autre script
        #cv2.imwrite(basefich+'_disk.png',frame_contrasted)
        # sauvegarde en png pour appliquer une colormap par autre script
        cv2.imwrite(basefich+'_diskHC.png',frame_contrasted2)
        # sauvegarde en png pour appliquer une colormap par autre script
        cv2.imwrite(basefich+'_protus.png',frame_contrasted3)
    
    # The 3 images are concatenated together in 1 image => 'Sun images'
    # The 'Sun images' is scaled for the monitor maximal dimension ... it is scaled to match the dimension of the monitor without 
    # changing the Y/X scale of the images 
    if flag_result_show:
        im_3 = cv2.hconcat([cc, frame_contrasted2, frame_contrasted3])
        screen = tk.Tk()
        screensize = screen.winfo_screenwidth(), screen.winfo_screenheight()
        screen.destroy()
        scale = min(screensize[0] / im_3.shape[1], screensize[1] / im_3.shape[0]) * 0.9
        cv2.namedWindow('Sun images', cv2.WINDOW_NORMAL)
        cv2.moveWindow('Sun images', 0, 0)
        cv2.resizeWindow('Sun images',int(im_3.shape[1] * scale), int(im_3.shape[0] * scale))
        cv2.imshow('Sun images',im_3)
        cv2.waitKey(options['tempo'])  # affiche et continue
        cv2.destroyAllWindows()

    frame2=np.copy(frame)
    frame2=np.array(cl1, dtype='uint16')
    # sauvegarde le fits
    if options['save_fit']:
        DiskHDU=fits.PrimaryHDU(frame2,header)
        DiskHDU.writeto(basefich+ '_clahe.fits', overwrite='True')
