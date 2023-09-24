"""
@author: Andrew Smith
contributors: Valerie Desnoux, Jean-Francois Pittet, Jean-Baptiste Butet, Pascal Berteau, Matt Considine
Version 14 September 2023

"""

import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt
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
from scipy.optimize import curve_fit
import datetime
import traceback

def clearlog(path, options):
    try:
        with open(output_path(path, options), 'w') as f:
            f.write('start time: ' + str(datetime.datetime.now()) + '\n')
    except Exception:
        traceback.print_exc()
        print('ERROR: failed to log file: ' + path)

def write_complete(path, options):
    try:
        with open(output_path(path, options), 'a') as f:
            f.write('end time: ' + str(datetime.datetime.now()) + '\n')
    except Exception:
        traceback.print_exc()
        print('ERROR: failed to log file: ' + path)
        

def logme(path, options, s):
    if '_nolog' in options:
        return
    try:
        with open(output_path(path, options), 'a') as f:
            f.write(s + '\n')
    except Exception:
        traceback.print_exc()
        print('ERROR: failed to log file: ' + path)

'''
if options['output_dir'] is empty, then output there
else output same file name, but into directory in options
'''
def output_path(path, options):
    if options['output_dir'].strip() == '':
        return path
    return os.path.join(options['output_dir'], os.path.basename(path))

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# return values in an array not "m-far" from mean
def reject_outliers(data, m = 2):
    #bins = np.linspace(0, np.max(data) + 1, 64)
    #inds = np.digitize(data, bins)
    #modals, counts = np.unique(inds, return_counts=True)
    #modal_value = bins[np.argmax(counts)]
    #print('modal value', modal_value)
    median_value = np.median(data)
    d = np.abs(data - median_value)
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return data[s<m]

#downscale an image
def downscale(image, f):
    return cv2.resize(image, (0,0), fx=f, fy=f) 

# read video and return constructed image of sun using fit
def read_video_improved(rdr, fit, options):
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
    #print('reader num frames:', rdr.FrameCount)
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

def compute_mean_max(rdr, options, basefich0):
    """IN : file path"
    OUT :numpy array
    """
    #basefich0 = os.path.splitext(file)[0] # file name without extension #TOTO delete this line
    
    logme(basefich0 + '_log.txt', options, 'Width, Height : ' + str(rdr.Width) + ' ' + str(rdr.Height))
    logme(basefich0 + '_log.txt', options, 'Number of frames : ' + str(rdr.FrameCount))
    my_data = np.zeros((rdr.ih, rdr.iw), dtype='uint64')
    max_data = np.zeros((rdr.ih, rdr.iw), dtype='uint16')
    while rdr.has_frames():
        img = rdr.next_frame()
        my_data += img
        max_data = np.maximum(max_data, img)
    return (my_data / rdr.FrameCount).astype('uint16'), max_data


def compute_mean_return_fit(vid_rdr, options, hdr, iw, ih, basefich0):
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
    mean_img, max_img = compute_mean_max(vid_rdr, options, basefich0)
    
    if options['save_fit']:
        DiskHDU = fits.PrimaryHDU(mean_img, header=hdr)
        DiskHDU.writeto(output_path(basefich0 + '_mean.fits', options), overwrite='True')

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
    clip = int((y2 - y1) * 0.05)
    y1 = min(max_img.shape[0]-1, y1+clip)
    y2 = max(0, y2-clip)
    logme(basefich0 + '_log.txt', options, 'Vertical limits y1, y2 : ' + str(y1) + ' ' + str(y2))
    blur_width_x = 25
    blur_width_y = int((y2 - y1) * 0.01)
    blur = cv2.blur(mean_img, ksize=(blur_width_x,blur_width_y))
    min_intensity = blur_width_x//2 + np.argmin(blur[:, blur_width_x//2:-blur_width_x//2], axis = 1) # use blurred mean image to detect spectral line
    
    p = np.flip(np.asarray(np.polyfit(np.arange(y1, y2), min_intensity[y1:y2], 3), dtype='d'))
    # remove outlier points and get new line fit
    delta = polyval(np.asarray(np.arange(y1,y2), dtype='d'), p) - min_intensity[y1:y2]
    stdv = np.std(delta)
    keep = np.abs(delta/stdv) < 3
    p = np.flip(np.asarray(np.polyfit(np.arange(y1, y2)[keep], min_intensity[y1:y2][keep], 3), dtype='d'))
    #logme('Spectral line polynomial fit: ' + str(p))

    # find shift to non-blurred minimum
    min_intensity_sharp = np.argmin(mean_img, axis = 1) # use original mean image to detect spectral line
    delta_sharp = polyval(np.asarray(np.arange(y1,y2), dtype='d'), p) - min_intensity_sharp[y1:y2]
    
    values, counts = np.unique(np.around(delta_sharp, 1),  return_counts=True)
    ind = np.argpartition(-counts, kth=2)[:2] 
    shift = values[ind[0]] # find mode
    #logme(f'shift correction : {shift}')
    
    #matplotlib.pyplot.hist(delta_sharp, np.linspace(-20, 20, 400))
    #matplotlib.pyplot.show()

    tol_line_fit = 5
    mask_good = np.abs(delta_sharp - shift) < tol_line_fit
    p = np.flip(np.asarray(np.polyfit(np.arange(y1, y2)[mask_good], min_intensity_sharp[y1:y2][mask_good], 3), dtype='d'))
    logme(basefich0 + '_log.txt', options, 'Spectral line polynomial fit: ' + str(p))
    
    curve = polyval(np.asarray(np.arange(ih), dtype='d'), p)
    fit = [[math.floor(curve[y]), curve[y] - math.floor(curve[y]), y, curve[y]] for y in range(ih)]

    
    
    if not options['clahe_only'] and not options['protus_only']:
        fig = matplotlib.figure.Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mean_img, cmap=matplotlib.pyplot.cm.gray)
        s = (y2-y1)//20 + 1
        ax.plot(min_intensity_sharp[y1:y2][mask_good][::s], np.arange(y1, y2)[mask_good][::s], 'rx', label='line detection')
        ax.plot(curve, np.arange(ih), label='polynomial fit')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_aspect(0.1)
        fig.tight_layout()
        fig.savefig(output_path(basefich0+'_spectral_line_data.png', options), dpi=400)
    return mean_img, np.array(fit), y1, y2


def apply_lin_filter(img, linlen, half_width, spurious_flag, y1, y2, circle):
    ## counts of spurious
    s_cumsum = np.cumsum(spurious_flag)
    delayed = np.roll(s_cumsum, half_width)
    advanced = np.roll(s_cumsum, -half_width)
    delayed[:half_width] = 0
    advanced[-half_width:] = advanced[-half_width-1]
    rolling_sums = 2*half_width - (advanced - delayed - spurious_flag)
    #plt.plot(rolling_sums)
    #plt.show()

    filtered = np.log(img)
    for j in np.where(spurious_flag):
        filtered[j, :] = 0
    #plt.imshow(filtered)
    #plt.show()
    kernel = np.ones((half_width*2+1, linlen))
    kernel[half_width, :] = 0
    result = cv2.filter2D(filtered, -1, kernel)
    result2 = result / (linlen * np.tile(rolling_sums, (img.shape[1], 1)).T)

    #plt.imshow(result)
    #plt.show()

    #plt.imshow(result2)
    #plt.show()

    filt2 = np.log(img)
    prev = np.zeros(img.shape[1])
    for i in range(img.shape[0]):
        if spurious_flag[i]:
            filt2[i, :] = prev/2
        else:
            prev = filt2[i, :]
    prev = np.zeros(img.shape[1])
    for i in range(img.shape[0]-1, -1, -1):
        if spurious_flag[i]:
            filt2[i, :] += prev/2
        else:
            prev = filt2[i, :]            
    result3 = cv2.filter2D(filt2, -1, kernel/np.sum(kernel))
    #plt.imshow(result3)
    #plt.show()

    kernelLin = np.ones((1, linlen))

    result4 = cv2.filter2D(np.log(img), -1, kernelLin/np.sum(kernelLin))

    delta = result4 - result3

    #plt.imshow(delta, cmap='bwr')
    #plt.show()

    a = 0.05 # taper width
    N = y2 - y1
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
    
    c = np.zeros(img.shape[0])
    c[y1:y2] = taper

    delta = fix_edge_effect(delta, circle, linlen+20) # add fudge factor of "+20"


    #plt.imshow(delta, cmap='bwr')
    #plt.show()
    
    return img * np.exp(-delta*c.reshape(-1, 1))


def fix_edge_effect(multiplier, circle, linlen):
    y1 = math.ceil(max(circle[1] - circle[2], 0))
    y2 = math.floor(min(circle[1] + circle[2], multiplier.shape[0] - 1))
    halflen = linlen // 2
    multiplier[:y1, :] = 0
    multiplier[y2+1:, :] = 0
    for y in range(y1, y2):
        dx = math.floor((circle[2]**2 - (y-circle[1])**2)**0.5)
        x2 = math.floor(min(circle[0] + dx, multiplier.shape[1] - 1))
        x1 = math.ceil(max(circle[0] - dx, 0))
        multiplier[y, :x1] = 0
        multiplier[y, x2:] = 0
        if x2 - x1 < linlen:
            continue # no reliable transversalium correction, just leave what we have
        if x1 > 0:
            multiplier[y, x1:x1+halflen] = multiplier[y, x1+halflen]
        if x2 < multiplier.shape[1] - 1:
            multiplier[y, x2-halflen:x2] = multiplier[y, x2-halflen-1]
    return multiplier

'''
img: np array
borders: [minX, minY, maxX, maxY]
circle: (centreX, centreY, radius)
reqFlag: 0 if this was a user-requested image, else: 1 if shift = 10, 2 if shift = 0 (non-user requested)
'''
def correct_transversalium2(img, circle, borders, options, reqFlag, basefich):
    y1 = math.ceil(max(circle[1] - circle[2], borders[1]))
    y2 = math.floor(min(circle[1] + circle[2], borders[3]))
    y_ratios_r = [0]
    y_ratios = [0]
    for y in range(y1 + 1, y2):
        dx = math.floor((circle[2]**2 - (y-circle[1])**2)**0.5)
        strip0 = img[y - 1, math.ceil(max(circle[0] - dx, borders[0])) : math.floor(min(circle[0] + dx, borders[2]))]
        strip1 = img[y, math.ceil(max(circle[0] - dx, borders[0])) : math.floor(min(circle[0] + dx, borders[2]))]
        
        rat = np.log(strip1 / strip0)
        y_ratios.append(np.mean(rat))
        y_ratios_r.append(np.mean(reject_outliers(rat)))
        if y % 100 == 0 and 0:
            print(y)
            plt.hist(rat, bins = np.linspace(0.5, 2, 128))
            plt.savefig()
    trend = savgol_filter(y_ratios_r, min(options['trans_strength'], len(y_ratios_r) // 2 * 2 - 1), 3)

    detrended = y_ratios_r - trend # remove trend (smoothed)
    detrended -= np.mean(detrended) # remove DC bias
    correction = np.exp(-np.cumsum(detrended))

    ''' 
    plt.plot(y_ratios)
    plt.plot(y_ratios_r)
    plt.plot(trend)
    plt.show()
    plt.plot(correction)
    plt.show()
    '''

    if options['stubborn_transversalium']:
        ### spurious lines
        c = np.zeros(img.shape[0])
        c[y1:y2] = np.log(correction)
        thresh_spur = np.where(np.abs(c) > np.std(np.log(correction))*2.5)
        spurious_flag = np.abs(c) > np.std(np.log(correction))*2.5
        spurious_flag = np.logical_or(spurious_flag, np.logical_or(np.roll(spurious_flag, -1), np.roll(spurious_flag, 1)))
        img_filt2 = apply_lin_filter(img, 101, 5, spurious_flag, y1, y2, circle)
        return np.minimum(img_filt2, 65535).astype('uint16')
    
    ### new test
    '''
    linlen = 301
    ylen = 15
    kernel = np.zeros((ylen, linlen))
    kernel[:, :] = -1
    kernel[ylen//2, :] = ylen-1#ylen//2

    #kernel[ylen//2+1:, :] = 0
    
    kernel = kernel / np.sum(np.abs(kernel))
    result = cv2.filter2D(np.log(img), -1, kernel)
    plt.imshow(result, cmap='bwr')
    plt.show()

    ###
    fig, axs = plt.subplots(2, 3)
    for i, yr in enumerate([1065, 1500, 2498, 2499, 2500, 2501]):
        axs[i//3, i%3].plot(result[yr, :])
        axs[i//3, i%3].set_title(str(yr))
    plt.show()

    for i, yr in enumerate(list(range(2496, 2505))+[2530, 2470]):
        plt.plot(result[yr, :], label=str(yr))
    plt.legend()
    plt.show()
    '''
    ###
    
    ###
        
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
    options['_transversalium_cache'] = c
    if (not reqFlag) and (not options['clahe_only'] and not options['protus_only']):
        fig = matplotlib.figure.Figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(c)
        ax.set_xlabel('y')
        ax.set_ylabel('transversalium correction factor')
        fig.savefig(output_path(basefich+'_transversalium_correction.png', options), dpi=300)
    ret = (img.T * c).T # multiply each row in image by correction factor

    '''
    ### new test
    linlen = 301
    ylen = 9
    kernel = np.zeros((ylen, linlen))
    kernel[:, :] = -1
    kernel[ylen//2, :] = ylen - 1#ylen//2

    #kernel[ylen//2+1:, :] = 0
    
    kernel = kernel / np.sum(np.abs(kernel))
    result = cv2.filter2D(np.log(ret), -1, kernel)
    plt.imshow(result, cmap='bwr')
    plt.show()
    ###

    correct_kernel = ret / np.exp(result)
    plt.imshow(correct_kernel, cmap='gray')
    plt.show()
    return correct_kernel
    ###
    '''
    
    
    ret[ret > 65535] = 65535 # prevent overflow
    return np.array(ret, dtype='uint16')


def rescale_brightness(img, lo, hi, alpha=1.0):
    sat = np.iinfo(img.dtype).max
    assert(sat >= hi > lo)
    rescaled = (float(sat) * alpha * (img - lo) / (hi - lo)) # convert to float to prevent integer multiplication
    rescaled[rescaled<0] = 0
    rescaled[rescaled>sat] = sat
    return rescaled.astype(img.dtype)

def image_process(frame, cercle, options, header, basefich):
    frame=frame.astype(np.uint16) # make sure we are working with uint16 data
    flag_result_show = options['flag_display']
    # create a CLAHE object (Arguments are optional)
    # clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(5,5))
    clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(2,2))
    cl1 = clahe.apply(frame)
    
    bright = np.percentile(frame, 99.9999) # basically the same as max
    dark_clahe=np.percentile(cl1, 10)
    bright_clahe=np.max(cl1)
    frame_raw    = frame # no rescale
    frame_HC     = rescale_brightness(frame, bright*0.25, bright)             
    frame_protus = rescale_brightness(frame, 0, bright*0.18)
    cc = rescale_brightness(cl1, dark_clahe, bright_clahe)
    if not cercle == (-1, -1, -1) and options['disk_display']:
        x0=int(cercle[0])
        y0=int(cercle[1])
        r=int(cercle[2]) + options['delta_radius']
        if r > 0:
            frame_protus = cv2.circle(frame_protus, (x0,y0),r,80,-1)
    
    # handle rotations
    frame_raw = np.rot90(frame_raw, options['img_rotate']//90, axes=(0,1))
    frame_HC = np.rot90(frame_HC, options['img_rotate']//90, axes=(0,1))
    frame_protus = np.rot90(frame_protus, options['img_rotate']//90, axes=(0,1))
    cc = np.rot90(cc, options['img_rotate']//90, axes=(0,1))

    # save the clahe as a png
    compression = 0
    if not '_nolog' in options: # '_nolog' is used in spectralAnalyser
        if options['clahe_only'] or not options['protus_only']:
            print('saving image to:' + basefich+'_clahe.png')
            cv2.imwrite(output_path(basefich+'_clahe.png', options),cc, [cv2.IMWRITE_PNG_COMPRESSION, compression])
        if options['protus_only'] or not options['clahe_only']:
            cv2.imwrite(output_path(basefich+'_protus.png', options), frame_protus, [cv2.IMWRITE_PNG_COMPRESSION, compression])
        if not options['clahe_only'] and not options['protus_only']:
            # save "high-contrast" and "unconstrasted" pngs
            cv2.imwrite(output_path(basefich+'_uncontrasted.png', options), frame_raw, [cv2.IMWRITE_PNG_COMPRESSION, compression])
            cv2.imwrite(output_path(basefich+'_high_contrast.png', options), frame_HC, [cv2.IMWRITE_PNG_COMPRESSION, compression])
        
    # The 3 images are concatenated together in 1 image => 'Sun images'
    # The 'Sun images' is scaled for the monitor maximal dimension ... it is scaled to match the dimension of the monitor without 
    # changing the Y/X scale of the images 
    if flag_result_show:
        im_3 = cv2.hconcat([cc, frame_HC, frame_protus])
        screen = tk.Tk()
        screensize = screen.winfo_screenwidth(), screen.winfo_screenheight()
        screen.destroy()
        scale = min(screensize[0] / im_3.shape[1], screensize[1] / im_3.shape[0]) * 0.9
        cv2.namedWindow('Sun images', cv2.WINDOW_NORMAL)
        cv2.moveWindow('Sun images', 0, 0)
        cv2.resizeWindow('Sun images',int(im_3.shape[1] * scale), int(im_3.shape[0] * scale))
        cv2.imshow('Sun images',im_3)
        cv2.waitKey(options['tempo'])  # wait a few seconds before deleting the window
        cv2.destroyAllWindows()
    
    if options['save_fit']:
        # save the fits file
        DiskHDU=fits.PrimaryHDU(cl1,header)
        DiskHDU.writeto(output_path(basefich+ '_clahe.fits', options), overwrite='True')
    return (cc, frame_protus)

def removeVignette(frame_circularized, cercle0):
    y_arr = np.percentile(frame_circularized, 90, axis = 0)
    y_arr2 = np.percentile(frame_circularized, 90, axis = 1)
    shrink = 25 # tuning parameter
    start1 = max(0, int(cercle0[0] - cercle0[2] + shrink))
    end1 = min(y_arr.shape[0], int(cercle0[0] + cercle0[2] + 1 - shrink))

    start2 = max(0, int(cercle0[1] - cercle0[2] + shrink))
    end2 = min(y_arr2.shape[0], int(cercle0[1] + cercle0[2] + 1 - shrink))

    y1 = y_arr[start1:end1]
    y2 = y_arr2[start2:end2]

    x1 = np.arange(y1.shape[0]) + start1 - int(cercle0[0])
    x2 = np.arange(y2.shape[0]) + start2 - int(cercle0[1])

    if y1.shape[0] < 20 or y2.shape[0] < 20:
        print("no de-vignette, due to not enough data")
        return frame_circularized
    print("vignette shapes:", y1.shape, y2.shape)
    scale_pix = int(min(y1.shape[0]//2.75, y2.shape[0]//2.75)) // 2 * 2 - 1
    trend1 = savgol_filter(y1, min(801, scale_pix), 3)
    trend2 = savgol_filter(y2, min(801, scale_pix), 3)

    '''
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.plot(x1, trend1)
    plt.plot(x2, trend2)
    plt.show()
    '''
    
    mm = min(np.min(x1), np.min(x2))
    dest = np.zeros((3, int(max(np.max(x1), np.max(x2)) - mm + 1)))
    dest.fill(np.NaN)
    dest[0, :] = np.arange(dest.shape[1]) + mm
    dest[1, int(x1[0] - mm) : int(x1[-1] - mm + 1)] = trend1
    dest[2, int(x2[0] - mm) : int(x2[-1] - mm + 1)] = trend2

    ratio_axes = dest[1, :] / dest[2, :]
    ratio_axes[dest[1, :] == 0] = np.NaN
    ratio_axes[dest[2, :] == 0] = np.NaN

    '''
    plt.plot(dest[0, :], ratio_axes)
    plt.show()
    '''
    
    correction_factor = np.zeros(frame_circularized.shape[0])
    correction_factor.fill(np.NaN)
    correction_factor[dest[0, :].astype(int) + int(cercle0[1])] = ratio_axes
    # forward and backward fill
    for i in range(1, len(correction_factor)):
        if np.isnan(correction_factor[i]):
            correction_factor[i] = correction_factor[i-1]
    for i in range(len(correction_factor) - 2, -1, -1):
        if np.isnan(correction_factor[i]):
            correction_factor[i] = correction_factor[i+1]
    correction_factor = gaussian_filter1d(correction_factor, max(2, min(150, scale_pix//4)))
    '''
    plt.plot(correction_factor)
    plt.plot(y_arr2 / np.max(y_arr2))
    plt.show()
    '''
    return frame_circularized * correction_factor.reshape((-1, 1))
