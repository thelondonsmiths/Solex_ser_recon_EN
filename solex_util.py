"""
@author: Valerie Desnoux
with improvements by Andrew Smith
Version 8 September 2021

"""

import numpy as np
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

mylog = []


def clearlog():
    mylog.clear()


def logme(s):
    print(s)
    mylog.append(s + '\n')


def detect_bord(img, axis):

    ymean = np.mean(img, axis)
    ysmooth = gaussian_filter1d(ymean, 21)
    '''
    ysmooth2 = savgol_filter(ymean, 301, 3)
    plt.plot(ymean)
    plt.plot(ysmooth)
    plt.plot(ysmooth2)
    plt.show()
    plt.plot(ysmooth/ymean)
    plt.plot(ysmooth2/ymean)
    plt.show()
    '''
    ygrad = np.gradient(ysmooth)
    return ygrad.argmax(), ygrad.argmin()
