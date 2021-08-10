"""
@author: Andrew Smith
Version 6 August 2021

"""
from solex_util import *

import skimage
import skimage.feature
import sys

import math
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage import transform
from skimage import filters
from skimage.transform import downscale_local_mean
import cv2

import scipy
from ellipse import LsqEllipse
from matplotlib.patches import Ellipse

NUM_REG = 1 #6 # include biggest NUM_REG regions in fit


def rot(x):
    return np.array([[np.cos(x), np.sin(x)], [-np.sin(x), np.cos(x)]])

def get_correction_matrix(phi, r):
    """
    IN: phi, ellipse axes ratio (height / width)
    OUT: correction matrix
    """
    stretch_matrix = rot(phi) @ np.array([[r, 0], [0, 1]]) @  rot(-phi)
    theta = np.arctan(-stretch_matrix[0, 1] / stretch_matrix[1, 1]) 
    unrotation_matrix = rot(theta)
    correction_matrix = unrotation_matrix @ stretch_matrix
    return np.linalg.inv(correction_matrix), theta

def dofit(points):
    """IN : numpy points coordinates
    OUT : center, width, height, phi, fit informations
    """
    reg = LsqEllipse().fit(points)
    center, width, height, phi = reg.as_parameters()
    return center, width, height, phi, reg.return_fit(n_points=100)


def two_step(points):
    """Launch twice an ellipse fit. One with all edge points, one with only tresholded values.
    IN : numpy array of edge points.
    OUT : np.array(center), height, phi, ratio, points_tresholded, ellipse_points
    """
    center, width, height, phi, _ = dofit(points)
    mat, _ = get_correction_matrix(phi, height / width)
    Xr = mat @ (points - np.array(center)).T * height
    values = np.linalg.norm(Xr, axis = 0) - 1
    #print(np.mean(values), np.std(values), max(values), min(values))
    anomaly_threshold = max(values)
    points_tresholded = points[values > -max(values)]
    center, width, height, phi, ellipse_points = dofit(points_tresholded)
    mat, _ = get_correction_matrix(phi, height / width)
    Xr = mat @ (points_tresholded - np.array(center)).T * height
    values = np.linalg.norm(Xr, axis = 0) - 1
    #print(np.mean(values), np.std(values), max(values), min(values))
    ratio = width / height
    return np.array(center), height, phi, ratio, points_tresholded, ellipse_points

def correct_image(image, phi, ratio, center):
    """correct image geometry. TODO : a rotation is made instead of a tilt
    IN : numpy array, float, float, numpy array (2 elements)
    OUT : numpy array, numpy array (2 elements)
    """
    logme('Y/X ratio : ' + "{:.3f}".format(ratio))
    logme('Tilt angle : ' + "{:.3f}".format(math.degrees(phi)) + " degrees")
    mat, theta = get_correction_matrix(phi, ratio)
    print('unrotation angle theta = ' + "{:.3f}".format(math.degrees(theta)) + " degrees")
    np.set_printoptions(suppress=True)
    logme('Linear transform correction matrix: \n' + str(mat))
    np.set_printoptions(suppress=False)
    mat3 = np.zeros((3, 3))
    mat3[:2, :2] = mat
    mat3[2, 2] = 1
    corners = np.array([[0,0], [0, image.shape[0]], [image.shape[1], 0], [image.shape[1], image.shape[0]]])
    new_corners = (np.linalg.inv(mat) @ corners.T).T # use inverse because we represent mat3 as inverse of transform
    new_h = np.max(new_corners[:, 1]) - np.min(new_corners[:, 1])
    new_w = np.max(new_corners[:, 0]) - np.min(new_corners[:, 0])
    mat3 = mat3 @ np.array([[1, 0, np.min(new_corners[:, 0])], [0, 1, np.min(new_corners[:, 1])], [0,   0,    1]]) # apply translation to prevent clipping
    my_transform = transform.ProjectiveTransform(matrix=mat3)
    corrected_img = transform.warp(image, my_transform, output_shape = (np.ceil(new_h),np.ceil(new_w)), cval  = image[0, 0])
    corrected_img = (2**16*corrected_img).astype(np.uint16) # note : 16-bit output
    new_center = (np.linalg.inv(mat) @ center.T).T - np.array([np.min(new_corners[:, 0]), np.min(new_corners[:, 1])])
    return corrected_img, new_center

def get_flood_image(image):
    """ return an image, where all the pixels brighter than the average
    are made saturated, and all those below average are zeroed.
    IN: original image
    OUT: modified image
    TODO: simplify this function?
    """
    thresh = 0.9 * np.sum(image) / (image.shape[0] * image.shape[1])
    image[image < thresh] = 0
    image[image >= thresh] = 65000
    return image

def get_edge_list(image, sigma = 2):
    """from a picture, return a numpy array containing edge points
    IN : frame as numpy array, integer
    OUT : numpy array
    TODO: simplify this function?
    """
    if sigma <= 0:
        logme('ERROR: could not find any edges')
        return image, (-1, -1, -1)
    
    low_threshold = np.median(cv2.blur(image, ksize=(5, 5))) / 10
    high_threshold = low_threshold*1.5
    print('using thresholds:', low_threshold, high_threshold)
    image = get_flood_image(image)
    edges = skimage.feature.canny(
        image=image,
        sigma=sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    raw_X = np.argwhere(edges)
    labelled, nf = scipy.ndimage.measurements.label(edges, structure  = [[1,1,1], [1,1,1],[1,1,1]])
    if nf == 0:
        return get_edge_list(image, sigma = sigma - 0.5) # try again with less blur, hope it will work
    region_sizes = [-1] + [np.sum(labelled == i) for i in range(1, nf+1)]
    filt = np.zeros(edges.shape)
    for label in sorted(region_sizes, reverse = True)[:min(nf, NUM_REG)]:
        filt[labelled == region_sizes.index(label)] = 1
        
    X = np.argwhere(filt) # find the non-zero pixels
    
    x_min, y_min, x_max, y_max = np.min(X[:, 0]), np.min(X[:, 1]), np.max(X[:, 0]), np.max(X[:, 1])
    dx = x_max - x_min
    dy = y_max - y_min
    crop = 0.015

    mask = np.zeros(filt.shape)
    mask[int(x_min+dx*crop):int(x_max-dx*crop), :] = 1
    filt *= mask
    X = np.argwhere(filt) # find the non-zero pixels again

    x_min, y_min, x_max, y_max = np.min(X[:, 0]), np.min(X[:, 1]), np.max(X[:, 0]), np.max(X[:, 1])
            
    X = np.array(X, dtype='float')
    return np.array([X, raw_X], dtype=object) 

def ellipse_to_circle(image, options):
    """from an entire sun frame, compute ellipse fit and return a circularise picture and center coordinates
    IN : numpy array, dictionnayr of options
    OUt :numpy array, numpy array (2 elements)
    """
    image = image / 65536 # assume 16 bit
    factor = 4
    processed = get_edge_list(downscale_local_mean(image, (factor,factor))) * factor# down-scaled, then upscaled back
    X, raw_X = processed[0], processed[1]
    center, height, phi, ratio, X_f, ellipse_points = two_step(X)
    center = np.array([center[1], center[0]])
    
    fix_img, center = correct_image(image, phi, ratio, center)
    
    if options['flag_display']:
        fig, ax = plt.subplots(ncols=2, nrows = 2)
        ax[0][0].imshow(image, cmap=plt.cm.gray)
        ax[0][0].set_title('uncorrected image', fontsize = 11)
        ax[0][1].imshow(image, cmap=plt.cm.gray)
        ax[0][1].plot(raw_X[:, 1], raw_X[:, 0], 'ro', label = 'edge detection')
        ax[0][1].set_xlim([0, image.shape[1]])
        ax[0][1].set_ylim([0, image.shape[0]])
        ax[0][1].legend()
        ax[1][1].plot(X_f[:, 1], image.shape[0] - X_f[:, 0], 'ro', label = 'filtered edges')
        ax[1][1].plot(ellipse_points[:, 1], image.shape[0] - ellipse_points[:, 0], color='b', label = 'ellipse fit')
        ax[1][1].set_xlim([0, image.shape[1]])
        ax[1][1].set_ylim([0, image.shape[0]])
        ax[1][1].legend()
        ax[1][0].imshow(fix_img, cmap=plt.cm.gray)
        ax[1][0].set_title('geometrically corrected image', fontsize=11)
        ax[0][1].set_title('remember to close this window \n by pressing the "X"', color = 'red')

        #creating a timer object to auto-close plot after some time
        def close_event():
            plt.close()
        timer = fig.canvas.new_timer(interval = options['tempo']) 
        timer.add_callback(close_event)
        timer.start()
        plt.show()
  
    circle = (center[0], center[1], height*ratio) # radius == height*ratio
    return fix_img, circle

    
