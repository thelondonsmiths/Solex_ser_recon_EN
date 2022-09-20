"""
@author: Andrew Smith
contributors: Valerie Desnoux, Jean-Francois Pittet
Version 21 September 2022

"""
from numpy import polynomial
from solex_util import *

import skimage
import skimage.feature
import sys

import math
import numpy as np
import matplotlib.figure
import matplotlib.pyplot

from skimage import data
from skimage import transform
from skimage import filters
from skimage.transform import downscale_local_mean
import cv2

import scipy
from ellipse import LsqEllipse
from matplotlib.patches import Ellipse

NUM_REG = 2  # 6 # include biggest NUM_REG regions in fit
             # for multiple full-disk scans this must be changed to 1


def rot(x):
    return np.array([[np.cos(x), np.sin(x)], [-np.sin(x), np.cos(x)]])


def get_correction_matrix(phi, r):
    """
    IN: phi, ellipse axes ratio (height / width)
    OUT: correction matrix
    """
    stretch_matrix = rot(phi) @ np.array([[r, 0], [0, 1]]) @  rot(-phi)
    theta = np.arctan(stretch_matrix[1, 0] / stretch_matrix[0, 0])
    unrotation_matrix = rot(theta)
    correction_matrix = unrotation_matrix @ stretch_matrix
    correction_matrix[1, 0] = 0  # set the bottom-left element to exactly zero
    correction_matrix /= correction_matrix[1, 1] # set bottom-right element to exactly one
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
    values = np.linalg.norm(Xr, axis=0) - 1
    #print(np.mean(values), np.std(values), max(values), min(values))
    anomaly_threshold = max(values)
    points_tresholded = points[values > -max(values)]
    center, width, height, phi, ellipse_points = dofit(points_tresholded)
    mat, _ = get_correction_matrix(phi, height / width)
    Xr = mat @ (points_tresholded - np.array(center)).T * height
    values = np.linalg.norm(Xr, axis=0) - 1
    #print(np.mean(values), np.std(values), max(values), min(values))
    ratio = width / height
    #try to get phi close to 0 (within pi/4) by swapping the semi-major and semi-minor axes labels
    for _ in range(2):
        if phi > math.pi / 4:
            phi -= math.pi/2
            ratio = 1/ratio
            height = height / ratio
        if phi < -math.pi / 4:
            phi += math.pi/2
            ratio = 1/ratio
            height = height / ratio
    return np.array(
        center), height, phi, ratio, points_tresholded, ellipse_points

# note: height is actually an ellipse axis
def correct_image(image, phi, ratio, center, height, print_log=False):
    """correct image geometry. TODO : a rotation is made instead of a tilt
    IN : numpy array, float, float, numpy array (2 elements)
    OUT : numpy array, numpy array (2 elements)
    """

    mat, theta = get_correction_matrix(phi, ratio) 
    mat3 = np.zeros((3, 3))
    mat3[:2, :2] = mat
    mat3[2, 2] = 1
    corners = np.array([[0, 0], [0, image.shape[0]], [image.shape[1], 0], [
                       image.shape[1], image.shape[0]]])
    # use inverse because we represent mat3 as inverse of transform
    new_corners = (np.linalg.inv(mat) @ corners.T).T
    new_h = np.max(new_corners[:, 1]) - np.min(new_corners[:, 1])
    new_w = np.max(new_corners[:, 0]) - np.min(new_corners[:, 0])
    mat3 = mat3 @ np.array([[1, 0, np.min(new_corners[:, 0])], [0, 1, np.min(
        new_corners[:, 1])], [0, 0, 1]])  # apply translation to prevent clipping
    my_transform = transform.ProjectiveTransform(matrix=mat3)
    corrected_img = transform.warp(image, my_transform, output_shape=(
        np.ceil(new_h), np.ceil(new_w)), cval=image[0, 0])
    corrected_img = (
        2**16 *
        corrected_img).astype(
        np.uint16)  # note : 16-bit output
    new_center = (np.linalg.inv(mat) @ center.T).T - \
        np.array([np.min(new_corners[:, 0]), np.min(new_corners[:, 1])])
    
    new_radius = height * np.sqrt(np.abs(ratio / np.linalg.det(mat))) # derivation: area of a circle / area of an ellipse
    if print_log:
        print(
            'unrotation angle theta = ' +
            "{:.3f}".format(
                math.degrees(theta)) +
            " degrees")
        np.set_printoptions(suppress=True)
        logme('Y/X ratio : ' + "{:.3f}".format(ratio))
        logme(
            'Tilt angle : ' +
            "{:.3f}".format(
                math.degrees(phi)) +
            " degrees")
        logme('Linear transform correction matrix : \n' + str(mat))
        logme('Disk position, radius : ' + ((str(new_center) + ', ' + "{:.3f}".format(new_radius)) if not height == -1.0 else 'UNKNOWN'))
        logme ('Unrotation : '  +
            "{:.3f}".format(
                math.degrees(theta)) +
            " degrees")
        np.set_printoptions(suppress=False)
    return corrected_img, (new_center[0], new_center[1], new_radius), mat3


def get_flood_image(image):
    """
    Return an image, where all the pixels brighter than a threshold
    are made saturated, and all those below average are zeroed.
    the threshhold is chosen as the local minimum of a cubic polynomial fit of the pixel-brightness
    histogram of the image. As a backup, the average brightness is used if
    a local minimum cannot be found.
    IN: original image
    OUT: modified image
    """

    thresh = 0.9 * np.sum(image) / (image.shape[0] * image.shape[1])
    print('thresh=', thresh)
    img_blurred = cv2.blur(image, ksize=(5, 5))
    n, bins = np.histogram(img_blurred.flatten(), bins=20)
    '''
    plt.hist(img_blurred.flatten(), bins=20)
    plt.show()
    '''
    # fit the histogram to a cubic graph
    coeff = polynomial.polynomial.Polynomial.fit(bins[1:], n, 3).convert().coef
    print('cubic fit coeffs. :', coeff)

    d, c, b, a = coeff

    def gf(x):
        return a * x**3 + b * x**2 + c * x + d

    '''
    y_fitted = [gf(x) for x in bins[1:]]
    plt.plot(bins[1:], y_fitted)
    plt.plot(bins[1:], n)
    plt.show()
    '''

    # derivative is 3 * a * x**2 + 2 * b * x + c
    # stationary points at {-2b +/- sqrt(4*b**2-12*a*c)} / 6a

    sign = 1  # for local minimum
    discriminant = 4 * b**2 - 12 * a * c
    if discriminant >= 0:
        minimum = (-2 * b + sign * np.sqrt(discriminant)) / (6 * a)
        thresh2 = minimum
    else:
        print('WARNING: cubic fit failed: no local minimum (falling back to mean threshhold)')
        thresh2 = thresh

    print('thresh2=', thresh2)
    start_i = -1
    for i in range(len(bins) - 1):
        if bins[i] <= thresh2 < bins[i + 1]:
            start_i = i
    if start_i == -1:
        print('WARNING: fail to get start_i from cubic fit (default to average thresh)')
        fail_flag = 1
        thresh3 = thresh
    else:
        i = start_i
        while i > 0 and i < len(bins) - 2:
            if n[i - 1] < n[i]:
                i -= 1
            elif n[i + 1] < n[i]:
                i += 1
            else:
                break
        if i >= 1:
            i -= 1  # make circle a little bigger
        thresh3 = bins[i]

    print('thresh3 = ', thresh3)
    img_blurred[img_blurred < thresh3] = 0
    img_blurred[img_blurred >= thresh3] = 65000
    return img_blurred


def get_edge_list(image, sigma=2):
    """from a picture, return a numpy array containing edge points
    IN : frame as numpy array, integer
    OUT : numpy array
    TODO: simplify this function?
    """
    if sigma <= 0:
        logme('ERROR: could not find any edges')
        return image, (-1, -1, -1)

    low_threshold = np.median(cv2.blur(image, ksize=(5, 5))) / 10
    high_threshold = low_threshold * 1.5
    print('using thresholds:', low_threshold, high_threshold)
    image_flooded = get_flood_image(image)
    edges = skimage.feature.canny(
        image=image_flooded,
        sigma=sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    raw_X = np.argwhere(edges)
    labelled, nf = scipy.ndimage.measurements.label(
        edges, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    if nf == 0:
        # try again with less blur, hope it will work
        return get_edge_list(image, sigma=sigma - 0.5)
    region_sizes = [-1] + [np.sum(labelled == i) for i in range(1, nf + 1)]
    filt = np.zeros(edges.shape)
    for label in sorted(region_sizes, reverse=True)[:min(nf, NUM_REG)]:
        filt[labelled == region_sizes.index(label)] = 1

    X = np.argwhere(filt)  # find the non-zero pixels

    x_min, y_min, x_max, y_max = np.min(X[:, 0]), np.min(
        X[:, 1]), np.max(X[:, 0]), np.max(X[:, 1])
    dx = x_max - x_min
    dy = y_max - y_min
    crop = 0.017  # was : 0.015

    mask = np.zeros(filt.shape)
    mask[int(x_min + dx * crop):int(x_max - dx * crop), :] = 1
    filt *= mask
    X = np.argwhere(filt)  # find the non-zero pixels again

    x_min, y_min, x_max, y_max = np.min(X[:, 0]), np.min(
        X[:, 1]), np.max(X[:, 0]), np.max(X[:, 1])

    X = np.array(X, dtype='float')
    return np.array([X, raw_X], dtype=object)


def ellipse_to_circle(image, options, basefich):
    """from an entire sun frame, compute ellipse fit and return a circularise picture and center coordinates
    IN : numpy array, dictionnayr of options
    OUt :numpy array, numpy array (2 elements)
    """
    image = image / 65536  # assume 16 bit
    factor = 4
    processed = get_edge_list(downscale_local_mean(
        image, (factor, factor))) * factor  # down-scaled, then upscaled back
    X, raw_X = processed[0], processed[1]
    center, height, phi, ratio, X_f, ellipse_points = two_step(X)
    center = np.array([center[1], center[0]])

    fix_img, new_circle, mat3 = correct_image(image, phi, ratio, center, height, print_log=True)


    X_f3 = np.ones((X_f.shape[0], 3))
    X_f3[:, 1] = X_f[:, 0]  # note that X_f is in (y, x) form while X_f3 is in (x, y)
    X_f3[:, 0] = X_f[:, 1]
    X_f3_t = (np.linalg.inv(mat3) @ X_f3.T).T
    borders = [np.min(X_f3_t[:, 0]), np.min(X_f3_t[:, 1]), np.max(X_f3_t[:, 0]), np.max(X_f3_t[:, 1])]
    print('sun borders found:' + str(borders))
    if not options['clahe_only']:
        fig = matplotlib.figure.Figure()
        ax = [[fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2)], [fig.add_subplot(2, 2, 3), fig.add_subplot(2, 2, 4)]]
        #fig, ax = plt.subplots(ncols=2, nrows=2)
        fig.tight_layout()
        ax[0][0].imshow(image, cmap=matplotlib.pyplot.cm.gray)
        ax[0][0].set_title('uncorrected image', fontsize=11)
        ax[0][0].set_aspect('equal')
        ax[0][1].set_aspect('equal')
        ax[0][1].imshow(image, cmap=matplotlib.pyplot.cm.gray)
        ax[0][1].plot(raw_X[:, 1], raw_X[:, 0], 'ro', label='edge detection')
        ax[0][1].legend()
        ax[1][1].set_aspect('equal')
        ax[1][1].plot(X_f[:, 1], X_f[:, 0], 'ro', label='filtered edges')
        ax[1][1].plot(ellipse_points[:, 1], ellipse_points[:, 0],
                      color='b', label='ellipse fit')
        ax[1][1].set_ylim([image.shape[0], 0])  # make y-axis upside-down
        ax[1][1].legend()
        ax[1][0].set_aspect('equal')
        ax[1][0].imshow(fix_img, cmap=matplotlib.pyplot.cm.gray)
        ax[1][0].axhline(y=borders[1])
        ax[1][0].axhline(y=borders[3])
        ax[1][0].axvline(x=borders[0])
        ax[1][0].axvline(x=borders[2])
        ax[1][0].set_title('geometrically corrected image', fontsize=11)    
        fig.savefig(basefich + '_ellipse_fit.png', dpi=300)
  
    return fix_img, new_circle, ratio, phi, borders
