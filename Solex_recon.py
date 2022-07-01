# -*- coding: utf-8 -*-
"""
@author: Valerie Desnoux
with improvements by Andrew Smith
contributors: Jean-Francois Pittet, Jean-Baptiste Butet, Pascal Berteau, Matt Considine
Version 30 June 2022

------------------------------------------------------------------------
Reconstruction of an image from the deviations between the minimum of the line and a reference line
-------------------------------------------------------------------------

"""

from solex_util import *
from video_reader import *
from ellipse_to_circle import ellipse_to_circle, correct_image


# read video and return constructed image of sun using fit and LineRecal
def read_video_improved(file, fit, LineRecal, options):
    rdr = video_reader(file)
    ih, iw = rdr.ih, rdr.iw
    FrameMax = rdr.FrameCount
    disk_list = [np.zeros((ih, FrameMax), dtype='uint16')
                 for _ in options['shift']]

    if options['flag_display']:
        cv2.namedWindow('disk', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('disk', FrameMax // 3, ih // 3)
        cv2.moveWindow('disk', 200, 0)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.moveWindow('image', 0, 0)
        cv2.resizeWindow('image', int(iw), int(ih))

    col_indeces = []

    for shift in options['shift']:
        ind_l = (np.asarray(fit)[:, 0] + np.ones(ih)
                 * (LineRecal + shift)).astype(int)

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
        if options['flag_display'] and rdr.FrameIndex % 10 == 0:
            cv2.imshow('image', img)
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                sys.exit()

        for i in range(len(options['shift'])):
            ind_l, ind_r = col_indeces[i]
            left_col = img[np.arange(ih), ind_l]
            right_col = img[np.arange(ih), ind_r]
            IntensiteRaie = left_col * left_weights + right_col * right_weights
            disk_list[i][:, rdr.FrameIndex] = IntensiteRaie

        if options['flag_display'] and rdr.FrameIndex % 10 == 0:
            # disk_list[1] is always shift = 0
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


def compute_mean_return_fit(file, options, LineRecal=1):
    global hdr, ih, iw

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
        cv2.namedWindow('Ser mean', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Ser mean', iw, ih)
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
    logme('spectral line polynomial fit: ' + str(p))
    curve = polyval(np.asarray(np.arange(ih), dtype='d'), p)
    fit = [[math.floor(curve[y]) - LineRecal, curve[y] - math.floor(curve[y]), y] for y in range(ih)]
    if not options['clahe_only']:
        fig, ax = plt.subplots()
        ax.imshow(mean_img, cmap=plt.cm.gray)
        s = (y2-y1)//20 + 1
        ax.plot(min_intensity[y1:y2:s], np.arange(y1, y2, s), 'rx', label='line detection')
        ax.plot(curve, np.arange(ih), label='polynomial fit')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_aspect(0.1)
        plt.tight_layout()
        plt.savefig(basefich0+'_spectral_line_data.png', dpi=400)
        plt.close()
    return fit, y1, y2

'''
img: np array
borders: [minX, minY, maxX, maxY]
cirlce: (centreX, centreY, radius)
not_fake: true/false on if this was a user-requested image
'''

def correct_transversalium2(img, circle, borders, options, not_fake):
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

    a = 0.05
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
        plt.plot(c)
        plt.xlabel('y')
        plt.ylabel('transversalium correction factor')
        plt.savefig(basefich+'_transversalium_correction.png')
        plt.close()

    ret = (img.T * c).T # multiply each row in image by correction factor
    ret[ret > 65535] = 65535 # prevent overflow
    return np.array(ret, dtype='uint16') 
    
def solex_proc(file, options):
    global hdr, ih, iw, basefich0, basefich
    clearlog()
    # plt.gray()              #palette de gris si utilise matplotlib pour visu
    # debug
    logme('Using pixel shift : ' + str(options['shift']))
    options['shift'] = [10, 0] + options['shift']  # 10, 0 are "fake"
    WorkDir = os.path.dirname(file) + "/"
    os.chdir(WorkDir)
    base = os.path.basename(file)
    basefich0 = os.path.splitext(base)[0]
    LineRecal = 1
    rdr = video_reader(file)
    hdr = make_header(rdr)
    ih = rdr.ih
    iw = rdr.iw

    fit, backup_y1, backup_y2 = compute_mean_return_fit(file, options, LineRecal)


    disk_list, ih, iw, FrameCount = read_video_improved(
        file, fit, LineRecal, options)

    hdr['NAXIS1'] = iw  # note: slightly dodgy, new width

    # sauve fichier disque reconstruit

    if options['flag_display']:
        cv2.destroyAllWindows()

    if options['transversalium']:
        logme('transversalium correction strength: ' + str(options['trans_strength']))
    else:
        logme('transversalium disabled')
        
    borders = [0,0,0,0]
    cercle = (-1, -1, -1)
    frames_circularized = []
    for i in range(len(disk_list)):
        basefich = basefich0 + '_shift=' + str(options['shift'][i])
        if options['save_fit'] and i >= 2:
            DiskHDU = fits.PrimaryHDU(disk_list[i], header=hdr)
            DiskHDU.writeto(basefich + '_raw.fits', overwrite='True')

        """
        We now apply ellipse_fit to apply the geometric correction

        """
        # disk_list[0] is always shift = 10, for more contrast for ellipse fit
        if options['ratio_fixe'] is None and options['slant_fix'] is None:
            frame_circularized, cercle, options['ratio_fixe'], phi, borders = ellipse_to_circle(
                disk_list[i], options, basefich)
            # in options angles are stored as degrees (slightly annoyingly)
            options['slant_fix'] = math.degrees(phi)

        else:
            ratio = options['ratio_fixe'] if not options['ratio_fixe'] is None else 1.0
            phi = math.radians(options['slant_fix']) if not options['slant_fix'] is None else 0.0
            frame_circularized = correct_image(disk_list[i] / 65536, phi, ratio, np.array([-1.0, -1.0]), -1.0, print_log=i == 0)[0]  # Note that we assume 16-bit


        if options['save_fit'] and i >= 2:  # first two shifts are not user specified
            DiskHDU = fits.PrimaryHDU(frame_circularized, header=hdr)
            DiskHDU.writeto(basefich + '_circular.fits', overwrite='True')


        if options['transversalium']:
            if not cercle == (-1, -1, -1):
                detransversaliumed = correct_transversalium2(frame_circularized, cercle, borders, options, i >= 2)
            else:
                detransversaliumed = correct_transversalium2(frame_circularized, (0,0,99999), [0, backup_y1+20, frame_circularized.shape[1] -1, backup_y2-20], options, i >= 2)
        else:
            detransversaliumed = frame_circularized

        if options['save_fit'] and i >= 2 and options['transversalium']:  # first two shifts are not user specified
            DiskHDU = fits.PrimaryHDU(detransversaliumed, header=hdr)
            DiskHDU.writeto(basefich + '_detransversaliumed.fits', overwrite='True')
        
        frames_circularized.append(detransversaliumed)
        
    with open(basefich0 + '_log.txt', "w") as logfile:
        logfile.writelines(mylog)

    return frames_circularized[2:], hdr, cercle
