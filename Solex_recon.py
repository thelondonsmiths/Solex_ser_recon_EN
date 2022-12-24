# -*- coding: utf-8 -*-
"""
@author: Andrew Smith
contributors: Valerie Desnoux, Jean-Francois Pittet, Jean-Baptiste Butet, Pascal Berteau, Matt Considine
Version 6 November 2022

------------------------------------------------------------------------
Reconstruction of an image from the deviations between the minimum of the line and a reference line
-------------------------------------------------------------------------

"""

from solex_util import *
from video_reader import *
from ellipse_to_circle import ellipse_to_circle, correct_image
import json


def solex_proc(file, options):
    clearlog()
    logme('Pixel shift : ' + str(options['shift']))
    options['shift'] = [10, 0] + options['shift']  # 10, 0 are "fake"
    WorkDir = os.path.dirname(file) + "/"
    os.chdir(WorkDir)
    base = os.path.basename(file)
    basefich0 = os.path.splitext(base)[0]
    rdr = video_reader(file)
    hdr = make_header(rdr)
    ih = rdr.ih
    iw = rdr.iw

    fit, backup_y1, backup_y2 = compute_mean_return_fit(file, options, hdr, iw, ih, basefich0)

    ####adding binning information###
    with open('camera_list.json') as json_file:
            cameras = json.load(json_file)
            bin_text = '0'
            for key in cameras.keys():
                if key in rdr.Observer or key in rdr.Telescope or key in rdr.Instrument :
                    logme(f'camera information found, your camera is a {key}')
                    bin_text = '_bin'+str(round(int(cameras[key])//rdr.Width,0))
                    break
            if bin_text == '0':
                logme('camera information not found. If width is <2000, bin2 is guessing')
                if rdr.Width <2000 :
                    bin_text = '_bin2'
                else :
                    bin_text = '_bin1'
    basefich0+=bin_text
    ##################################

    disk_list, ih, iw, FrameCount = read_video_improved(file, fit, options)
    
    hdr['NAXIS1'] = iw  # note: slightly dodgy, new width

    # sauve fichier disque reconstruit

    if options['flag_display']:
        cv2.destroyAllWindows()

    if options['transversalium']:
        logme('Transversalium correction : ' + str(options['trans_strength']))
    else:
        logme('transversalium disabled')
    logme('Mirror X : ' + str(options['flip_x']))
    logme('Post-rotation : ' + str(options['img_rotate']) + ' degrees')
    logme(f'Protus adjustment : {options["delta_radius"]}')
    borders = [0,0,0,0]
    cercle0 = (-1, -1, -1)
    frames_circularized = []
    for i in range(len(disk_list)):
        if options['flip_x']:
            disk_list[i] = np.flip(disk_list[i], axis = 1)
        basefich = basefich0 + '_shift=' + str(options['shift'][i])
        if options['save_fit'] and i >= 2:
            DiskHDU = fits.PrimaryHDU(disk_list[i], header=hdr)
            DiskHDU.writeto(basefich + '_raw.fits', overwrite='True')

        """
        We now apply ellipse_fit to apply the geometric correction

        """
        # disk_list[0] is always shift = 10, for more contrast for ellipse fit
        if options['ratio_fixe'] is None and options['slant_fix'] is None:
            frame_circularized, cercle0, options['ratio_fixe'], phi, borders = ellipse_to_circle(
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
            if not cercle0 == (-1, -1, -1):
                detransversaliumed = correct_transversalium2(frame_circularized, cercle0, borders, options, i >= 2, basefich)
            else:
                detransversaliumed = correct_transversalium2(frame_circularized, (0,0,99999), [0, backup_y1+20, frame_circularized.shape[1] -1, backup_y2-20], options, i >= 2, basefich)
        else:
            detransversaliumed = frame_circularized

        if options['save_fit'] and i >= 2 and options['transversalium']:  # first two shifts are not user specified
            DiskHDU = fits.PrimaryHDU(detransversaliumed, header=hdr)
            DiskHDU.writeto(basefich + '_detransversaliumed.fits', overwrite='True')

        cercle = cercle0
        if not options['fixed_width'] == None or options['crop_width_square']:
            h, w = detransversaliumed.shape
            nw = h if options['fixed_width'] == None else options['fixed_width'] # new width
            nw2 = nw // 2
            cx = w // 2 if cercle == (-1, -1, -1) else int(cercle[0])
            tx = nw2 - cx
            new_img = np.full((h, nw), detransversaliumed[0, 0], dtype=detransversaliumed.dtype)

            new_img[:, :min(cx + nw2, detransversaliumed.shape[1]) - max(0, cx - nw2)] = detransversaliumed[:, max(0, cx - nw2) : min(cx + nw2, detransversaliumed.shape[1])]
            
            if tx > 0:
                new_img = np.roll(new_img, tx, axis = 1)
                new_img[:, :tx] = detransversaliumed[0, 0]

            if not cercle == (-1, -1, -1):
                cercle = (nw2, cercle[1], cercle[2])
            detransversaliumed = new_img    
       

        if i >= 2:
            image_process(detransversaliumed, cercle, options, hdr, basefich)
        
        
    with open(basefich0 + '_log.txt', "w") as logfile:
        logfile.writelines(mylog)

    return frames_circularized[2:], hdr, cercle
