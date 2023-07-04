# -*- coding: utf-8 -*-
"""
@author: Andrew Smith
contributors: Valerie Desnoux, Jean-Francois Pittet, Jean-Baptiste Butet, Pascal Berteau, Matt Considine
Version 3 July 2023

------------------------------------------------------------------------
Reconstruction of an image from the deviations between the minimum of the line and a reference line
-------------------------------------------------------------------------

"""

from solex_util import *
from video_reader import *
from ellipse_to_circle import ellipse_to_circle, correct_image
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import PySimpleGUI as sg # for progress bar


'''
process files: call solex_read and solex_proc to process a list of files with specified options
input: tasks: list of tuples (file, option)
'''

def solex_do_work(tasks, flag_command_line = False):
    multi = False
    with Pool(4) as p:
        results = []
        for i, (file, options) in enumerate(tasks):
            print('file %s is processing'%file)
            if len(tasks) > 1 and not flag_command_line:  
                sg.one_line_progress_meter('Progress Bar', i, len(tasks), '','Reading file...')
            disk_list, hdr = solex_read(file, options)
            if multi:
                result = p.apply_async(solex_process, args = (options, disk_list, hdr)) # TODO: prints won't be visible inside new thread, can this be fixed?
                results.append(result)
            else:
                solex_process(options, disk_list, hdr)
        [result.get() for result in results]
        if len(tasks) > 1 and not flag_command_line:  
            sg.one_line_progress_meter('Progress Bar', len(tasks), len(tasks), '','Done.')
        
'''
read a solex file and return a list of numpy arrays representing the raw result
'''
def solex_read(file, options):
    basefich0 = os.path.splitext(file)[0] # file name without extension
    options['basefich0'] = basefich0
    clearlog(basefich0 + '_log.txt')
    logme(basefich0 + '_log.txt', 'Pixel shift : ' + str(options['shift']))
    options['shift_requested'] = options['shift']
    options['shift'] = list(dict.fromkeys([10, 0] + options['shift']))  # 10, 0 are "fake", but if they are requested, then don't double count
    rdr = video_reader(file)
    hdr = make_header(rdr)
    ih = rdr.ih
    iw = rdr.iw

    fit, backup_y1, backup_y2 = compute_mean_return_fit(file, options, hdr, iw, ih, basefich0)


    disk_list, ih, iw, FrameCount = read_video_improved(file, fit, options)
    
    hdr['NAXIS1'] = iw  # note: slightly dodgy, new width for subsequent fits file

    # sauve fichier disque reconstruit

    if options['flag_display']:
        cv2.destroyAllWindows()

    for i in range(len(disk_list)):
        if options['flip_x']:
            disk_list[i] = np.flip(disk_list[i], axis = 1)
        basefich = basefich0 + '_shift=' + str(options['shift'][i])
        flag_requested = options['shift'][i] in options['shift_requested']
        
        if options['save_fit'] and flag_requested:
            DiskHDU = fits.PrimaryHDU(disk_list[i], header=hdr)
            DiskHDU.writeto(basefich + '_raw.fits', overwrite='True')
    return disk_list, hdr
    
'''
process the raw disks: circularise, detransversalium, crop, and adjust contrast
'''
def solex_process(options, disk_list, hdr):
    basefich0 = options['basefich0']
    if options['transversalium']:
        logme(basefich0 + '_log.txt', 'Transversalium correction : ' + str(options['trans_strength']))
    else:
        logme(basefich0 + '_log.txt', 'transversalium disabled')
    logme(basefich0 + '_log.txt', 'Mirror X : ' + str(options['flip_x']))
    logme(basefich0 + '_log.txt', 'Post-rotation : ' + str(options['img_rotate']) + ' degrees')
    logme(basefich0 + '_log.txt', f'Protus adjustment : {options["delta_radius"]}')
    borders = [0,0,0,0]
    cercle0 = (-1, -1, -1)
    frames_circularized = []
    for i in range(len(disk_list)):
        flag_requested = options['shift'][i] in options['shift_requested']
        basefich = basefich0 + '_shift=' + str(options['shift'][i])
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
            if flag_requested:
                frame_circularized = correct_image(disk_list[i] / 65536, phi, ratio, np.array([-1.0, -1.0]), -1.0, options['basefich0'], print_log=i == 0)[0]  # Note that we assume 16-bit

        if not flag_requested:
            continue # skip processing if shift is not desired
        
        if options['save_fit']:  # first two shifts are not user specified
            DiskHDU = fits.PrimaryHDU(frame_circularized, header=hdr)
            DiskHDU.writeto(basefich + '_circular.fits', overwrite='True')


        if options['transversalium']:
            if not cercle0 == (-1, -1, -1):
                detransversaliumed = correct_transversalium2(frame_circularized, cercle0, borders, options, 0, basefich)
            else:
                detransversaliumed = correct_transversalium2(frame_circularized, (0,0,99999), [0, backup_y1+20, frame_circularized.shape[1] -1, backup_y2-20], options, 0, basefich)
        else:
            detransversaliumed = frame_circularized

        if options['save_fit'] and options['transversalium']:  # first two shifts are not user specified
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
       

        image_process(detransversaliumed, cercle, options, hdr, basefich)
        write_complete(basefich0 + '_log.txt')
