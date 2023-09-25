# -*- coding: utf-8 -*-
"""
@author: Andrew Smith
based on code by Valerie Desnoux
contributors: Jean-Francois Pittet, Jean-Baptiste Butet, Pascal Berteau, Matt Considine
Version 24 September 2023

--------------------------------------------------------------
Front end of spectroheliograph processing of SER and AVI files
- interface able to select one or more files
- call to the solex_recon module which processes the sequence and generates the PNG and FITS files
- offers with openCV a display of the resultant image
- wavelength selection with the pixel shift function, including multiple wavelengths and a range of wavelengths
- geometric correction with a fixed Y/X ratio
- if Y/X is blank, then this will be calculated automatically
--------------------------------------------------------------

"""
import math
import numpy as np

import os
import sys
import Solex_recon
import UI_handler
import CLI_handler
from astropy.io import fits
import cProfile
import PySimpleGUI as sg
import traceback
import cv2
import json
import time
from multiprocessing import freeze_support
import glob
import solex_util
import video_reader

serfiles = []

options = {
    'language':'English',           #
    'shift':[0],                    # argument: w
    'flag_display':False,           # argument: d
    'ratio_fixe' : None,            # argument: x
    'slant_fix' : None ,            #
    'save_fit' : False,             # argument: f
    'clahe_only' : False,           # argument: c
    'protus_only': False,
    'disk_display' : True,          # argument: p
    'delta_radius' : 0,             #
    'crop_width_square' : False,    # argument: s
    'transversalium' : True,        # argument: t
    'stubborn_transversalium': False,
    'trans_strength': 301,          #
    'img_rotate': 0,                #
    'flip_x': False,                # argument: m
    'workDir': '',                  #
    'fixed_width': None,            # argument: r
    'output_dir': '',               #
    'input_dir': '',                #
    'specDir': '',                  # for spectral analyser
    'selected_mode': 'File input mode',
    'continuous_detect_mode': False,#
    'dispersion':0.05,              # for spectral analyser
    'ellipse_fit_shift':10,         # secret parameter for ellipse fit
    'de-vignette':False             # remove vigenette
}


'''
open config.txt and read parameters
return parameters from file, or default if file not found or invalid
'''
def read_ini():
    # check for config.txt file for working directory
    print('loading config file...')

    try:
        mydir_ini=os.path.join(os.path.dirname(sys.argv[0]),'SHG_config.txt')
        with open(mydir_ini, 'r', encoding="utf-8") as fp:
            global options
            options.update(json.load(fp)) # if config has missing entries keep default   
    except Exception:
        print('note: error reading config file - using default parameters')


def write_ini():
    try:
        print('saving config file ...')
        mydir_ini = os.path.join(os.path.dirname(sys.argv[0]),'SHG_config.txt')
        with open(mydir_ini, 'w', encoding="utf-8") as fp:
            json.dump(options, fp, sort_keys=True, indent=4)
    except Exception:
        traceback.print_exc()
        print('ERROR: failed to write config file: ' + mydir_ini)

def precheck_files(serfiles, options):
    if len(serfiles)==1:
        options['tempo']=30000 #to manage the display time in ms of result images in cv2.waitKey
    else:
        options['tempo']=5000

    good_tasks = []
    for serfile in serfiles:
        print(serfile)
        if serfile=='':
            print("ERROR filename empty")
            continue
        base = os.path.basename(serfile)
        if base == '':
            print('filename ERROR : ', serfile)
            continue

        # try to open the file to see if it is possible
        try:
            f=open(serfile, "rb")
            f.close()
        except:
            traceback.print_exc()
            print('ERROR opening file : ', serfile)
            continue
        
        if not good_tasks:
            # save parameters to config file if this is the first good task
            if options['selected_mode'] == 'File input mode':
                options['workDir'] = os.path.dirname(serfile)+"/"
            write_ini()
        good_tasks.append((serfile, options.copy()))
    if not good_tasks:
        write_ini() # save to config file if it never happened
    return good_tasks

def handle_files(files, options, flag_command_line = False):
    good_tasks = precheck_files(files, options)
    try : 
       Solex_recon.solex_do_work(good_tasks, flag_command_line)
    except:
        print('ERROR ENCOUNTERED')
        traceback.print_exc()
        cv2.destroyAllWindows() # ? TODO needed?
        if not flag_command_line:
            sg.popup_ok('ERROR message: ' + traceback.format_exc()) # show pop_up of error message

def is_openable(file):
    try:
        f=open(file, "rb")
        f.close()
        rdr = video_reader.video_reader(file)
        return rdr.FrameCount > 0
    except:
        return False

def handle_folder(options):
    if not options['continuous_detect_mode']:
        files_todo = glob.glob(os.path.join(options['input_dir'], '*.ser')) + glob.glob(os.path.join(options['input_dir'], '*.avi'))
        print(f'number of files todo: {len(files_todo)}')
        handle_files(files_todo, options)
        return
    
    files_processed = set()
    layout = [
        [sg.Text('Auto processing of SHG video files', font='Any 12', key='Auto processing of SHG video files'), sg.Push(), sg.Button('Stop')],
        [sg.Text(f'Number of files processed: {len(files_processed)}', key='auto_info'), sg.Push(), sg.Text('Looking for files ...', key='status_info')],
        [sg.Image(UI_handler.resource_path(os.path.join('language_data', 'Black.png')), size=(600, 600), key='_prev_img')],
        [sg.Text('Last: none', key='last')],        
    ]
    window = sg.Window('Continuous processing mode', layout, keep_on_top=True)
    window.finalize()
    stop=False
    
    
    window.perform_long_operation(lambda : time.sleep(0.01), '-END SLEEP-') # dummy function to get started
    prev=None
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        if event == '-END KEY-':
            window['status_info'].update('Looking for files ...')
            window['auto_info'].update(f'Number of files processed: {len(files_processed)}')
            if stop:
                window.close()
                break
            time.sleep(0.1)
            if not prev is None:
                window['_prev_img'].update(data=UI_handler.get_img_data(prev, maxsize=(600,600), first=True))
                window['last'].update('Last: ' + prev)
            window.perform_long_operation(lambda : time.sleep(1), '-END SLEEP-')

        if event == '-END SLEEP-':
            files_todo = glob.glob(os.path.join(options['input_dir'], '*.ser')) + glob.glob(os.path.join(options['input_dir'], '*.avi'))
            files_todo = [x for x in files_todo if not x in files_processed and os.access(x, os.R_OK) and is_openable(x)]
            files_todo = files_todo[:min(1, len(files_todo))] # maximum batch size 1
            if files_todo:
                window['status_info'].update(f'About to process {len(files_todo)} file')
                prev=files_todo[-1]
                prev=os.path.join(solex_util.output_path(os.path.splitext(prev)[0] + f'_shift={options["shift"][-1]}_clahe.png', options)).replace('\\', "/")
                print('the image file:' + str(prev))
                window.perform_long_operation(lambda : handle_files(files_todo, options, True), '-END KEY-')
            else:
                window['status_info'].update('Looking for files ...')
                window.perform_long_operation(lambda : time.sleep(1), '-END KEY-')
            files_processed.update(files_todo)
            
        if event == 'Stop':
            stop=True
            window['status_info'].update(f'WILL STOP AFTER PROCESSING CURRENT BATCH OF {len(files_todo)} FILE(S)')
        
        


"""
-------------------------------------------------------------------------------------------
le programme commence ici !
--------------------------------------------------------------------------------------------
"""
if __name__ == '__main__':
    freeze_support() # enables multiprocessing for py-2-exe
    
    # check for CLI input
    if len(sys.argv)>1: 
        serfiles.extend(CLI_handler.handle_CLI(options))
        
    if 0: #test code for performance test
        read_ini()
        serfiles.extend(UI_handler.inputUI(options))
        cProfile.run('handle_files(serfiles, options)', sort='cumtime')
    else:
        # if no command line arguments, open GUI interface
        if len(serfiles)==0:
            # read initial parameters from config.txt file
            read_ini()
            while True:
                newfiles = UI_handler.inputUI(options) # get files
                if newfiles is None:
                    break # end loop
                serfiles.extend(newfiles) 
                if options['selected_mode'] == 'File input mode':
                    handle_files(serfiles, options) # handle files
                elif options['selected_mode'] == 'Folder input mode':
                    handle_folder(options)
                else:
                    raise Exception('invalid selected_mode: ' + options['selected_mode'])
                serfiles.clear() # clear files that have been processed
            write_ini()       
        else:
            handle_files(serfiles, options, flag_command_line = True) # use inputs from CLI
            

