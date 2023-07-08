# -*- coding: utf-8 -*-
"""
@author: Valerie Desnoux
with improvements by Andrew Smith
contributors: Jean-Francois Pittet, Jean-Baptiste Butet, Pascal Berteau, Matt Considine
Version 3 July 2023

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


serfiles = []

options = {    
    'shift':[0],                    # argument: w
    'flag_display':False,           # argument: d
    'ratio_fixe' : None,            # argument: x
    'slant_fix' : None ,            #
    'save_fit' : False,             # argument: f
    'clahe_only' : False,           # argument: c
    'disk_display' : True,          # argument: p
    'delta_radius' : 0,             #
    'crop_width_square' : False,    # argument: s
    'transversalium' : True,        # argument: t
    'trans_strength': 301,          #
    'img_rotate': 0,                #
    'flip_x': False,                # argument: m
    'workDir': '',                  #
    'fixed_width': None,            # argument: r
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
        with open(mydir_ini, 'r') as fp:
            global options
            options = json.load(fp)   
    except Exception:
        traceback.print_exc()
        print('note: error reading config file - using default parameters')


def write_ini():
    try:
        print('saving config file ...')
        mydir_ini = os.path.join(os.path.dirname(sys.argv[0]),'SHG_config.txt')
        with open(mydir_ini, 'w') as fp:
            json.dump(options, fp, sort_keys=True, indent=4)
    except Exception:
        traceback.print_exc()
        print('ERROR: failed to write config file: ' + mydir_ini)

def precheck_files(serfiles, options):
    if len(serfiles)==1:
        options['tempo']=60000 #4000 #pour gerer la tempo des affichages des images resultats dans cv2.waitKey
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
            print('ERROR opening file : ', serfile)
            continue
        
        if not good_tasks:
            # save parameters to config file if this is the first good task
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
                serfiles.extend(UI_handler.inputUI(options)) # get files
                handle_files(serfiles, options) # handle files
                serfiles.clear() # clear files that have been processed
                    
        else:
            handle_files(serfiles, options, flag_command_line = True) # use inputs from CLI
            

