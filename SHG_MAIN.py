# -*- coding: utf-8 -*-
"""
@author: Valerie Desnoux
with improvements by Andrew Smith
contributors: Jean-Francois Pittet, Jean-Baptiste Butet, Pascal Berteau, Matt Considine
Version 4 July 2022

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
import Solex_recon as sol
from astropy.io import fits
import cProfile
import PySimpleGUI as sg
import traceback
import cv2

def usage():
    usage_ = "SHG_MAIN.py [-dcfpstwm] [file(s) to treat, * allowed]\n"
    usage_ += "'d' : 'flag_display', display all graphics (False by default)\n"
    usage_ += "'c' : 'clahe_only',  only final clahe image is saved (False by default)\n"
    usage_ += "'f' : 'save_fit', save all fits files (False by default)\n"
    usage_ += "'m' : 'mirror flip', mirror flip in x-direction (False by default)\n"
    usage_ += "'p' : 'disk_display' disable black disk on protuberance images (False by default)\n"
    usage_ += "'s' : 'crop_square_width', crop the width to equal the height (False by default)\n"
    usage_ += "'t' : 'disable transversalium', disable transversalium correction (False by default)\n"
    usage_ += "'w' : 'a,b,c'  produce images at a, b and c pixels.\n"
    usage_ += "'w' : 'x:y:w'  produce images starting at x, finishing at y, every w pixels."
    return usage_
    
def treat_flag_at_cli(arguments):
    global options
    #reading arguments
    i=0
    while i < len(argument[1:]): #there's a '-' at first)
        character = argument[1:][i]
        if character=='h':
            print(usage())
            sys.exit()
        elif character=='w' :
            #find characters for shifting
            shift=''
            stop = False
            try : 
                while not stop : 
                    if argument[1:][i+1].isdigit() or argument[1:][i+1]==':' or argument[1:][i+1]==',' or argument[1:][i+1]=='-': 
                        shift+=argument[1:][i+1]
                        i+=1
                    else : 
                        i+=1
                        stop=True
            except IndexError :
                i+=1 #the reach the end of arguments.
            shift_choice = shift.split(':')
            if len(shift_choice) == 1:
                options['shift'] = list(map(int, [x.strip() for x in shift.split(',')]))
            elif len(shift_choice) == 2:
                options['shift'] = list(range(int(shift_choice[0].strip()), int(shift_choice[1].strip())+1))
            elif len(shift_choice) == 3:
                options['shift'] = list(range(int(shift_choice[0].strip()), int(shift_choice[1].strip())+1, int(shift_choice[2].strip())))
            else:
                print('invalid shift input')
                sys.exit()
        elif character=='t':
            options['transversalium'] = False
            i+=1
        elif character=='c':
            options['clahe_only'] = False
            i+=1
        elif character=='p':
            options['disk_display'] = False
            i+=1
        else : 
            try : #all others
                options[flag_dictionnary[character]]=True if flag_dictionnary.get(character) else False
                i+=1
            except KeyError : 
                print('ERROR !!! At least one argument is not accepted')
                print(usage())
                i+=1
    print('options %s'%(options))

def UI_SerBrowse (WorkDir, default_graphics, default_fits, default_clahe_only, default_crop_square, default_transversalium, default_transversalium_strength, default_rotation):
    sg.theme('Dark2')
    sg.theme_button_color(('white', '#500000'))
    
    layout = [
    [sg.Text('File(s)', size=(5, 1)), sg.InputText(default_text='',size=(75,1),key='-FILE-'),
     sg.FilesBrowse('Open',file_types=(("SER Files", "*.ser"),("AVI Files", "*.avi"),),initial_folder=WorkDir)],
    [sg.Checkbox('Show graphics', default=default_graphics, key='-DISP-')],
    [sg.Checkbox('Save fits files', default=default_fits, key='-FIT-')],
    [sg.Checkbox('Save clahe.png only', default=default_clahe_only, key='-CLAHE_ONLY-')],
    [sg.Checkbox('Crop square', default=default_crop_square, key='-crop_width_square-')],
    [sg.Checkbox('Mirror X', default=False, key='-flip_x-')],
    [sg.Text("Rotate png images:", key='img_rotate_slider')],
    [sg.Slider(range=(0,270),
         default_value=default_rotation,
         resolution=90,     
         size=(25,15),
         orientation='horizontal',
         font=('Helvetica', 12),
         key='img_rotate')],
    [sg.Checkbox('Correct transversalium lines', default=default_transversalium, key='-transversalium-', enable_events=True)],
    [sg.Text("Transversalium correction strength (pixels x 100) :", key='text_trans', visible=default_transversalium)],
    [sg.Slider(range=(0.5,7),
         default_value=default_transversalium_strength,
         resolution=0.5,     
         size=(25,15),
         orientation='horizontal',
         font=('Helvetica', 12),
         key='-trans_strength-',
         visible=default_transversalium)],
    [sg.Text('Y/X ratio (blank for auto)', size=(25,1)), sg.Input(default_text='', size=(8,1),key='-RATIO-')],
    [sg.Text('Tilt angle (blank for auto)',size=(25,1)),sg.Input(default_text='',size=(8,1),key='-SLANT-',enable_events=True)],
    [sg.Text('Pixel offset',size=(25,1)),sg.Input(default_text='0',size=(8,1),tooltip= "a,b,c will produce images at a, b and c\n x:y:w will produce images starting at x, finishing at y, every w pixels",key='-DX-',enable_events=True)],
    [sg.Text('Protus adjustment', size=(25,1)), sg.Input(default_text='0', size=(8,1), tooltip = 'make the black circle bigger or smaller by inputting an integer', key='-delta_radius-')],
    [sg.Button('OK'), sg.Cancel()]
    ] 
    
    window = sg.Window('Processing', layout, finalize=True)
    window['-FILE-'].update(WorkDir) 
    window.BringToFront()
    
    
    while True:
        event, values = window.read()
        if event==sg.WIN_CLOSED or event=='Cancel': 
            sys.exit()
        
        if event=='OK':
            if not values['-FILE-'] == WorkDir and not values['-FILE-'] == '':
                try:
                    serfiles, options = interpret_UI_values(values)
                    window.close()
                    return serfiles, options
                except Exception as inst:
                    sg.Popup('Error: ' + inst.args[0], keep_on_top=True)
                    
            else:
                # display pop-up file not entered
                sg.Popup('Error: file not entered! Please enter file(s)', keep_on_top=True)
        window.Element('-trans_strength-').Update(visible = values['-transversalium-'])
        window.Element('text_trans').Update(visible = values['-transversalium-'])    

    

'''
open SHG.ini and read parameters
return parameters from file, or default if file not found or invalid
'''
def read_ini():
    # check for .ini file for working directory           
    try:
        mydir_ini=os.path.dirname(sys.argv[0])+'/SHG.ini'
        with open(mydir_ini, "r") as f1:   
            param_init = f1.read().splitlines()
            WorkDir=param_init[0]
            default_graphics, default_fits, default_clahe_only, default_crop_square, default_transversalium, default_transversalium_strength, default_rotation = param_init[1] == 'True', param_init[2] == 'True', param_init[3] == 'True' , param_init[4] == 'True', param_init[5] == 'True', float(param_init[6]), int(param_init[7])
    except:
        print('note: error reading .ini file - using default parameters')
        WorkDir=''
        default_graphics, default_fits, default_clahe_only, default_crop_square, default_transversalium, default_transversalium_strength, default_rotation = False, False, False, False, True, 3, 0
    return WorkDir, default_graphics, default_fits, default_clahe_only, default_crop_square, default_transversalium, default_transversalium_strength, default_rotation

def interpret_UI_values(ui_values):
    try:
        shift = ui_values['-DX-']
        shift_choice = shift.split(':')
        if len(shift_choice) == 1:
            options['shift'] = list(map(int, [x.strip() for x in shift.split(',')]))
        elif len(shift_choice) == 2:
            options['shift'] = list(range(int(shift_choice[0].strip()), int(shift_choice[1].strip())+1))
        elif len(shift_choice) == 3:
            options['shift'] = list(range(int(shift_choice[0].strip()), int(shift_choice[1].strip())+1, int(shift_choice[2].strip())))
        else:
            raise Exception('invalid offset input!')
        if len(options['shift']) == 0:
            raise Exception('Error: pixel offset input lower bound greater than upper bound!')
    except ValueError : 
        raise Exception('invalid pixel offset value!')        
    options['flag_display'] = ui_values['-DISP-']
    try : 
        options['ratio_fixe'] = float(ui_values['-RATIO-']) if ui_values['-RATIO-'] else None
    except ValueError : 
        raise Exception('invalid Y/X ratio value')
    try : 
        options['slant_fix'] = float(ui_values['-SLANT-']) if ui_values['-SLANT-'] else None
    except ValueError : 
        raise Exception('invalid tilt angle value!')
    try:
        options['delta_radius'] = int(ui_values['-delta_radius-'])
    except ValueError:
        raise Exception('invalid protus_radius_adjustment')
    options['save_fit'] = ui_values['-FIT-']
    options['clahe_only'] = ui_values['-CLAHE_ONLY-']
    options['crop_width_square'] = ui_values['-crop_width_square-']
    options['transversalium'] = ui_values['-transversalium-']
    options['trans_strength'] = int(ui_values['-trans_strength-']*100) + 1
    options['flip_x'] = ui_values['-flip_x-']
    options['img_rotate'] = int(ui_values['img_rotate'])
    serfiles=ui_values['-FILE-'].split(';')
    try:
        for serfile in serfiles:
            f=open(serfile, "rb")
            f.close()
    except:
        raise Exception('ERROR opening file :'+serfile+'!')
    return serfiles, options

# get and return options and serfiles from user using GUI
def inputUI():
    WorkDir, default_graphics, default_fits, default_clahe_only, default_crop_square, default_transversalium, default_transversalium_strength, default_rotation = read_ini()
    serfiles, options = UI_SerBrowse(WorkDir, default_graphics, default_fits, default_clahe_only, default_crop_square, default_transversalium, default_transversalium_strength, default_rotation) #TODO as options is defined as global, only serfiles could be returned
    return options, serfiles, WorkDir



"""
-------------------------------------------------------------------------------------------
le programme commence ici !
--------------------------------------------------------------------------------------------
"""
serfiles = []
WorkDir = ''

options = {    
'shift':[0],
'flag_display':False,
'ratio_fixe' : None,
'slant_fix' : None ,
'save_fit' : False,
'clahe_only' : True,
'disk_display' : True, #protus
'delta_radius' : 0,
'crop_width_square' : False,
'transversalium' : True,
'trans_strength': 301,
'img_rotate': 0,
'flip_x': False,
}

flag_dictionnary = {
    'd' : 'flag_display', #True False display all pictures
    'c' : 'clahe_only',  #True/False
    'f' : 'save_fit', #True/False
    'p' : 'disk_display', #True/False protuberances 
    'w' : 'shift',
    's' : 'crop_width_square', # True / False
    't' : 'transversalium', # True / False
    'm' : 'flip_x' # True / False
    }

# list of files to process
## add a command line argument.
if len(sys.argv)>1 : 
    for argument in sys.argv[1:]:
        if '-' == argument[0]: #it's flag options
            treat_flag_at_cli(argument)
        else : #it's a file or some files
            if argument.split('.')[-1].upper()=='SER' or argument.split('.')[-1].upper()=='AVI': 
                serfiles.append(argument)
    print('theses files are going to be processed : ', serfiles)

def do_work():
    global options, WorkDir
    if len(serfiles)==1:
        options['tempo']=60000 #4000 #pour gerer la tempo des affichages des images resultats dans cv2.waitKey
    else:
        options['tempo']=5000
        
    # boucle sur la liste des fichers
    for serfile in serfiles:
        if serfile=='':
            sys.exit()
        print('file %s is processing'%serfile)
        WorkDir=os.path.dirname(serfile)+"/"
        os.chdir(WorkDir)
        base=os.path.basename(serfile)
        basefich=os.path.splitext(base)[0]
        if base=='':
            print('filename ERROR : ',serfile)
            sys.exit()

        # ouverture du fichier ser
        try:
            f=open(serfile, "rb")
            f.close()
        except:
            print('ERROR opening file : ',serfile)
            sys.exit()

        # save parameters to .ini file
        try:
            mydir_ini=os.path.dirname(sys.argv[0])+'/SHG.ini'
            with open(mydir_ini, "w") as f1:
                f1.write('\n'.join([WorkDir, str(options['flag_display']), str(options['save_fit']), str(options['clahe_only']), str(options['crop_width_square']), str(options['transversalium']), str(options['trans_strength']/100), str(options['img_rotate'])]))
        except:
            traceback.print_exc()
            print('ERROR: couldnt write file ' + mydir_ini)    
        
        # appel au module d'extraction, reconstruction et correction
        #
        # basefich: nom du fichier ser sans extension et sans repertoire
        # dx: decalage en pixel par rapport au centre de la raie

        try : 
            sol.solex_proc(serfile,options.copy()) 
        except:
            print('ERROR ENCOUNTERED')
            traceback.print_exc()
            cv2.destroyAllWindows()

if 0:
    options, serfiles = inputUI()
    cProfile.run('do_work()', sort='cumtime')
else:
    # if no command line arguments, open GUI interface
    if len(serfiles)==0:
        while True:
            options, serfiles, WorkDir = inputUI()
            do_work()
    else:
        do_work() # use inputs from CLI
