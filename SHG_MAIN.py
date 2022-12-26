# -*- coding: utf-8 -*-
"""
@author: Valerie Desnoux
with improvements by Andrew Smith
contributors: Jean-Francois Pittet, Jean-Baptiste Butet, Pascal Berteau, Matt Considine
Version 6 November 2022

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
import json
try :
    from PIL import Image, ImageMath
    pil_module=True
except ModuleNotFoundError:
    print("No module named PIL found. If you want generate GIF picture, you will need it. \n https://pillow.readthedocs.io/en/stable/installation.html")
    pil_module=False


serfiles = []

options = {
    'shift':[0],
    'flag_display':False,
    'ratio_fixe' : None,
    'slant_fix' : None ,
    'save_fit' : False,
    'clahe_only' : False,
    'disk_display' : True, #protus
    'delta_radius' : 0,
    'crop_width_square' : False,
    'transversalium' : True,
    'trans_strength': 301,
    'img_rotate': 0,
    'flip_x': False,
    'workDir': '',
    'fixed_width': None,
    'poly_fit': None,
    'doppler': None,
    'doppler_picture': 0,
    'gif': 0,

}

flag_dictionnary = { #add True/False flag here. Managed near line 147
    'd' : 'flag_display', #True False display all pictures
    'c' : 'clahe_only',  #True/False
    'f' : 'save_fit', #True/False
    'p' : 'disk_display', #True/False protuberances
    'w' : 'shift',
    's' : 'crop_width_square', # True / False
    't' : 'transversalium', # True / False
    'm' : 'flip_x' # True / False

}



def usage():
    usage_ = "SHG_MAIN.py [-dcfpstwmpf] [file(s) to treat, * allowed]\n"
    usage_ += "'d' : 'flag_display', display all graphics (False by default)\n"
    usage_ += "'c' : 'clahe_only',  only final clahe image is saved (False by default)\n"
    usage_ += "'f' : 'save_fit', save all fits files (False by default)\n"
    usage_ += "'m' : 'mirror flip', mirror flip in x-direction (False by default)\n"
    usage_ += "'p' : 'no disk_display' turn off black disk with protuberance images (False by default)\n"
    usage_ += "'s' : 'crop_square_width', crop the width to equal the height (False by default)\n"
    usage_ += "'t' : 'disable transversalium', disable transversalium correction (False by default)\n"
    usage_ += "'w' : 'int a,int b,int c'  produce images at a, b and c pixels from minima\n"
    usage_ += "'w' : 'int x:int y:int w'  produce images starting at x, finishing at y, every w pixels from minima\n"
    #usage_ += "'P' : 'a,b,c'  using polynome a*x²+b*x+c or a*x³+b*x²+c*x+d as fitting\n"
    usage_ += "'D' : int 'n'      produce 4 pictures, from -n pixels, n pixel from minimum and a mean of 2 and a dopplergram. CAREFUL ! INCOMPATIBLE with 'w' or 'g' option\n"
    usage_ += "'r' : int 'w'  crop width to a constant no. of pixels."
    usage_ += "'g' : int 'n' Make a animated GIF from -n pixels to n pixels aside absorption ray. Crop to 2000 pixels width."
    return usage_

def treat_flag_at_cli(arguments):
    """read cli arguments and produce options variable"""
    #reading arguments
    i=0
    while i < len(argument[1:]): #there's a '-' at first)
        character = argument[1:][i]
        if character=='h': #asking help menu
            print(usage())
            sys.exit()
        elif character=='P':
            #find characters for shifting
            poly = ''
            stop = False
            try :
                while not stop:
                    if argument[1:][i+1].isdigit() or argument[1:][i+1]==',' or argument[1:][i+1]=='.' or argument[1:][i+1]=='e' or argument[1:][i+1]=='+' or argument[1:][i+1]=='-':
                        poly += argument[1:][i+1]
                        i += 1
                    else :
                        i += 1
                        stop = True
            except IndexError:
                i += 1 #the reach the end of arguments.
            poly_choices = poly.split(',')
            try:
                if len(poly_choices)==3 :
                    a, b, c = poly_choices
                elif len(poly_choices)==4 :
                    a, b, c, d = poly_choices
            except ValueError:
                print('ERROR : invalid polynome fitting input : ', shift_choice)
                print('USAGE : python3 SHG_MAIN.py -P1.45881927e+02,-2.16219665e-01,9.45250257e-05 files')
                sys.exit()
            options['poly_fit'] = [float(a),float(b),float(c)]

        elif character == 'w':
            #find characters for shifting
            shift = ''
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
                print('ERROR : invalid shift input')
                print(usage())
                sys.exit()
        elif character=='t':
            options['transversalium'] = False
            i+=1
        elif character=='p':
            options['disk_display'] = False
            i+=1
        elif character=='r':
            fw = ''
            try:
                while argument[1:][i+1].isdigit():
                    fw += argument[1:][i+1]
                    i += 1
                i += 1
            except IndexError:
                i+=1 #the reach the end of arguments.
            options['fixed_width'] = int(fw)
        elif character=='g':
            if not pil_module :
                print('ERROR : Generating doppler picture need PIL module. Please install it')
                sys.exit()
            gif = ''
            try:
                while argument[1:][i+1].isdigit():
                    gif += argument[1:][i+1]
                    i += 1
                i += 1
            except IndexError:
                i+=1 #the reach the end of arguments.

            try :
                options['gif'] = int(gif)
                options['shift'] = [i for i in range(-options['gif'], options['gif']+1)]
                options['crop_width_square']=True
                options['fixedd_width']=2000
            except ValueError :
                print('ERROR : Generating doppler picture need one integer')
                print(usage())
                sys.exit()

            i+=1
        elif character=='D':
            #find characters for shifting
            decal = ''
            stop = False
            try :
                while not stop :
                    if argument[1:][i+1].isdigit():
                        decal+=argument[1:][i+1]
                        i+=1
                    else :
                        i+=1
                        stop=True

            except IndexError :
                i+=1 #the reach the end of arguments
                try :
                    options['doppler_picture'] = int(decal)
                    options['shift'] = [-int(decal), 0, int(decal)]
                except ValueError :
                    print('ERROR : Generating doppler picture need one integer')
                    print(usage())
                    sys.exit()

            except:
                print('ERROR : Generating doppler picture need one integer')
                print(usage())
                sys.exit()
        else :
            try : #all others
                options[flag_dictionnary[character]]=True if flag_dictionnary.get(character) else False
                i+=1
            except KeyError:
                print('ERROR !!! At least one argument is not accepted')
                print(usage())
                i+=1
    if options['doppler'] and options['poly_fit'] is None:
        print('ERROR !!! g option need a polynome provided by P option.')
        print('USAGE : python3 SHG_MAIN.py -gP1.45881927e+02,-2.16219665e-01,9.45250257e-05 files')
        print(usage())
        sys.exit()
    print('options :  %s' % (options))

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
    try :
        options['fixed_width'] = int(ui_values['-fixed_width-']) if ui_values['-fixed_width-'] else None
    except ValueError :
        raise Exception('invalid fixed width value!')
    try:
        options['delta_radius'] = int(ui_values['-delta_radius-'])
    except ValueError:
        raise Exception('invalid protus_radius_adjustment')
    options['save_fit'] = ui_values['-FIT-']
    options['clahe_only'] = ui_values['-CLAHE_ONLY-']
    options['crop_width_square'] = ui_values['-crop_width_square-']
    options['doppler_picture'] = int(ui_values['-dopplergram-'])
    if options['doppler_picture'] > 0:
        decal = options['doppler_picture']
        options['shift'] = [-int(decal), 0, int(decal)]
    options['gif'] = int(ui_values['-gif-'])
    if options['gif'] > 0:
        options['shift'] = [i for i in range(-options['gif'], options['gif']+1)]
        options['crop_width_square']=True
        options['fixedd_width']=2000

    options['transversalium'] = ui_values['-transversalium-']
    options['trans_strength'] = int(ui_values['-trans_strength-']*100) + 1
    options['flip_x'] = ui_values['-flip_x-']
    options['img_rotate'] = int(ui_values['img_rotate'])
    global serfiles
    serfiles=ui_values['-FILE-'].split(';')
    try:
        for serfile in serfiles:
            f=open(serfile, "rb")
            f.close()
    except:
        raise Exception('ERROR opening file :'+serfile+'!')

def inputUI():
    sg.theme('Dark2')
    sg.theme_button_color(('white', '#500000'))
    
    layout = [
    [sg.Text('File(s)', size=(5, 1)), sg.InputText(default_text=options['workDir'],size=(75,1),key='-FILE-'),
     sg.FilesBrowse('Open',file_types=(("SER Files", "*.ser"),("AVI Files", "*.avi"),),initial_folder=options['workDir'])],
    [sg.Checkbox('Show graphics', default=options['flag_display'], key='-DISP-')],
    [sg.Checkbox('Save fits files', default=options['save_fit'], key='-FIT-')],
    [sg.Checkbox('Save clahe.png only', default=options['clahe_only'], key='-CLAHE_ONLY-')],
    [sg.Checkbox('Crop square', default=options['crop_width_square'], key='-crop_width_square-')],
    [sg.Text('Fixed image width (blank for none)', size=(35,1)), sg.Input(default_text=options['fixed_width'], size=(8,1),key='-fixed_width-')],
    [sg.Checkbox('Mirror X', default=False, key='-flip_x-')],
    [sg.Text("Rotate png images:", key='img_rotate_slider')],
    [sg.Slider(range=(0,270),
         default_value=options['img_rotate'],
         resolution=90,     
         size=(25,15),
         orientation='horizontal',
         font=('Helvetica', 12),
         key='img_rotate')],
    [sg.Checkbox('Correct transversalium lines', default=options['transversalium'], key='-transversalium-', enable_events=True)],
    [sg.Text("Transversalium correction strength (pixels x 100) :", key='text_trans', visible=options['transversalium'])],
    [sg.Slider(range=(0.25,7),
         default_value=options['trans_strength']/100,
         resolution=0.25,     
         size=(25,15),
         orientation='horizontal',
         font=('Helvetica', 12),
         key='-trans_strength-',
         visible=options['transversalium'])],
    [sg.Text('Y/X ratio (blank for auto)', size=(25,1)), sg.Input(default_text='', size=(8,1),key='-RATIO-')],
    [sg.Text('Tilt angle (blank for auto)',size=(25,1)),sg.Input(default_text='',size=(8,1),key='-SLANT-',enable_events=True)],
    [sg.Text('Pixel offset',size=(25,1)),sg.Input(default_text='0',size=(8,1),tooltip= "a,b,c will produce images at a, b and c\n x:y:w will produce images starting at x, finishing at y, every w pixels",key='-DX-',enable_events=True)],
    [sg.Text('Protus adjustment', size=(25,1)), sg.Input(default_text=str(options['delta_radius']), size=(8,1), tooltip = 'make the black circle bigger or smaller by inputting an integer', key='-delta_radius-')],

    [sg.Text('Dopplergram with shift \n(0 for none): ', size=(25,2)), sg.Input(default_text=0, size=(8,1),key='-dopplergram-')],
    [sg.Text('Generate a GIF from -n to n pixel aside ray center (0 for none): ', size=(25,2)), sg.Input(default_text=0, size=(8,1),key='-gif-')],
    [sg.Button('OK'), sg.Cancel()]
    ] 
    
    window = sg.Window('Processing', layout, finalize=True)
    window.BringToFront()
    
    
    while True:
        event, values = window.read()
        if event==sg.WIN_CLOSED or event=='Cancel':
            window.close()
            sys.exit()
        
        if event=='OK':
            if not values['-FILE-'] == options['workDir'] and not values['-FILE-'] == '':
                try:
                    interpret_UI_values(values)
                    window.close()
                    return
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
    print('loading config file...')

    try:
        mydir_ini=os.path.join(os.path.dirname(sys.argv[0]),'SHG_config.txt')
        with open(mydir_ini, 'r') as fp:
            global options
            options_ = json.load(fp)
            options.update(options)
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

def do_work(serfiles, options, cli = False):
    print('Processing')
    if len(serfiles)==1:
        options['tempo']=60000 #4000 #pour gerer la tempo des affichages des images resultats dans cv2.waitKey
    else:
        options['tempo']=5000
        
    # boucle sur la liste des fichers
    for serfile in serfiles:
        if serfile=='':
            print("ERROR filename empty")
            return
        print('file %s is processing'%serfile)
        options['workDir'] = os.path.dirname(serfile)
        try :
            os.chdir(options['workDir'])
        except :
            os.chdir('.')
        base = os.path.basename(serfile)
        basefich = os.path.splitext(base)[0]
        if base == '':
            print('filename ERROR : ',serfile)
            return

        # ouverture du fichier ser
        try:
            f=open(serfile, "rb")
            f.close()
        except:
            print('ERROR opening file : ',serfile)
            return

        # save parameters to .ini file
        write_ini()
        try : 
            sol.solex_proc(serfile,options.copy()) 
        except:
            print('ERROR ENCOUNTERED')
            traceback.print_exc()
            cv2.destroyAllWindows()

"""
-------------------------------------------------------------------------------------------
le programme commence ici !
--------------------------------------------------------------------------------------------
"""
if __name__ == '__main__':
    # check for CLI input

    if len(sys.argv)>1:
        for argument in sys.argv[1:]:
            if '-' == argument[0]: #it's flag options
                treat_flag_at_cli(argument)
            else : #it's a file or some files
                if argument.split('.')[-1].upper()=='SER' or argument.split('.')[-1].upper()=='AVI': 
                    dirname = os.path.dirname(os.path.abspath(argument))
                    serfiles.append(os.path.join(dirname,argument))
        print('theses files are going to be processed : ', serfiles)

    if 0: #test code for performance test
        inputUI()
        cProfile.run('do_work(serfiles, options)', sort='cumtime')
    else:
        # if no command line arguments, open GUI interface
        if len(serfiles)==0:
            # read initial parameters from .ini file
            read_ini()
            while True:
                inputUI()
                do_work(serfiles, options)
        else:
            do_work(serfiles, options) # use inputs from CLI


