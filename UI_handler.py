"""
@author: Andrew Smith
based on code by Valerie Desnoux
contributors: Jean-Francois Pittet, Jean-Baptiste Butet, Pascal Berteau, Matt Considine
Version 24 July 2023
"""

import math
import PySimpleGUI as sg
import sys
import json
import os
import traceback
from PIL import Image, ImageTk
import io

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def interpret_UI_values(options, ui_values):
    try:
        shift = ui_values['_pixel_offset']
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
    options['flag_display'] = ui_values['Show graphics']
    try : 
        options['ratio_fixe'] = float(ui_values['_y/x_ratio']) if ui_values['_y/x_ratio'] else None
    except ValueError : 
        raise Exception('invalid Y/X ratio value')
    try : 
        options['slant_fix'] = float(ui_values['_tilt']) if ui_values['_tilt'] else None
    except ValueError : 
        raise Exception('invalid tilt angle value!')
    try : 
        options['fixed_width'] = int(ui_values['_fixed_width']) if ui_values['_fixed_width'] else None
    except ValueError : 
        raise Exception('invalid fixed width value!')
    try:
        options['delta_radius'] = int(ui_values['_protus_adjustment'])
        options['disk_display'] = True
    except ValueError:
        raise Exception('invalid protus_radius_adjustment')
    options['save_fit'] = ui_values['Save fits files']
    options['clahe_only'] = ui_values['Save clahe.png only']
    options['crop_width_square'] = ui_values['Crop square']
    options['transversalium'] = ui_values['Correct transversalium lines']
    options['trans_strength'] = int(ui_values['-trans_strength-']*100) + 1
    options['flip_x'] = ui_values['Mirror X']
    options['img_rotate'] = int(ui_values['img_rotate'])
    serfiles=ui_values['-FILE-'].split(';')
    options['output_dir'] = ui_values['output_dir']
    if options['selected_mode'] == 'Folder input mode':
        options['input_dir'] = ui_values['input_dir']
    options['continuous_detect_mode'] = ui_values['Continuous detect mode']
    if options['selected_mode'] == 'File input mode':
        try:
            for serfile in serfiles:
                f=open(serfile, "rb")
                f.close()
            return serfiles
        except:
            traceback.print_exc()
            raise Exception('ERROR opening file :'+serfile+'!')
    elif options['selected_mode'] == 'Folder input mode':
        if not os.path.isdir(options['input_dir']):
            raise Exception('ERROR opening folder :'+options['input_dir'])
        return []
    else:
        raise Exception('ERROR: Invalid mode selection: ' + options['selected_mode'])
        

def read_langs():
    prefixed = sorted([filename for filename in os.listdir(resource_path('language_data')) if filename.startswith('dict_lang') and filename.endswith('.txt')])
    langs = []
    lang_dicts = []
    for file in prefixed:
#        print('loading lang: ', file)
        try:
            with open(resource_path(os.path.join('language_data', file)), encoding="utf-8") as fp:
                lang_dict = json.load(fp)
        except Exception:
            traceback.print_exc()
            print(f'note: error reading lang file {file}')
            continue
        langs.append(lang_dict['_lang_name'])
        lang_dicts.append(lang_dict)
    #print(langs)
    #for x in lang_dicts:
    #    print(x)
    return langs, lang_dicts

# ------------------------------------------------------------------------------
# use PIL to read data of one image
# ------------------------------------------------------------------------------


def get_img_data(f, maxsize=(30, 18), first=False):
    """Generate image data using PIL
    """
    try:
        img = Image.open(f)
        img.thumbnail(maxsize)
        if first:                     # tkinter is inactive the first time
            bio = io.BytesIO()
            img.save(bio, format="PNG")
            del img
            return bio.getvalue()
        return ImageTk.PhotoImage(img)
    except Exception:
        traceback.print_exc()
        print(f'note: error reading flag thumbnail file {f}')
        return None

def change_langs(window, popup_messages, lang_dict, flag_change=True):
    flag_ok = 0
    checkboxes = set(['Show graphics', 'Save fits files', 'Save clahe.png only', 'Crop square', 'Mirror X', 'Correct transversalium lines', 'Continuous detect mode'])
    popup_ids = set(['no_file_error', 'no_folder_error'])
    for k, v in lang_dict.items():
        if k == '_flag_icon':
            if flag_change:
                window['_flag_icon'].update(data=get_img_data(resource_path(os.path.join('language_data', v))))
            flag_ok = 1
        elif k in popup_ids:
            popup_messages[k] = v
        elif k == 'pixel_offset_tooltip':
            window['_pixel_offset'].TooltipObject.text = v
        elif k == 'protus_adjustment_tooltip':
            window['_protus_adjustment'].TooltipObject.text = v
        elif k == '_lang_name':
            pass
        else:
            try:
                if k in window.AllKeysDict:
                    if k in checkboxes:
                        window[k].update(text=v)
                    else:
                        window[k].update(v)
                else:
                    print(f'ERROR: language error for setting text (no such key!): "{k}" : "{v}"')
            except Exception:
                traceback.print_exc()
                print(f'ERROR: language error for setting text: "{k}" : "{v}"')
    if not flag_ok:
        window['_flag_icon'].update(data=None)
    

def inputUI(options):
    langs, lang_dicts = read_langs()
    popup_messages = {"no_file_error": "Error: file not entered! Please enter file(s)", "no_folder_error": "Error: folder not entered! Please enter folder"}
    image_elem = sg.Image(data=get_img_data(resource_path(os.path.join('language_data', 'flagEN.png')), first=True), key = '_flag_icon')
        
    sg.theme('Dark2')
    sg.theme_button_color(('white', '#500000'))

    layout_title = [
        [sg.Text('Solar disk reconstruction from SHG video files', font='Any 14', key='Solar disk reconstruction from SHG video files'), sg.Push(), image_elem, sg.Combo(langs, key="lang_input", enable_events=True, default_value='English', size=(10, 12), readonly=True)], # TODO: save default in options
    ]

    layout_file_input = [
        [sg.Text('File(s)', size=(7, 1), key = 'File(s)'), sg.InputText(default_text=options['workDir'],size=(75,1),key='-FILE-'),
         sg.FilesBrowse('Choose file(s)', key = 'Choose file(s)', file_types=(("Video Files (AVI or SER)", "*.ser *.avi"),),initial_folder=options['workDir'])],
    ]

    layout_folder_input = [
        [sg.Text('Folder', size=(7, 1), key = 'Folder'), sg.InputText(default_text='',size=(75,1),key='input_dir'),
         sg.FolderBrowse('Choose input folder', key = 'Choose input folder', initial_folder=options['input_dir'])],
        [sg.Checkbox('Continuous detect mode', default=options['continuous_detect_mode'], key='Continuous detect mode')],
    ]

    layout_folder_output = [
        [sg.Text('Output folder (blank for same as input):', size=(50, 1), key = 'Output Folder (blank for same as input):')],
        [sg.InputText(default_text=options['output_dir'],size=(75,1),key='output_dir'),
            sg.FolderBrowse('Choose output folder', key = 'Choose output folder',initial_folder=options['output_dir'])],
    ]

    layout_base = [
    
    [sg.Checkbox('Show graphics', default=options['flag_display'], key='Show graphics')],
    [sg.Checkbox('Save fits files', default=options['save_fit'], key='Save fits files')],
    [sg.Checkbox('Save clahe.png only', default=options['clahe_only'], key='Save clahe.png only')],
    [sg.Checkbox('Crop square', default=options['crop_width_square'], key='Crop square')],
    [sg.Text('Fixed image width (blank for none)', size=(32,1), key='Fixed image width (blank for none)'), sg.Input(default_text=options['fixed_width'], size=(8,1),key='_fixed_width')],
    [sg.Checkbox('Mirror X', default=False, key='Mirror X')],
    [sg.Text("Rotate png images:", key='Rotate png images:')],
    [sg.Slider(range=(0,270),
         default_value=options['img_rotate'],
         resolution=90,     
         size=(30,15),
         orientation='horizontal',
         font=('Helvetica', 12),
         key='img_rotate')],
    [sg.Checkbox('Correct transversalium lines', default=options['transversalium'], key='Correct transversalium lines', enable_events=True)],
    [sg.Text("Transversalium correction strength (pixels x 100) :", key='Transversalium correction strength (pixels x 100) :', visible=options['transversalium'])],
    [sg.Slider(range=(0.25,7),
         default_value=options['trans_strength']/100,
         resolution=0.25,     
         size=(30,15),
         orientation='horizontal',
         font=('Helvetica', 12),
         key='-trans_strength-',
         visible=options['transversalium'])],
    [sg.Text('Y/X ratio (blank for auto)', key='Y/X ratio (blank for auto)', size=(32,1)), sg.Input(default_text='', key = '_y/x_ratio', size=(8,1))],
    [sg.Text('Tilt angle (blank for auto)',size=(32,1), key='Tilt angle (blank for auto)'), sg.Input(default_text='',size=(8,1),key='_tilt',enable_events=True)],
    [sg.Text('Pixel offset',size=(32,1), key='Pixel offset'),sg.Input(default_text='0',size=(8,1),tooltip= "a,b,c will produce images at a, b and c\n x:y:w will produce images starting at x, finishing at y, every w pixels",key='_pixel_offset',enable_events=True)],
    [sg.Text('Protus adjustment', size=(32,1), key='Protus adjustment'), sg.Input(default_text=str(options['delta_radius']), size=(8,1), tooltip = 'make the black circle bigger or smaller by inputting an integer', key='_protus_adjustment')],
    [sg.Button('OK'), sg.Cancel(), sg.Push(), sg.Button("Open output folder", key='Open output folder', enable_events=True)]
    ] 

    tab_group = sg.TabGroup([[sg.Tab('File input mode', layout_file_input, tooltip='', key='File input mode'), sg.Tab('Folder input mode', layout_folder_input, key='Folder input mode')]])
    layout = [
        layout_title + [[tab_group]] + layout_folder_output + layout_base    
    ]  
    
    window = sg.Window('SHG Version 4.1', layout, finalize=True)
    window.BringToFront()

    if options['language'] in langs:
        window['lang_input'].update(options['language'])
        lang_dict = lang_dicts[langs.index(options['language'])]
        change_langs(window, popup_messages, lang_dicts[langs.index('English')], flag_change=False)
        change_langs(window, popup_messages, lang_dict)
    else:
        print(f'ERROR: language not available: "{options["language"]}"')

    while True:
        event, values = window.read()
        if event==sg.WIN_CLOSED or event=='Cancel':
            window.close()
            sys.exit()
        if event == 'lang_input':
            lang_dict = lang_dicts[langs.index(values['lang_input'])]
            change_langs(window, popup_messages, lang_dicts[langs.index('English')], flag_change=False) # if missing will be English
            change_langs(window, popup_messages, lang_dict)
            options['language'] = values['lang_input']

        if event=='Open output folder':
            x = values['output_dir'].strip()
            if not x:
                selected_mode = tab_group.Get()
                if selected_mode == 'File input mode':
                    x = options['workDir']
                elif selected_mode == 'Folder input mode':
                    x = options['input_dir']
                else:
                    sg.Popup(popup_messages['ERROR: mode selection error: ' + selected_mode], keep_on_top=True)
            if x and os.path.isdir(x):
                path = os.startfile(os.path.realpath(x))
            else:
                sg.Popup(popup_messages['no_folder_error'], keep_on_top=True)
        if event=='OK':
            selected_mode = tab_group.Get()
            if selected_mode == 'File input mode':
                if not values['-FILE-'] == options['workDir'] and not values['-FILE-'] == '':
                    input_okay_flag = True
                else:
                    # display pop-up file not entered
                    input_okay_flag = False
                    sg.Popup(popup_messages['no_file_error'], keep_on_top=True)
            elif selected_mode == 'Folder input mode':
                if not values['input_dir'] ==  '':
                    input_okay_flag = True
                else:
                    # display pop-up folder not entered
                    input_okay_flag = False
                    sg.Popup(popup_messages['no_folder_error'], keep_on_top=True)
            else:
                sg.Popup(popup_messages['ERROR: mode selection error: ' + selected_mode], keep_on_top=True)
            if input_okay_flag:
                options['selected_mode'] = selected_mode
                try:
                    serfiles = interpret_UI_values(options, values)
                    window.close()
                    return serfiles
                except Exception as inst:
                    traceback.print_exc()
                    sg.Popup('Error: ' + inst.args[0], keep_on_top=True)
                    
        
        window.Element('-trans_strength-').Update(visible = values['Correct transversalium lines'])
        window.Element('Transversalium correction strength (pixels x 100) :').Update(visible = values['Correct transversalium lines'])    
