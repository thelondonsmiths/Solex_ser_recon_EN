"""
@author: Andrew Smith
Version 18 July 2023
"""
from solex_util import rescale_brightness
import math
import FreeSimpleGUI as sg
import sys
import json
import os
import traceback
from PIL import Image, ImageTk
import io
import cv2
import numpy as np

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def interpret_UI_values(options, ui_values):
    options['hi'] = int(ui_values['hi'])
    options['lo'] = int(ui_values['lo'])
    options['tile_size'] = int(ui_values['tile_size'])
    options['sat'] = int(ui_values['sat'])
    options['do_stretch'] = ui_values['Use high/low stretch']
    files=ui_values['-FILE-'].split(';')
    try:
        for file in files:
            f=open(file, "rb")
            f.close()
        return files
    except:
        raise Exception('ERROR opening file :'+file+'!')

def read_langs():
    prefixed = sorted([filename for filename in os.listdir(resource_path('language_data')) if filename.startswith('dict_lang') and filename.endswith('.txt')])
    langs = []
    lang_dicts = []
    for file in prefixed:
        print('loading lang: ', file)
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
        img = Image.open(resource_path(f))
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
    checkboxes = set(['Use high/low stretch'])
    for k, v in lang_dict.items():
        if k == '_flag_icon':
            if flag_change:
                window['_flag_icon'].update(data=get_img_data(os.path.join('language_data', v)))
            flag_ok = 1
        elif k == 'no_file_error':
            popup_messages['no_file_error'] = v
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
                    pass
                    #print(f'ERROR: language error for setting text (no such key!): "{k}" : "{v}"')
            except Exception:
                traceback.print_exc()
                print(f'ERROR: language error for setting text: "{k}" : "{v}"')
    if not flag_ok:
        window['_flag_icon'].update(data=None)


def inputUI(options):
    langs, lang_dicts = read_langs()
    popup_messages = {"no_file_error": "Error: file not entered! Please enter file(s)", "hi_less_than_lo_error": "Error: the low percentile must be less than the high percentile!"}
    image_elem = sg.Image(data=get_img_data(os.path.join('language_data', 'flagEN.png'), first=True), key = '_flag_icon')

    sg.theme('Dark2')
    sg.theme_button_color(('white', '#500000'))

    layout = [
        [sg.Text('Apply CLAHE to images', font='Any 14', key='Apply CLAHE'), sg.Push(), image_elem, sg.Combo(langs, key="lang_input", enable_events=True, default_value='English', size=(10, 12), readonly=True)], # TODO: save default in options
        [sg.Text('File(s)', size=(7, 1), key = 'File(s)'), sg.InputText(default_text=options['workDir'],size=(75,1),key='-FILE-'),
         sg.FilesBrowse('Open', key = 'Open', file_types=(("Image files", "*.png *.tif"),),initial_folder=options['workDir'])],
        [sg.Text("Tile size:", key='Tile size')],
        [sg.Slider(range=(1,4),
             default_value=options['tile_size'],
             resolution=1,
             size=(30,15),
             orientation='horizontal',
             font=('Helvetica', 12),
             key='tile_size')],
        [sg.Checkbox('Use high/low stretch', default=options['do_stretch'], key='Use high/low stretch', enable_events=True)],
        [sg.Text("Low threshhold:", key='Low threshhold', visible=options['do_stretch'])],
        [sg.Slider(range=(0,100),
             default_value=options['lo'],
             resolution=1,
             size=(30,15),
             orientation='horizontal',
             font=('Helvetica', 12),
             key='lo',
             visible=options['do_stretch'])],
        [sg.Text("High threshhold:", key='High threshhold', visible=options['do_stretch'])],
        [sg.Slider(range=(0,100),
             default_value=options['hi'],
             resolution=1,
             size=(30,15),
             orientation='horizontal',
             font=('Helvetica', 12),
             key='hi',
             visible=options['do_stretch'])],
        [sg.Text("Saturation percentage:", key='Saturation percentage', visible=options['do_stretch'])],
        [sg.Slider(range=(50,100),
             default_value=options['sat'],
             resolution=1,
             size=(30,15),
             orientation='horizontal',
             font=('Helvetica', 12),
             key='sat',
             visible=options['do_stretch'])],
        [sg.Button('OK'), sg.Cancel()]
    ]

    window = sg.Window('CLAHE_Apply v1.0', layout, finalize=True)
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
        if event=='OK':

            if not values['-FILE-'] == options['workDir'] and not values['-FILE-'] == '':
                try:
                    print('before:', options)
                    files = interpret_UI_values(options, values)
                    print('after:', options)
                    options['workDir'] = os.path.dirname(files[0])+"/"
                    if options['hi'] <= options['lo']:
                        sg.Popup(popup_messages['hi_less_than_lo_error'], keep_on_top=True)
                    else:
                        window.close()
                        return files
                except Exception as inst:
                    traceback.print_exc()
                    sg.Popup('Error: ' + inst.args[0], keep_on_top=True)

            else:
                # display pop-up file not entered
                sg.Popup(popup_messages['no_file_error'], keep_on_top=True)
        if event == 'do_stretch':
            window.Element('lo').Update(visible = values['do_stretch'])
            window.Element('hi').Update(visible = values['do_stretch'])
            window.Element('Low threshhold').Update(visible = values['do_stretch'])
            window.Element('High threshhold').Update(visible = values['do_stretch'])
            window.Element('Saturation percentage').Update(visible = values['do_stretch'])
            window.Element('sat').Update(visible = values['do_stretch'])


'''
open config.txt and read parameters
return parameters from file, or default if file not found or invalid
'''
def read_ini():
    # check for config.txt file for working directory
    print('loading config file...')

    try:
        mydir_ini=os.path.join(os.path.dirname(sys.argv[0]),'clahe_config.txt')
        with open(mydir_ini, 'r', encoding="utf-8") as fp:
            global options
            options.update(json.load(fp)) # if config has missing entries keep default
    except Exception:
        traceback.print_exc()
        print('note: error reading config file - using default parameters')


def write_ini():
    try:
        print('saving config file ...')
        mydir_ini = os.path.join(os.path.dirname(sys.argv[0]),'clahe_config.txt')
        with open(mydir_ini, 'w', encoding="utf-8") as fp:
            json.dump(options, fp, sort_keys=True, indent=4)
    except Exception:
        traceback.print_exc()
        print('ERROR: failed to write config file: ' + mydir_ini)


def apply_clahe(file, options, write_file=True):
    frame = cv2.imread(file, cv2.IMREAD_ANYDEPTH)
    if len(frame.shape) > 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # make sure gray not color
    clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(options['tile_size'],options['tile_size']))
    cl1 = clahe.apply(frame)
    dark = np.percentile(frame, options['lo'])
    bright = np.percentile(frame, options['hi'])
    if options['do_stretch']:
        cl1 = rescale_brightness(cl1, dark, bright, alpha=options['sat']/100)
    if write_file:
        print('save:', os.path.splitext(file)[0]+'_clahe.png')
        cv2.imwrite(os.path.splitext(file)[0]+'_clahe.png',cl1)
    return cl1

options = {'workDir':'', 'language':'English', 'lo':0, 'hi':100, 'do_stretch':False, 'sat':80, 'tile_size':2}

if __name__ == '__main__':
    while True:
        read_ini()
        files = inputUI(options)
        write_ini()

        for file in files:
            apply_clahe(file, options)

