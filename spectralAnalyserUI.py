# -*- coding: utf-8 -*-
"""
@author: Andrew Smith
Version 6 August 2023

------------------------------------------------------------------------
Spectral Analyser
-------------------------------------------------------------------------

"""
import math
import PySimpleGUI as sg
import sys
import json
import os
import traceback
from tkinter import *
from random import randint
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.backends.backend_tkagg as tkagg
import tkinter as Tk
import numpy as np
from PIL import Image, ImageTk
from solex_util import compute_mean_return_fit
from io import BytesIO
import cv2

from solex_util import *
from video_reader import *
from ellipse_to_circle import ellipse_to_circle, correct_image
from Solex_recon import single_image_process

def tuple_downscale(x, f):
    return tuple([int(_*f) for _ in x])

def get_spectrum(sample):
    mean = np.mean(sample, axis = 0)
    return mean[mean.shape[0]//2, :]

def select(data, lo, hi):
    #indx = np.where(lo < data[:, 0] < hi)
    #print(indx)
    
    #return data[min(indx): max(indx)+1, :]
    indx = np.logical_and(lo <= data[:, 0], data[:, 0] < hi)
    v = np.where(indx==1)[0]
    return data[min(v):max(v)+1, :]

def load_lines(path):
    l, names = [], []
    with open(resource_path(path), encoding='utf-8') as f:
        for line in f:
            v = line.split(' ')
            l.append(float(v[0]))
            names.append(v[1])
    names_num = [names[i] + '('+str(l[i])+')' for i in range(len(names))]
    return l, names, names_num

def analyseSpectrum(options, file, lang_dict):
    npzfile = np.load(resource_path('language_data/alps.npz'))
    line_data = np.vstack((np.arange(npzfile['first'], npzfile['last'], npzfile['step']), npzfile['y']/255)).T
    
    anchor_cands, anchor_cand_names, anchors = load_lines('line_data/anchor_candidates.txt')
    target_nums, target_names, targets = load_lines('line_data/line_targets.txt')
    
    options_orig = options
    options = options.copy() # deep copy
    options['clahe_only'] = True
    options['save_fit'] = False
    options['flag_display'] = False
    options['_nolog'] = True
    options['shift'] = [0]
    options['basefich0'] = ''
    fig = Figure()
    fig.set_tight_layout(True)
    ((ax1, ax3), (ax2, ax4)) = fig.subplots(2, 2)
    ax2_twin = ax2.twinx()
    fig_sz = fig.get_size_inches()
    fig.set_size_inches(fig_sz[0]*1.8, fig_sz[1]*2) 
    ax1.set_xlabel("X axis")
    ax1.set_ylabel("Y axis")
    ax1.grid()

    layout_file_input = [
        [sg.Text('File', size=(7, 1), key = 'File'), sg.InputText(default_text=options['specDir'],size=(75,1),key='-FILE2-'),
         sg.FilesBrowse('Choose file', key = 'Choose file', file_types=(("Video Files (AVI or SER)", "*.ser *.avi"),),initial_folder=options['specDir'], enable_events=True),
         sg.Button("Start analysis", key='Start analysis', enable_events=True), sg.Button('Save image'), sg.Button('Exit')]
    ]


    c1 = sg.Combo(anchors, readonly=True, key='-anchor-', enable_events=True)
    c2 = sg.Combo(targets, readonly=True, key='-target-', enable_events=True)
    s1 = sg.Spin(list(range(-999, 1000)), initial_value=0, readonly=False, size=4, enable_events=True, key='-shift-')

    in1 = sg.InputText('', key='-ashift-', size=(10, 1))
    in2 = sg.InputText(options['dispersion'], key='-dispersion-', size=(10, 1))
    
    
    layout = [
          
        [sg.T('Anchor line'), c1, sg.T('GOTO line'), c2, sg.T("GOTO wavelength (Å)"), in1, sg.T('Pixel shift', key='shift:'), s1, sg.T("Wavelength shift: None", key="Ångstrom Shift:")],
        [sg.T('Dispersion (Å/pixel)'), in2, sg.B('Auto dispersion')],
        [sg.Canvas(size=(1000, 800), key='canvas')],
    ]

    window = sg.Window('Pixel Offset Live', layout_file_input+layout, finalize=True, resizable=True, keep_on_top=False)
    # needed to access the canvas element prior to reading the window
    window['-shift-'].bind("<Return>", "_Enter")
    window['-ashift-'].bind("<Return>", "_Enter")
    window['-dispersion-'].bind("<Return>", "_Enter")
    canvas_elem = window['canvas']

    graph = FigureCanvasTkAgg(fig, master=canvas_elem.TKCanvas)
    canvas = canvas_elem.TKCanvas

    original_ratio = options['ratio_fixe']
    original_slant = options['slant_fix']
    
    mean, fit, y1, y2 = None, None, None, None
    all_rdr = None
    borders = [0,0,0,0]
    cercle0 = (-1, -1, -1)
    backup_bounds = (-1, -1)
    hdr = None
    spectrum = None
    spectrum2 = None
    downscale_f = 0.33
    disk_memo = None
    file = None
    refresh_anchor = False
    dispersion = None
    while True:
        event, values = window.read(timeout=20)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            window.close()
            if values is None:
                return None
            return values['-shift-']
     
        display_refresh = False
        if event == 'Choose file' or event == "Start analysis":
            options['ratio_fixe'] = original_ratio
            options['slant_fix'] = original_slant
            window['-shift-'].update(0)
            window["Ångstrom Shift:"].update("Wavelength shift: None")
            options['shift'] = [0]
            display_refresh = True
            try:
                file = values['-FILE2-'].split(';')[0]
                window['-FILE2-'].update(file)
                options['specDir'] = os.path.dirname(file)
                options_orig['specDir'] = os.path.dirname(file) # this is to feed back into SHG config
                all_rdr = all_video_reader(file)
                
                ih = all_rdr.ih
                iw = all_rdr.iw
                hdr = make_header(all_rdr)
                mean, fit, y1, y2 = compute_mean_return_fit(all_rdr, options, hdr, iw, ih, '')

                target_height = max(1000, ih / 3)
                downscale_f = target_height / ih
                
                brightest = np.argmax(all_rdr.means)
                spectrum = get_spectrum(all_rdr.frames[max(0, brightest - 5) : min(all_rdr.FrameCount - 1, brightest + 5), :, :])
                spectrum2 = mean[mean.shape[0]//2, :] 
                backup_bounds = (int(y1), int(y2))
                if options['ratio_fixe'] is None and options['slant_fix'] is None:
                    options['shift'] = [10]
                    all_rdr.reset()
                    disklist,_,_,_ = read_video_improved(all_rdr, fit, options)
                    if options['flip_x']:
                        disklist[0] = np.flip(disklist[0], axis = 1)
                    frame_circularized, cercle0, options['ratio_fixe'], phi, borders = ellipse_to_circle(disklist[0], options, '')
                    options['slant_fix'] = math.degrees(phi)
                options['shift'] = [0] # back to zero now    
            

            except Exception as inst:
                traceback.print_exc()
                sg.Popup('Error: ' + inst.args[0], keep_on_top=True)
                mean = None

        if event == '-ashift-_Enter':
            if mean is None or dispersion is None or values['-anchor-']=='':
                sg.Popup("Not ready to GOTO wavelength. First load and a file and press start analysis and choose anchor line!", keep_on_top=True)
            else:
                try:
                    gotolambda = float(values['-ashift-'])
                    j = anchors.index(values['-anchor-'])
                    anchor_guess = anchor_cands[j]
                    shift = int((gotolambda - anchor_guess)/dispersion)
                    if 0 <= shift + fit[len(fit)//2][0]+fit[len(fit)//2][1] < spectrum2.shape[0]:          
                        options['shift'] = [shift]
                        window['-shift-'].update(shift)
                        display_refresh = True
                    else:
                        sg.Popup("That line does not appear to be in image!", keep_on_top=True)
                except:
                    sg.Popup("invalid wavelength", keep_on_top=True)
                
    
        if event == '-shift-' or event == '-shift-_Enter':
            try:
                x = int(values['-shift-'])
                display_refresh = True
                options['shift'] = [x]
            except Exception as inst:
                sg.Popup('Error: invalid shift: ' + values['-shift-'], keep_on_top=True)


        if event == 'Auto dispersion':
            if not mean is None and values['-anchor-']:
                display_refresh = True
                anchor_refresh = True
            else:
                sg.Popup('First load and a file and press start analysis and choose anchor line!', keep_on_top=True)

        if event == '-dispersion-_Enter' or event == '-target-' or event == '-anchor-':
            try:
                dispersion = float(values['-dispersion-'])
                if dispersion <= 0:
                    raise Exception("dispersion must be positive")
                options['dispersion'] = round(dispersion, 6)
                options_orig['dispersion'] = round(dispersion, 6)
                if values['-anchor-']:
                    display_refresh = True
                else:
                    sg.Popup("Choose an anchor first!", keep_on_top=True)
                
            except:
                sg.Popup('Invalid dispersion', keep_on_top=True)

        if event == '-target-':
            if dispersion is None or values['-anchor-']=='' or mean is None:
                sg.Popup("Not ready to do that yet: load file and find dispersion first!", keep_on_top=True)
            else:
                j = anchors.index(values['-anchor-'])
                anchor_guess = anchor_cands[j]
                i = targets.index(values['-target-'])
                shift = int((target_nums[i] - anchor_guess)/dispersion)
                if 0 <= shift + fit[len(fit)//2][0]+fit[len(fit)//2][1] < spectrum2.shape[0]:          
                    options['shift'] = [shift]
                    window['-shift-'].update(shift)
                    display_refresh = True
                else:
                    sg.Popup("That line does not appear to be in image!", keep_on_top=True)
            
        if display_refresh:   
            ax1.cla()
            ax2.cla()
            ax2_twin.cla()
            ax2.grid()
            ax3.cla()   
            ax3.axis('off')
            ax4.axis('off')

            if not mean is None:

                # fit anchor:
                if anchor_refresh:
                    if values['-anchor-']:
                        anchor_x = fit[len(fit)//2][0]+fit[len(fit)//2][1]
                        scale_guesses = np.linspace(0.03, 0.12, spectrum2.shape[0]*2)
                        #scale_guesses = [0.057]
                        i = anchors.index(values['-anchor-'])
                        anchor_guess = anchor_cands[i]
                        corr_values = []
                        for scale in scale_guesses:
                            line_data_scaled_x = (line_data[:, 0] - anchor_guess)/scale + anchor_x

                            copy_line_data = np.copy(line_data)
                            copy_line_data[:, 0] = line_data_scaled_x

                            select_line_data = select(copy_line_data, 0, spectrum2.shape[0])
                            interp_line_data = np.interp(np.arange(spectrum2.shape[0]), select_line_data[:, 0], select_line_data[:, 1])

                            exc_width = 5
                            interp_line_data[max(0, int(anchor_x) - exc_width): min(int(anchor_x) + exc_width, interp_line_data.shape[0] - 1)] = np.mean(interp_line_data)
                            lspec = np.log(spectrum2)
                            lspec[max(0, int(anchor_x) - exc_width): min(int(anchor_x) + exc_width, lspec.shape[0] - 1)] = np.mean(lspec)
                            corr_value = np.corrcoef(interp_line_data, lspec)[0, 1]
                            corr_values.append(corr_value)
                        max_ind = np.argmax(corr_values)
                        dispersion = scale_guesses[max_ind]
                        print(f"the dispersion is:{dispersion}")
                        window['-dispersion-'].update(f'{dispersion:.6f}')
                        options['dispersion'] = round(dispersion, 6)
                        options_orig['dispersion'] = round(dispersion, 6)
                    else:
                        dispersion = None
                
                if dispersion is None:
                    ax2.plot(np.log(spectrum2), color='green', label='data')
                    ax2.set_xlim((0, spectrum.shape[0]-1))
                    ax2.axvline(x=fit[len(fit)//2][0]+fit[len(fit)//2][1]+options['shift'][0], color='red', linestyle='--')
                    ax2.axvline(x=fit[len(fit)//2][0]+fit[len(fit)//2][1], color='blue')
                    ax2.legend()
                else:
                    # update plot
                    i = anchors.index(values['-anchor-'])
                    anchor_val = anchor_cands[i]
                    anchor_px = fit[len(fit)//2][0]+fit[len(fit)//2][1]
                    hi_clip = (spectrum2.shape - anchor_px) * dispersion + anchor_val
                    low_clip = (-anchor_px) * dispersion + anchor_val
                    
                    
                    select_data = select(line_data, low_clip, hi_clip)
                    
                    ln1 = ax2_twin.plot(select_data[:, 0], select_data[:, 1], color = 'purple', label = 'reference')
                    ln2 = ax2.plot((np.arange(spectrum2.shape[0]) - anchor_px) * dispersion + anchor_val, np.log(spectrum2), color='green', label='data')
                    ax2.set_xlabel(f'wavelength(Å); dispersion: {dispersion:.4f} Å/pixel')
                    ax2.set_xlim((low_clip, hi_clip))
                    ax2_twin.set_xlim((low_clip, hi_clip))
                    ax2.axvline(x=anchor_val, color='blue')
                    myline = anchor_val+options['shift'][0]*dispersion
                    ax2.axvline(x=myline, color='red', linestyle='--')
                    lns = ln1+ln2
                    labs = [l.get_label() for l in lns]
                    ax2.legend(lns, labs) # shared legend for it and twin
                    # update Angstrom shift
                    window['-ashift-'].update(f'{myline:.3f}')
                    ashift = options['shift'][0]*dispersion
                    window["Ångstrom Shift:"].update(f"Wavelength shift: {options['shift'][0]*dispersion:.3f}Å")


                
                    
                ax1.imshow(mean, cmap='gray', aspect='auto')
                ax1.plot([x[0]+x[1]+options['shift'] for x in fit], range(ih), 'r--')
                ax1.plot([x[0]+x[1] for x in fit], range(ih), 'b')
                ax1.set_xlim((0, mean.shape[1]-1))
                         
                all_rdr.reset()
                disklist,_,_,_ = read_video_improved(all_rdr, fit, options)
                if options['flip_x']:
                    disklist[0] = np.flip(disklist[0], axis = 1)
                disk_memo = disklist[0]
                # process
                ratio = options['ratio_fixe'] if not options['ratio_fixe'] is None else 1.0
                phi = math.radians(options['slant_fix']) if not options['slant_fix'] is None else 0.0
                frame_circularized = correct_image(downscale(disklist[0], downscale_f) / 65536, phi, ratio, np.array([-1.0, -1.0]), -1.0, options, print_log=False)[0]  # Note that we assume 16-bit
                clahe, protus = single_image_process(frame_circularized, hdr, options, tuple_downscale(cercle0, downscale_f) if not cercle0 == (-1, -1, -1) else (-1, -1, -1), tuple_downscale(borders, downscale_f), '', tuple_downscale(backup_bounds, downscale_f))
          
                ax3.imshow(clahe, cmap='gray', aspect='equal')
                ax4.imshow(protus, cmap='gray', aspect='equal')

                graph.draw()
                
                figure_x, figure_y, figure_w, figure_h = fig.bbox.bounds
                figure_w, figure_h = int(figure_w), int(figure_h)
                photo = Tk.PhotoImage(master=canvas, width=figure_w, height=figure_h)

                canvas.create_image(figure_w, figure_h, image=photo)

            graph.get_tk_widget().pack(side='top', fill='both', expand=1)
            window.refresh()
        if event == 'Save image':
            if not mean is None:                    
                ratio = options['ratio_fixe'] if not options['ratio_fixe'] is None else 1.0
                phi = math.radians(options['slant_fix']) if not options['slant_fix'] is None else 0.0
                frame_circularized = correct_image(disk_memo / 65536, phi, ratio, np.array([-1.0, -1.0]), -1.0, options, print_log=False)[0]  # Note that we assume 16-bit
                clahe, protus = single_image_process(frame_circularized, hdr, options, cercle0, borders, '', backup_bounds)
                compression = 0
                basename = os.path.splitext(file)[0] + '_shift='+str(options['shift'][0])
                cv2.imwrite(output_path(basename+'_clahe.png', options), clahe, [cv2.IMWRITE_PNG_COMPRESSION, compression])   # Modification Jean-Francois: placed before the IF for clear reading
                cv2.imwrite(output_path(basename+'_protus.png', options), protus, [cv2.IMWRITE_PNG_COMPRESSION, compression])
        anchor_refresh = False
        display_refresh = False
