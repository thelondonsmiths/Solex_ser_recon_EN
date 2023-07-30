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

from solex_util import *
from video_reader import *
from ellipse_to_circle import ellipse_to_circle, correct_image
from Solex_recon import single_image_process


def get_spectrum(sample):
    mean = np.mean(sample, axis = 0)
    return mean[mean.shape[0]//2, :]

def analyseSpectrum(options, file, lang_dict):
    options = options.copy() # deep copy
    options['clahe_only'] = True
    options['save_fit'] = False
    options['flag_display'] = False
    options['_nolog'] = True
    options['shift'] = [0]
    options['basefich0'] = ''
    fig = Figure()

    (ax1, ax2, ax3) = fig.subplots(3, 1, gridspec_kw={'height_ratios': [1, 1, 3.5]})
    fig_sz = fig.get_size_inches()
    fig.set_size_inches(fig_sz[0]*1.8, fig_sz[1]*2) 
    ax1.set_xlabel("X axis")
    ax1.set_ylabel("Y axis")
    ax1.grid()

    layout_file_input = [
        [sg.Text('File', size=(7, 1), key = 'File'), sg.InputText(default_text=options['workDir'],size=(75,1),key='-FILE2-'),
         sg.FilesBrowse('Choose file', key = 'Choose file', file_types=(("Video Files (AVI or SER)", "*.ser *.avi"),),initial_folder=options['workDir'], enable_events=True),
         sg.Button("Start analysis", key='Start analysis', enable_events=True), sg.Button('Exit')]
    ]

    layout = [
          [sg.T('shift:', key='shift:'), sg.Spin(list(range(-999, 1000)), initial_value=0, readonly=False, size=4, enable_events=True, key='-shift-')],
          [sg.Canvas(size=(1000, 800), key='canvas')],
    ]

    window = sg.Window('Demo Application - Embedding Matplotlib In PySimpleGUI', layout_file_input+layout, finalize=True, resizable=True, keep_on_top=True)
    # needed to access the canvas element prior to reading the window

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
    downscale_f = 0.33
    
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
            options['shift'] = [0]
            display_refresh = True
            try:
                file = values['-FILE2-'].split(';')[0]
                window['-FILE2-'].update(file)
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
                backup_bounds = (int(y1*downscale_f), int(y2*downscale_f))
                if options['ratio_fixe'] is None and options['slant_fix'] is None:
                    options['shift'] = [10]
                    all_rdr.reset()
                    disklist,_,_,_ = read_video_improved(all_rdr, fit, options)
                    frame_circularized, cercle0, options['ratio_fixe'], phi, borders = ellipse_to_circle(downscale(disklist[0], downscale_f), options, '')
                    options['slant_fix'] = math.degrees(phi)
                options['shift'] = [0] # back to zero now    
            

            except Exception as inst:
                traceback.print_exc()
                sg.Popup('Error: ' + inst.args[0], keep_on_top=True)
                mean = None

        if event == '-shift-':
            display_refresh = True
            options['shift'] = [values['-shift-']]
            
        if display_refresh:   
            ax1.cla()
            ax2.cla()
            ax3.cla()
            ax2.grid()
            ax3.axis('off')

            if not mean is None:
                ax2.plot(np.log(spectrum), color='purple')
                ax2.plot(np.log(spectrum2), color='green')
                ax2.set_xlim((0, spectrum.shape[0]-1))
                ax2.axvline(x=fit[len(fit)//2][0]+fit[len(fit)//2][1]+options['shift'], color='red', linestyle='--')
                ax2.axvline(x=fit[len(fit)//2][0]+fit[len(fit)//2][1], color='blue')
                ax1.imshow(mean, cmap='gray', aspect='auto')
                ax1.plot([x[0]+x[1]+options['shift'] for x in fit], range(ih), 'r--')
                ax1.plot([x[0]+x[1] for x in fit], range(ih), 'b')
                ax1.set_xlim((0, mean.shape[1]-1))

                all_rdr.reset()
                disklist,_,_,_ = read_video_improved(all_rdr, fit, options)
                
                # process
                ratio = options['ratio_fixe'] if not options['ratio_fixe'] is None else 1.0
                phi = math.radians(options['slant_fix']) if not options['slant_fix'] is None else 0.0
                frame_circularized = correct_image(downscale(disklist[0], downscale_f) / 65536, phi, ratio, np.array([-1.0, -1.0]), -1.0, options, print_log=False)[0]  # Note that we assume 16-bit
                clahe, protus = single_image_process(frame_circularized, hdr, options, cercle0, borders, '', backup_bounds)
        

                
                
                ax3.imshow(np.concatenate((clahe, protus), axis = 1), cmap='gray', aspect='equal')

                '''
                # update the figure canvas
                plt.subplots_adjust(0,0,1,1,0,0)
                for ax in fig.axes:
                    ax.axis('off')
                    ax.margins(0,0)
                    ax.xaxis.set_major_locator(plt.NullLocator())
                    ax.yaxis.set_major_locator(plt.NullLocator())
                '''
                #plt.tight_layout(pad=0)
                graph.draw()
                
                figure_x, figure_y, figure_w, figure_h = fig.bbox.bounds
                figure_w, figure_h = int(figure_w), int(figure_h)
                photo = Tk.PhotoImage(master=canvas, width=figure_w, height=figure_h)

                canvas.create_image(1000, 800, image=photo)

            graph.get_tk_widget().pack(side='top', fill='both', expand=1)
            window.refresh()


    '''
    dpts = [randint(0, 10) for x in range(10000)]
    # Our event loop      
    for i in range(len(dpts)):
        event, values = window.read(timeout=20)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            window.close()
            return

        ax.cla()
        ax.grid()

        ax.plot(range(20), dpts[i:i + 20], color='purple')
        graph.draw()
        figure_x, figure_y, figure_w, figure_h = fig.bbox.bounds
        figure_w, figure_h = int(figure_w), int(figure_h)
        photo = Tk.PhotoImage(master=canvas, width=figure_w, height=figure_h)

        canvas.create_image(640 / 2, 480 / 2, image=photo)

        graph.get_tk_widget().pack(side='top', fill='both', expand=1)
        #tkagg.FigureCanvasTkAgg.blit(photo, figure_canvas_agg.get_renderer()._renderer)
        window.refresh()
    '''
