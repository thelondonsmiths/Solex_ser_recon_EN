import math
import PySimpleGUI as sg


def interpret_UI_values(options, ui_values):
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
        options['fixed_width'] = int(ui_values['fixed_width']) if ui_values['fixed_width'] else None
    except ValueError : 
        raise Exception('invalid fixed width value!')
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
        return serfiles
    except:
        raise Exception('ERROR opening file :'+serfile+'!')

def inputUI(options):
    sg.theme('Dark2')
    sg.theme_button_color(('white', '#500000'))
    
    layout = [
    [sg.Text('File(s)', size=(5, 1)), sg.InputText(default_text=options['workDir'],size=(75,1),key='-FILE-'),
     sg.FilesBrowse('Open',file_types=(("SER Files", "*.ser"),("AVI Files", "*.avi"),),initial_folder=options['workDir'])],
    [sg.Checkbox('Show graphics', default=options['flag_display'], key='-DISP-')],
    [sg.Checkbox('Save fits files', default=options['save_fit'], key='-FIT-')],
    [sg.Checkbox('Save clahe.png only', default=options['clahe_only'], key='-CLAHE_ONLY-')],
    [sg.Checkbox('Crop square', default=options['crop_width_square'], key='-crop_width_square-')],
    [sg.Text('Fixed image width (blank for none)', size=(25,1)), sg.Input(default_text=options['fixed_width'], size=(8,1),key='fixed_width')],
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
                    serfiles = interpret_UI_values(options, values)
                    window.close()
                    return serfiles
                except Exception as inst:
                    sg.Popup('Error: ' + inst.args[0], keep_on_top=True)
                    
            else:
                # display pop-up file not entered
                sg.Popup('Error: file not entered! Please enter file(s)', keep_on_top=True)
        window.Element('-trans_strength-').Update(visible = values['-transversalium-'])
        window.Element('text_trans').Update(visible = values['-transversalium-'])    
