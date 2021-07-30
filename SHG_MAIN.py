# -*- coding: utf-8 -*-
"""
Version 24 July 2021
@author: valerie desnoux
with improvements by Andrew Smith
Front end de traitements spectro helio de fichier ser
- interface pour selectionner un ou plusieurs fichiers
- appel au module solex_recon qui traite la sequence et genere les fichiers fits
- propose avec openCV un affichage de l'image resultat ou pas
- decalage en longueur d'onde avec Shift
- ajout d'une zone pour entrer un ratio fixe. Si reste à zero alors il sera calculé
automatiquement
- ajout de sauvegarde png _protus avec flag disk_display en dur
"""
import math
import numpy as np

import os
import sys
import Solex_recon as sol
from astropy.io import fits
import cProfile
import PySimpleGUI as sg
import tkinter as tk
import ctypes # Modification Jean-Francois: for reading the monitor size
import cv2


def usage():
    return "SHG_MAIN.py [-dcfrp] [file(s) to treat]"

def treat_flag_at_cli(arguments):
    global options
    #reading arguments
    for caracter in argument[1:]: #remove '-'
        try : 
            options[flag_dictionnary[caracter]]=True if flag_dictionnary.get(caracter) else False
        except KeyError : 
            print('ERROR !!! At least one argument is not accepted')
            print(usage())
    print('options %s'%(options))

def UI_SerBrowse (WorkDir):
    """
    Parameters
    ----------
    WorkDir : TYPE string
        repertoire par defaut à l'ouverture de la boite de dialogue

    Returns 
    -------
    Filenames : TYPE string
        liste des fichiers selectionnés, avec leur extension et le chemin complet
    Shift : Type string
        Ecart en pixel demandé pour reconstruire le disque 
        sur une longeur d'onde en relatif par rapport au centre de la raie  
    ratio_fixe : ratio Y/X en fixe, si egal à zéro alors calcul automatique
    flag_isplay: affiche ou non la construction du disque en temps réel
    """
    sg.theme('Dark2')
    sg.theme_button_color(('white', '#500000'))
    
    layout = [
    [sg.Text('SER file name(s)', size=(15, 1)), sg.InputText(default_text='',size=(75,1),key='-FILE-'),
     sg.FilesBrowse('Open',file_types=(("SER Files", "*.ser"),),initial_folder=WorkDir)],
    [sg.Checkbox('Show graphics', default=False, key='-DISP-')],
    [sg.Checkbox('Save .fits files', default=False, key='-FIT-')],
    [sg.Checkbox('Save CLAHE.png only', default=False, key='-CLAHE_ONLY-')],
    [sg.Text('Y/X ratio (blank for auto)', size=(20,1)), sg.Input(default_text='', size=(8,1),key='-RATIO-')],
    [sg.Text('Tilt angle (blank for auto)',size=(20,1)),sg.Input(default_text='',size=(8,1),key='-SLANT-',enable_events=True)],
    [sg.Text('Pixel offset',size=(20,1)),sg.Input(default_text='0',size=(8,1),key='-DX-',enable_events=True)],
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
            break

    window.close()
               
    FileNames=values['-FILE-']
    
    
    return FileNames, values['-DX-'], values['-DISP-'], 0 if values['-RATIO-']=='' else values['-RATIO-'] , 0 if values['-SLANT-']=='' else values['-SLANT-'], values['-FIT-'], values['-CLAHE_ONLY-']

"""
-------------------------------------------------------------------------------------------
le programme commence ici !
--------------------------------------------------------------------------------------------
"""
disk_display=True
serfiles = []

options = {    
'shift':0,
'flag_display':False,
'ratio_fixe' : 0,
'slant_fix' : 0 ,
'save_fit' : True,
'clahe_only' : False,
'disk_display' : True #protus
}

flag_dictionnary = {
    'd' : 'flag_display', #True False display all pictures
    'c' : 'clahe_only',  #True/False
    'f' : 'save_fit', #True/False
    'p' : 'disk_display' #True/False protuberances 
    }

# list of files to process
## add a command line argument.
if len(sys.argv)>1 : 
    for argument in sys.argv : 
        if '-' in argument: #it's flag options
            treat_flag_at_cli(argument)
        else : #it's a file or some files
            if argument.split('.')[-1].upper()=='SER' : 
                serfiles.append(argument)
    print('theses files are going to be processed : ', serfiles)
print('Processing will begin with values : \n shift %s, flag_display %s, ratio_fixe "%s", slant_fix "%s", save_fit %s, clahe_only %s, disk_display %s' %(options['shift'], options['flag_display'], options['ratio_fixe'], options['slant_fix'], options['save_fit'], options['clahe_only'], options['disk_display']) )

# check for .ini file for working directory           
try:
    mydir_ini=os.path.dirname(sys.argv[0])+'/SHG.ini'
    with open(mydir_ini, "r") as f1:   
        param_init = f1.readlines()
        WorkDir=param_init[0]
except:
    WorkDir=''
    
# if no command line arguments, open GUI interface
if len(serfiles)==0 : 
    serfiles, shift, flag_display, ratio_fixe, slant_fix, save_fit, clahe_only =UI_SerBrowse(WorkDir) #TODO as options is defined as global, only serfiles could be returned
    try : 
        options['shift'] = int(shift)
    except ValueError : 
        print('invalid shift value')
        sys.exit()
        
    options['flag_display'] = flag_display
    try : 
        options['ratio_fixe'] = float(ratio_fixe)
    except ValueError : 
        print('invalid ratio_fixe value')
        sys.exit()
    try : 
        options['slant_fix'] = float(slant_fix)
    except ValueError : 
        print('invalid slant_fix value')
        sys.exit()
    options['save_fit'] = save_fit
    options['clahe_only'] = clahe_only
    
    serfiles=serfiles.split(';')

#pour gerer la tempo des affichages des images resultats dans cv2.waitKey
#si plusieurs fichiers à traiter

def do_work():
    global options
    if len(serfiles)==1:
        options['tempo']=60000 #4000
    else:
        options['tempo']=1000
        
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
            print('erreur nom de fichier : ',serfile)
            sys.exit()

        # ouverture du fichier ser
        try:
            f=open(serfile, "rb")
            f.close()
        except:
            print('erreur ouverture fichier : ',serfile)
            sys.exit()

        # save working directory
        try:
            with open(mydir_ini, "w") as f1:
                f1.writelines(WorkDir)
        except:
            print('ERROR: couldnt write file ' + mydir_ini)    
        
        # appel au module d'extraction, reconstruction et correction
        #
        # basefich: nom du fichier ser sans extension et sans repertoire
        # dx: decalage en pixel par rapport au centre de la raie

        try : 
            frame, header, cercle=sol.solex_proc(serfile,options)
        
            print('circle = ' , cercle)

            base=os.path.basename(serfile)
            basefich=os.path.splitext(base)[0]
            
            flag_result_show = options['flag_display']
            
            # create a CLAHE object (Arguments are optional)
            # clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(5,5))
            clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(2,2))
            cl1 = clahe.apply(frame)
            
            # image leger seuils
            frame1=np.copy(frame)
            Seuil_bas=np.percentile(frame, 25)
            Seuil_haut=np.percentile(frame,99.9999)
            frame1[frame1>Seuil_haut]=65000
            print('Seuil bas       :', np.floor(Seuil_bas))
            print('Seuil haut      :', np.floor(Seuil_haut))
            fc=(frame1-Seuil_bas)* (65000/(Seuil_haut-Seuil_bas))
            fc[fc<0]=0
            frame_contrasted=np.array(fc, dtype='uint16')
            
            # image seuils serres 
            frame1=np.copy(frame)
            Seuil_bas=(Seuil_haut*0.25)
            Seuil_haut=np.percentile(frame1,99.9999)
            print('Seuil bas HC    :', np.floor(Seuil_bas))
            print('Seuil haut HC   :', np.floor(Seuil_haut))
            frame1[frame1>Seuil_haut]=65000
            fc2=(frame1-Seuil_bas)* (65000/(Seuil_haut-Seuil_bas))
            fc2[fc2<0]=0
            frame_contrasted2=np.array(fc2, dtype='uint16')
            
            # image seuils protus
            frame1=np.copy(frame)
            Seuil_bas=0
            Seuil_haut=np.percentile(frame1,99.9999)*0.18        
            print('Seuil bas protu :', np.floor(Seuil_bas))
            print('Seuil haut protu:', np.floor(Seuil_haut))
            frame1[frame1>Seuil_haut]=Seuil_haut
            fc2=(frame1-Seuil_bas)* (65000/(Seuil_haut-Seuil_bas))
            fc2[fc2<0]=0
            frame_contrasted3=np.array(fc2, dtype='uint16')
            if not cercle == (-1, -1, -1) and disk_display==True:
                x0=int(cercle[0])
                y0=int(cercle[1])
                r=int(cercle[2]) - 4
                frame_contrasted3=cv2.circle(frame_contrasted3, (x0,y0),r,80,-1)
            
            Seuil_bas=np.percentile(cl1, 25)
            Seuil_haut=np.percentile(cl1,99.9999)*1.05
            cc=(cl1-Seuil_bas)*(65000/(Seuil_haut-Seuil_bas))
            cc[cc<0]=0
            cc=np.array(cc, dtype='uint16')

            # sauvegarde en png de clahe
            cv2.imwrite(basefich+'_clahe.png',cc)   # Modification Jean-Francois: placed before the IF for clear reading
            if not options['clahe_only']:
                # sauvegarde en png pour appliquer une colormap par autre script
                cv2.imwrite(basefich+'_disk.png',frame_contrasted)
                # sauvegarde en png pour appliquer une colormap par autre script
                cv2.imwrite(basefich+'_diskHC.png',frame_contrasted2)
                # sauvegarde en png pour appliquer une colormap par autre script
                cv2.imwrite(basefich+'_protus.png',frame_contrasted3)
            
            # Modification Jean-Francois: the 4 images are concatenated together in 1 image => 'Sun images'
            # The 'Sun images' is scaled for the monitor maximal dimension ... it is scaled to match the dimension of the monitor without 
            # changing the Y/X scale of the images 
            if flag_result_show:
                im_1 = cv2.hconcat([frame_contrasted, frame_contrasted2])
                im_2 = cv2.hconcat([frame_contrasted3, cc])
                im_3 = cv2.vconcat([im_1, im_2])
                screen = tk.Tk()
                screensize = screen.winfo_screenwidth(), screen.winfo_screenheight()   
                scale = min(screensize[0] / im_3.shape[1], screensize[1] / im_3.shape[0])
                cv2.namedWindow('Sun images', cv2.WINDOW_NORMAL)
                cv2.moveWindow('Sun images', 0, 0)
                cv2.resizeWindow('Sun images',int(im_3.shape[1] * scale), int(im_3.shape[0] * scale))
                cv2.imshow('Sun images',im_3)
                cv2.waitKey(options['tempo'])  # affiche et continue
            
            """
            #create colormap
            im = cv2.imread(basefich+'_disk.png')
            im_max=(np.amax(im))*1.3
            im[im>im_max]=200
            print ('im_max : ',im_max)
            scale=255/im_max
            imnp=np.array(im*scale, dtype='uint8')
            imC = cv2.applyColorMap(imnp, cv2.COLORMAP_HOT)
            iw=int(imC.shape[1]*sc)
            ih=int(imC.shape[0]*sc)
            cv2.resize(imC,dsize=(ih,iw))
            cv2.namedWindow('color', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('color', iw, ih)
            cv2.moveWindow('color',int(newiw*sc), 0)
            cv2.imshow('color',imC)
            cv2.waitKey(5000)
            cv2.imwrite(basefich+'_color.png',imC)
            """

            frame2=np.copy(frame)
            frame2=np.array(cl1, dtype='uint16')
            # sauvegarde le fits
            if options['save_fit']:
                DiskHDU=fits.PrimaryHDU(frame2,header)
                DiskHDU.writeto(basefich+'_clahe.fits', overwrite='True')
        except :
            print('treatment collapse')
            cv2.destroyAllWindows()

if 0:        
    cProfile.run('do_work()', sort='cumtime')
else:
    do_work()
