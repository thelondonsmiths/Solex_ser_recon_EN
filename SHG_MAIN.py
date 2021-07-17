# -*- coding: utf-8 -*-
"""
Version 14 july 2021


@author: valerie desnoux
with improvements by Andrew Smith


Front end de traitements spectro helio de fichier ser
- interface pour selectionner un ou plusieurs fichiers
- appel au module solex_recon qui traite la sequence et genere les fichiers fits
- propose avec openCV un affichage de l'image resultat ou pas
- decalage en longueur d'onde avec Shift
- ajout d'une zone pour entrer un ratio fixe. Si resteà zero alors il sera calculé
automatiquement
- ajout de sauvegarde png _protus avec flag disk_display en dur

"""
import math
import numpy as np
import cv2
import os
import sys
import Solex_recon as sol
from astropy.io import fits
import cProfile
#import time
import PySimpleGUI as sg

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
    [sg.Text('SER file name(s)', size=(20, 1)), sg.InputText(default_text='',size=(65,1),key='-FILE-'),
     sg.FilesBrowse('Open',file_types=(("SER Files", "*.ser"),),initial_folder=WorkDir)],
    [sg.Checkbox('Show images', default=False, key='-DISP-')],
    [sg.Checkbox('Save .fit files', default=False, key='-FIT-')],
    [sg.Checkbox('Save CLAHE image only', default=False, key='-CLAHE_ONLY-')],
    [sg.Text('Y/X ratio (blank for auto)', size=(20,1)), sg.Input(default_text='', size=(8,1),key='-RATIO-')],
    [sg.Text('Slant angle (blank for auto)',size=(20,1)),sg.Input(default_text='',size=(8,1),key='-SLANT-',enable_events=True)],
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
    shift=values['-DX-']
    flag_display=values['-DISP-']
    if values['-RATIO-'] == '':
        ratio_fixe = 0
    else:
        ratio_fixe=float(values['-RATIO-'])
    
    return FileNames, shift, flag_display, ratio_fixe, values['-SLANT-'], values['-FIT-'], values['-CLAHE_ONLY-']

"""
-------------------------------------------------------------------------------------------
le programme commence ici !

Si version windows
# recupere les parametres utilisateurs enregistrés lors de la session
# precedente dans un fichier txt "pysolex.ini" qui va etre placé ici en dur
# dans repertoire c:/py/ pour l'instant

Si version mac
#recupere les noms de fichiers par un input console
#valeurs de flag_display, Shift et ratio_fixe sont en dur dans le programme

--------------------------------------------------------------------------------------------
"""
version_mac =False
disk_display=False

if not(version_mac):
    try:
        with open('c:/py/pysolex.ini', "r") as f1:
        
            param_init = f1.readlines()
            WorkDir=param_init[0]
    except:
        WorkDir=''

    # Recupere paramatres de la boite de dialogue
    serfiles, shift, flag_display, ratio_fixe, slant_fix, save_fit, clahe_only =UI_SerBrowse(WorkDir)
    serfiles=serfiles.split(';')
    
else:
    WorkDir='/Users/macbuil/ocuments/pyser/'
    serfiles=[]
    print('nom du fichier sans extension, ou des fichiers sans extension séparés par une virgule')
    basefichs=input('nom(s): ')
    basefichs=basefichs.split(',')
    for b in basefichs:
        serfiles.append(WorkDir+b.strip()+'.ser')
    # parametres en dur
    flag_display=False
    ratio_fixe=0
    shift=0
    
    sys.exit()
    
#code commun mac ou windows
#************************************************************************************************

#pour gerer la tempo des affichages des images resultats dans cv2.waitKey
#sit plusieurs fichiers à traiter

def do_work():
    if len(serfiles)==1:
        tempo=4000
    else:
        tempo=1
        
    # boucle sur la liste des fichers
    for serfile in serfiles:
        print (serfile)

        if serfile=='':
            sys.exit()
        
        WorkDir=os.path.dirname(serfile)+"/"
        os.chdir(WorkDir)
        base=os.path.basename(serfile)
        basefich=os.path.splitext(base)[0]
        if base=='':
            print('erreur nom de fichier : ',serfile)
            sys.exit()
        
        # met a jour le repertoire si on a changé dans le fichier ini
        try:
            with open('c:/py/pysolex.ini', "w") as f1:
                f1.writelines(WorkDir)
        except:
            pass
        
        # ouverture du fichier ser
        try:
            f=open(serfile, "rb")
        except:
            print('erreur ouverture fichier : ',serfile)
            sys.exit()
        f.close()
            
        
        # appel au module d'extraction, reconstruction et correction
        #
        # basefich: nom du fichier ser sans extension et sans repertoire
        # dx: decalage en pixel par rapport au centre de la raie 
        try:
            shift=int(shift)
        except:
            shift=0

        options = {'flag_display':flag_display, 'shift':shift, 'save_fit':save_fit}
        if not ratio_fixe == 0:
            options['ratio_fixe'] = ratio_fixe
        if not slant_fix == '':
            try:           
                options['slant_fix'] = math.radians(float(slant_fix))
            except:
                print('invalid slant input: '+ slant_fix)
                pass
        
        frame, header, cercle=sol.solex_proc(serfile,options)
        
        base=os.path.basename(serfile)
        basefich=os.path.splitext(base)[0]
       
        
        ih=frame.shape[0]
        newiw=frame.shape[1]
        
        
        
        # gere reduction image png
        if ih>600:
            sc=0.5
            
        # gere reduction image png
        if ih>1900:
            sc=0.4
            
        # gere reduction image png
        if ih>2500:
            sc=0.2

        flag_result_show = flag_display
        if flag_result_show:       
            cv2.namedWindow('sun0', cv2.WINDOW_NORMAL)
            cv2.moveWindow('sun0', -10, 0)
            cv2.resizeWindow('sun0', (int(newiw*sc), int(ih*sc)))
            cv2.imshow('sun0',frame)
            
            
            cv2.namedWindow('sun', cv2.WINDOW_NORMAL)
            cv2.moveWindow('sun', 0, 0)
            cv2.resizeWindow('sun', (int(newiw*sc), int(ih*sc)))
            
            cv2.namedWindow('sun2', cv2.WINDOW_NORMAL)
            cv2.moveWindow('sun2', 200, 0)
            cv2.resizeWindow('sun2', (int(newiw*sc), int(ih*sc)))
            
            if ratio_fixe==0:
                cv2.namedWindow('sun3', cv2.WINDOW_NORMAL)
                cv2.moveWindow('sun3', 400, 0)
                cv2.resizeWindow('sun3', (int(newiw*sc), int(ih*sc)))

            cv2.namedWindow('clahe', cv2.WINDOW_NORMAL)
            cv2.moveWindow('clahe', 600, 0)
            cv2.resizeWindow('clahe',(int(newiw*sc), int(ih*sc)))
        
        # create a CLAHE object (Arguments are optional)
        #clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(5,5))
        clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(2,2))
        cl1 = clahe.apply(frame)
        
        #image leger seuils
        frame1=np.copy(frame)
        Seuil_bas=np.percentile(frame, 25)
        Seuil_haut=np.percentile(frame,99.9999)
        frame1[frame1>Seuil_haut]=65000
        #print('seuil bas', Seuil_bas)
        #print('seuil haut', Seuil_haut)
        fc=(frame1-Seuil_bas)* (65000/(Seuil_haut-Seuil_bas))
        fc[fc<0]=0
        frame_contrasted=np.array(fc, dtype='uint16')


        if flag_result_show:
            cv2.imshow('sun',frame_contrasted)
        #cv2.waitKey(0)
        
        #image seuils serres 
        frame1=np.copy(frame)
        Seuil_haut=np.percentile(frame1,99.9999)
        Seuil_bas=(Seuil_haut*0.25)
        #print('seuil bas HC', Seuil_bas)
        #print('seuil haut HC', Seuil_haut)
        frame1[frame1>Seuil_haut]=65000
        fc2=(frame1-Seuil_bas)* (65000/(Seuil_haut-Seuil_bas))
        fc2[fc2<0]=0
        frame_contrasted2=np.array(fc2, dtype='uint16')
        if flag_result_show:
            cv2.imshow('sun2',frame_contrasted2)
            #cv2.waitKey(0)
        
        #image seuils protus
        frame1=np.copy(frame)
        #MattC cut out disk first
        #MattC BTW given default setting of disk_display this was never executed
        #if ratio_fixe==0 and disk_display==True:
        #MattC for now, let's always do this
        if 1==1 :
            #MattC - demo simple processing of proms seperate from disc settings
            clahe2 = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(1,1))
            cl2 = clahe2.apply(frame1)
            x0=cercle[0]
            y0=cercle[1]-1
            r=int(cercle[2]*0.5)+1
            #MattC - turn disc to black, more aesthetically pleasing, IMHO
            frame1=cv2.circle(cl2, (x0,y0),r,0,-1)
        #MattC adjust scaling
        Seuil_haut=np.percentile(frame1,99.9999)
        Seuil_bas=0
        print('seuil bas protu', Seuil_bas)
        print('seuil haut protu', Seuil_haut)
        frame1[frame1>Seuil_haut]=Seuil_haut
        fc2=(frame1-Seuil_bas)* (65000/(Seuil_haut-Seuil_bas))
        fc2[fc2<0]=0
        frame_contrasted3=np.array(fc2, dtype='uint16')
        #MattC code that was at this line moved up above ...
        if flag_result_show:
            cv2.imshow('sun3',frame_contrasted3)
            #cv2.waitKey(0)
        
        Seuil_bas=np.percentile(cl1, 25)
        Seuil_haut=np.percentile(cl1,99.9999)*1.05
        cc=(cl1-Seuil_bas)*(65000/(Seuil_haut-Seuil_bas))
        cc[cc<0]=0
        cc=np.array(cc, dtype='uint16')
        if flag_result_show:
            cv2.imshow('clahe',cc)
            cv2.waitKey(tempo)  #affiche 15s et continue
        if not clahe_only:
            #sauvegarde en png pour appliquer une colormap par autre script
            cv2.imwrite(basefich+'_disk.png',frame_contrasted)
            #sauvegarde en png pour appliquer une colormap par autre script
            cv2.imwrite(basefich+'_diskHC.png',frame_contrasted2)
            #sauvegarde en png pour appliquer une colormap par autre script
            cv2.imwrite(basefich+'_protus.png',frame_contrasted3)
        #sauvegarde en png de clahe
        cv2.imwrite(basefich+'_clahe.png',cc)
        
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
        #sauvegarde le fits
        if options['save_fit']:
            DiskHDU=fits.PrimaryHDU(frame2,header)
            DiskHDU.writeto(basefich+'_clahe.fit', overwrite='True')

        cv2.destroyAllWindows()

if 0:        
    cProfile.run('do_work()', sort='cumtime')
else:
    do_work()
