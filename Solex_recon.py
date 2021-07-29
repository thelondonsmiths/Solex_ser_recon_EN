# -*- coding: utf-8 -*-
"""
@author: valerie desnoux
with improvements by Andrew Smith
Version 24 July 2021

------------------------------------------------------------------------

modif seuil dans section flat sur NewImg pour eviter de clamper à background 1000
mise en commentaire d'un ajout d'une detection des bords du disque solaire et d'un ajustement ce cercle apres
la circularisation en calculant le rayon a partir de la hauteur du disque
------------------------------------------------------------------------

-----------------------------------------------------------------------------
calcul sur une image des ecarts simples entre min de la raie
et une ligne de reference
-----------------------------------------------------------------------------
"""

from solex_util import *
from ser_read_video import *
from ellipse_to_circle import ellipse_to_circle, correct_image


def make_header(rdr):        
    # initialisation d'une entete fits (etait utilisé pour sauver les trames individuelles
    hdr= fits.Header()
    hdr['SIMPLE']='T'
    hdr['BITPIX']=32
    hdr['NAXIS']=2
    hdr['NAXIS1'] = rdr.iw
    hdr['NAXIS2'] = rdr.ih
    hdr['BZERO']=0
    hdr['BSCALE']=1
    hdr['BIN1']=1
    hdr['BIN2']=1
    hdr['EXPTIME']=0
    return hdr

# compute mean image of video
def compute_mean(serfile):
    """IN : serfile path"
    OUT :numpy array
    """
    rdr = ser_reader(serfile)
    logme('Width, Height : '+str(rdr.Width)+' '+str(rdr.Height)) 
    logme('Number of frames : '+str(rdr.FrameCount))
    my_data = np.zeros((rdr.ih, rdr.iw),dtype='uint64')
    while rdr.has_frames():
        img = rdr.next_frame()
        my_data += img
    return (my_data / rdr.FrameCount).astype('uint16')

def compute_mean_return_fit(serfile, options, LineRecal = 1): 
    global hdr, ih, iw

    """
    ----------------------------------------------------------------------------
    Reconstuit l'image du disque a partir de l'image moyenne des trames et 
    des trames extraite du fichier ser
    avec un fit polynomial
    Corrige de mauvaises lignes et transversallium
 
    basefich: nom du fichier de base de la video sans extension, sans repertoire
    shift: ecart en pixel par rapport au centre de la raie pour explorer 
    longueur d'onde decalée
    ----------------------------------------------------------------------------
    """
    flag_display = options['flag_display']
    # first compute mean image
    # rdr is the ser_reader object
    mean_img= compute_mean(serfile)

    """
    ----------------------------------------------------------------------------
    Calcul polynome ecart sur une image au centre de la sequence
    ----------------------------------------------------------------------------
    """

    
    #savefich=basefich+'_mean'

    if options['save_fit']:
        DiskHDU=fits.PrimaryHDU(mean_img,header=hdr)
        DiskHDU.writeto(basefich+'_mean.fits', overwrite='True')


    #affiche image moyenne
    if flag_display:
        cv2.namedWindow('Ser mean', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Ser mean', iw, ih)
        cv2.moveWindow('Ser mean', 100, 0)
        cv2.imshow ('Ser mean', mean_img)
        if cv2.waitKey(2000) == 27:                     # exit if Escape is hit
            cv2.destroyAllWindows()
            sys.exit()
        
        cv2.destroyAllWindows()
    
    y1,y2=detect_bord(mean_img, axis=1, offset=5)
    logme('Vertical limits y1, y2 : '+str(y1)+' '+str(y2))
    
    PosRaieHaut=y1
    PosRaieBas=y2
    
    """
    -----------------------------------------------------------
    Trouve les min intensité de la raie
    -----------------------------------------------------------
    """
    # construit le tableau des min de la raie a partir du haut jusqu'en bas
    MinOfRaie=[]
    
    for i in range(PosRaieHaut,PosRaieBas):
        line_h=mean_img[i,:]
        MinX=line_h.argmin()
        MinOfRaie.append([MinX,i])
        #print('MinOfRaie x,y', MinX,i)
    
    #best fit d'un polynome degre 2, les lignes y sont les x et les colonnes x sont les y
    np_m=np.asarray(MinOfRaie)
    xm,ym=np_m.T
    #LineRecal=xm.min()
    
    p=np.polyfit(ym,xm,2)
    
    #calcul des x colonnes pour les y lignes du polynome
    a=p[0]
    b=p[1]
    c=p[2]
    fit=[]
    #ecart=[]
    for y in range(0,ih):
        x=a*y**2+b*y+c
        deci=x-int(x)
        fit.append([int(x)-LineRecal,deci,y])
    return fit, a, b, c

def correct_bad_lines_and_geom(Disk, options):
    global hdr, basefich
    
    iw=Disk.shape[1]
    ih=Disk.shape[0]
    img=Disk
    
    y1,y2=detect_bord (img, axis=1,offset=5)    # bords verticaux
    
    #detection de mauvaises lignes
    
    # somme de lignes projetées sur axe Y
    ysum=np.mean(img,1)
    #plt.plot(ysum)
    #plt.show()
    # ne considere que les lignes du disque avec marge de 15 lignes 
    ysum=ysum[y1+15:y2-15]
    
    # filtrage sur fenetre de 31 pixels, polynome ordre 3 (etait 101 avant)
    yc=savgol_filter(ysum,31, 3)

    # divise le profil somme par le profil filtré pour avoir les hautes frequences
    hcol=np.divide(ysum,yc)

    # met à zero les pixels dont l'intensité est inferieur à 1.03 (3%)
    hcol[abs(hcol-1)<=0.03]=0

    
    # tableau de zero en debut et en fin pour completer le tableau du disque
    a=[0]*(y1+15)
    b=[0]*(ih-y2+15)
    hcol=np.concatenate((a,hcol,b))
    #plt.plot(hcol)
    #plt.show()
    
    # creation du tableau d'indice des lignes a corriger
    l_col=np.where(hcol!=0)
    listcol=l_col[0]
    

    # correction de lignes par filtrage median 13 lignes, empririque
    for c in listcol:
        m=img[c-7:c+6,]
        s=np.median(m,0)
        img[c-1:c,]=s
    
    #sauvegarde le fits

    if options['save_fit']:
        DiskHDU=fits.PrimaryHDU(img,header=hdr)
        DiskHDU.writeto(basefich+'_corr.fits', overwrite='True')
    return img

def correct_transversalium(img, flag_nobords, options):
    global hdr, ih, basefich
    frame = img
    newiw=img.shape[1]
    ih=img.shape[0]
    flag_nobords = False
    # on cherche la projection de la taille max du soleil en Y
    y1,y2=detect_bord(frame, axis=1,offset=0)
    #print ('flat ',y1,y2)
    # si mauvaise detection des bords en x alors on doit prendre toute l'image
    if flag_nobords:
        ydisk=np.median(img,1)
    else:
        #plt.hist(frame.ravel(),bins=1000,)
        #plt.show()
        #plt.hist(frame.ravel(),bins=1000,cumulative=True)
       # plt.show()
        seuil_bas=np.percentile(frame,25)
        seuil_haut=np.percentile(frame,97) 
        #print ('Seuils de flat: ',seuil_bas, seuil_haut)
        #print ('Seuils bas x: ',seuil_bas*4)
        #print ('Seuils haut x: ',seuil_haut*0.25)
        #myseuil=seuil_haut*0.2
        myseuil=seuil_haut*0.5
        # filtre le profil moyen en Y en ne prenant que le disque
        ydisk=np.empty(ih+1)
        for j in range(0,ih):
            temp=np.copy(frame[j,:])
            temp=temp[temp>myseuil]
            if len(temp)!=0:
                ydisk[j]=np.median(temp)
            else:
                ydisk[j]=1
    y1=y1
    y2=y2
    ToSpline= ydisk[y1:y2]
 
    
    Smoothed2=savgol_filter(ToSpline,301, 3) # window size, polynomial order
    #best fit d'un polynome degre 4
    np_m=np.asarray(ToSpline)
    ym=np_m.T
    xm=np.arange(y2-y1)
    p=np.polyfit(xm,ym,4)
    
    #calcul des x colonnes pour les y lignes du polynome
    a=p[0]
    b=p[1]
    c=p[2]
    d=p[3]
    e=p[4]
    Smoothed=[]
    for x in range(0,y2-y1):
        y=a*x**4+b*x**3+c*x**2+d*x+e
        Smoothed.append(y)
    
    """
    plt.plot(ToSpline)
    plt.plot(Smoothed)
    plt.plot(Smoothed2)
    plt.show()
    """

    
    # divise le profil reel par son filtre ce qui nous donne le flat
    hf=np.divide(ToSpline,Smoothed2)
       
    # elimine possible artefact de bord
    hf=hf[5:-5]
    
    #reconstruit le tableau du pofil complet 
    a=[1]*(y1+5)
    b=[1]*(ih-y2+5)
    hf=np.concatenate((a,hf,b))
    
    
    Smoothed=np.concatenate((a,Smoothed,b))
    ToSpline=np.concatenate((a,ToSpline,b))
    Smoothed2=np.concatenate((a,Smoothed2,b))
    
    """
    plt.plot(ToSpline)
    plt.plot(Smoothed2)
    plt.show()
    
    plt.plot(hf)
    plt.show()
    """
    
    # genere tableau image de flat 
    flat=[]
    for i in range(0,newiw):
        flat.append(hf)
        
    np_flat=np.asarray(flat)
    flat = np_flat.T
    #evite les divisions par zeros...
    flat[flat==0]=1
    
    """
    plt.imshow(flat)
    plt.show()
    """
    
    # divise image par le flat
    BelleImage=np.divide(frame,flat)
    frame=np.array(BelleImage, dtype='uint16')
    # sauvegarde de l'image deflattée
    if options['save_fit']:
        DiskHDU=fits.PrimaryHDU(frame,header=hdr)
        DiskHDU.writeto(basefich+'_flat.fits', overwrite='True')
    return frame

def solex_proc(serfile, options):
    global hdr, ih, iw, basefich
    clearlog()
    #plt.gray()              #palette de gris si utilise matplotlib pour visu debug
    logme('Using pixel shift : ' + str(options['shift']))
    WorkDir=os.path.dirname(serfile)+"/"
    os.chdir(WorkDir)
    base=os.path.basename(serfile)
    basefich=os.path.splitext(base)[0]
    LineRecal=1
    rdr = ser_reader(serfile)
    hdr = make_header(rdr)
    ih = rdr.ih
    iw = rdr.iw
    
    fit, a, b, c = compute_mean_return_fit(serfile, options, LineRecal)
    
    # Modification Jean-Francois: correct the variable names: A0, A1, A2
    logme('Coeff A0, A1, A2 :  '+str(a)+'  '+str(b)+'  '+str(c))
    
    Disk, ih, iw, FrameCount = read_video_improved(serfile, fit, LineRecal, options)

    hdr['NAXIS1']=iw # note: slightly dodgy, new width
   
    #sauve fichier disque reconstruit

    if options['save_fit']:
        DiskHDU=fits.PrimaryHDU(Disk,header=hdr)
        DiskHDU.writeto(basefich+'_img.fits', overwrite='True')
    
    if options['flag_display']:
        cv2.destroyAllWindows()
    
    """
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    on passe au calcul des mauvaises lignes et de la correction geometrique
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    """
    img = correct_bad_lines_and_geom(Disk, options)
        
    """
    --------------------------------------------------------------
    on echaine avec la correction de transversallium
    --------------------------------------------------------------
    """
    flag_nobords = False
    frame_flatted = correct_transversalium(img,flag_nobords, options)

    """
    We now apply ellipse_fit to apply the geometric correction

    """

    if not 'ratio_fixe' in options and not 'slant_fix' in options:
        frame_circularized, cercle = ellipse_to_circle(frame_flatted, options)
    else:
        ratio = options['ratio_fixe'] if 'ratio_fixe' in options else 1.0
        phi = math.radians(options['slant_fix']) if 'slant_fix' in options else 0.0
        frame_circularized, cercle = correct_image(frame_flatted / 65536, phi, ratio), (-1, -1, -1) # Note that we assume 16-bit

    # sauvegarde en fits de l'image finale
    
    if options['save_fit']:
        DiskHDU=fits.PrimaryHDU(frame_circularized,header=hdr)
        DiskHDU.writeto(basefich+'_recon.fits', overwrite='True')
            
    with  open(basefich+'_log.txt', "w") as logfile:
        logfile.writelines(mylog)
    
    return frame_circularized, hdr, cercle
    


