"""
@author: valerie desnoux
with improvements by Andrew Smith

"""
from solex_util import *

class ser_reader:

    def __init__(self, serfile):
            #ouverture et lecture de l'entete du fichier ser
        self.serfile = serfile
        self.FileID=np.fromfile(serfile, dtype='int8',count=14)
        offset=14

        self.LuID=np.fromfile(serfile, dtype=np.uint32, count=1, offset=offset)
        offset=offset+4
        
        self.ColorID=np.fromfile(serfile, dtype='uint32', count=1, offset=offset)
        offset=offset+4
        
        self.littleEndian=np.fromfile(serfile, dtype='uint32', count=1,offset=offset)
        offset=offset+4
        
        self.Width=np.fromfile(serfile, dtype='uint32', count=1,offset=offset)[0]
        offset=offset+4
        
        self.Height=np.fromfile(serfile, dtype='uint32', count=1,offset=offset)[0]
        offset=offset+4
        
        PixelDepthPerPlane=np.fromfile(serfile, dtype='uint32', count=1,offset=offset)
        self.PixelDepthPerPlane=PixelDepthPerPlane[0]
        offset=offset+4
        
        FrameCount=np.fromfile(serfile, dtype='uint32', count=1,offset=offset)
        self.FrameCount=FrameCount[0]
        
        self.count=self.Width*self.Height       # Nombre d'octet d'une trame
        self.FrameIndex=-1             # Index de trame, on evite les deux premieres
        self.offset=178               # Offset de l'entete fichier ser
    
        if self.Width>self.Height:
            self.flag_rotate=True
            self.ih=self.Width
            self.iw=self.Height
        else:
            self.flag_rotate=False
            self.iw=self.Width
            self.ih=self.Height
        

    def next_frame(self):
        self.FrameIndex += 1
        self.offset=178+self.FrameIndex*self.count*2
        img=np.fromfile(self.serfile, dtype='uint16',count=self.count, offset=self.offset)
        img=np.reshape(img,(self.Height,self.Width))
        
        if self.flag_rotate:
            img=np.rot90(img)
        return img

    def has_frames(self):
        return self.FrameIndex+1 < self.FrameCount

# read video and return constructed image of sun using fit and LineRecal
def read_video_improved(serfile, fit, LineRecal, options):
    rdr = ser_reader(serfile)
    ih, iw = rdr.ih, rdr.iw
    
    if options['flag_display']:
        cv2.namedWindow('disk', cv2.WINDOW_NORMAL)
        FrameMax=rdr.FrameCount
        cv2.resizeWindow('disk', FrameMax//3, ih//3)
        cv2.moveWindow('disk', 200, 0)
        #initialize le tableau qui va recevoir la raie spectrale de chaque trame
        Disk=np.zeros((ih,FrameMax), dtype='uint16')
        
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.moveWindow('image', 0, 0)
        cv2.resizeWindow('image', int(iw), int(ih))
    else:
        #Disk=np.zeros((ih,1), dtype='uint16')
        FrameMax=rdr.FrameCount
        Disk=np.zeros((ih,FrameMax), dtype='uint16')
        
    shift = options['shift']
    ind_l = (np.asarray(fit)[:, 0] + np.ones(ih) * (LineRecal + shift)).astype(int)
    
    #CLEAN if fitting goes too far
    ind_l[ind_l < 0] = 0
    ind_l[ind_l > iw - 2] = iw - 2
    ind_r = (ind_l + np.ones(ih)).astype(int)
    left_weights = np.ones(ih) - np.asarray(fit)[:, 1]
    right_weights = np.ones(ih) - left_weights

    # lance la reconstruction du disk a partir des trames
    print('reader num frames:', rdr.FrameCount)
    while rdr.has_frames():
        img = rdr.next_frame()               
        if options['flag_display'] and rdr.FrameIndex % 10 == 0 :
            cv2.imshow('image', img)
            if cv2.waitKey(1)==27:
                cv2.destroyAllWindows()
                sys.exit()

        left_col = img[np.arange(ih), ind_l]
        right_col = img[np.arange(ih), ind_r]
        IntensiteRaie = left_col*left_weights + right_col*right_weights
        
        #ajoute au tableau disk 
        Disk[:,rdr.FrameIndex]=IntensiteRaie
        
        if options['flag_display'] and rdr.FrameIndex % 10 ==0:
            cv2.imshow ('disk', Disk)
            if cv2.waitKey(1) == 27:                     # exit if Escape is hit
                cv2.destroyAllWindows()    
                sys.exit()
    return Disk, ih, iw, rdr.FrameCount



