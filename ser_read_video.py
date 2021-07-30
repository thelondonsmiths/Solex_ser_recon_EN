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




