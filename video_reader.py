"""
@author: Andrew Smith 
contributors: Valerie Desnoux, Matt Considine
Version 24 July 2023

"""
import numpy as np
import cv2 #MattC

class video_reader:

    def __init__(self, file, buffer_size = 25):
        # ouverture et lecture de l'entete du fichier ser
        self.file = file
        self.buffer_size = buffer_size
        self.buffer_remaining = 0
        
        if self.file.upper().endswith('.SER'): #MattC 20210726
            self.SER_flag=True
            self.AVI_flag=False
        elif self.file.upper().endswith('.AVI'):
            self.SER_flag=False
            self.AVI_flag=True
            self.infiledatatype='uint8'
        else:
            raise Exception('error input file ' + file + 'neither is SER nor AVI')
        
        #ouverture et lecture de l'entete du fichier ser

        if self.SER_flag: #MattC
            self.FileID=np.fromfile(file, dtype='int8',count=14)
            offset=14

            self.LuID=np.fromfile(file, dtype=np.uint32, count=1, offset=offset)
            offset=offset+4
        
            self.ColorID=np.fromfile(file, dtype='uint32', count=1, offset=offset)
            offset=offset+4
        
            self.littleEndian=np.fromfile(file, dtype='uint32', count=1,offset=offset)
            offset=offset+4
        
            self.Width=np.fromfile(file, dtype='uint32', count=1,offset=offset)[0]
            offset=offset+4
        
            self.Height=np.fromfile(file, dtype='uint32', count=1,offset=offset)[0]
            offset=offset+4
        
            PixelDepthPerPlane=np.fromfile(file, dtype='uint32', count=1,offset=offset)
            self.PixelDepthPerPlane=PixelDepthPerPlane[0]
            offset=offset+4
        
            FrameCount=np.fromfile(file, dtype='uint32', count=1,offset=offset)
            self.FrameCount=FrameCount[0]
        
            if self.PixelDepthPerPlane==8:
                self.infiledatatype='uint8'
                self.count=self.Width*self.Height       # Nombre d'octet d'une trame
                self.infilebytes=1
            else:
                self.infiledatatype='uint16'
                self.count=self.Width*self.Height      # Nombre d'octet d'une trame
                self.infilebytes=2
            self.FrameIndex=-1             # Index de trame, on evite les deux premieres
            self.offset=178               # Offset de l'entete fichier ser
            self.fileoffset=178 #MattC to avoid stomping on offset accumulator
            
        elif self.AVI_flag: #MattC 
    	    #deal with avi file
            self.file = cv2.VideoCapture(file)

            self.Width = int(self.file.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.Height = int(self.file.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.PixelDepthPerPlane=1*8
            self.FrameCount = int(self.file.get(cv2.CAP_PROP_FRAME_COUNT))            
            self.count=self.Width*self.Height
            self.infilebytes=1            
            self.FrameIndex=-1
            self.offset = 0
            self.fileoffset = 0 #MattC to avoid stomping on offset accumulator
        else: #MattC
    	    ok_flag = False

        if self.Width > self.Height:
            self.flag_rotate = True
            self.ih = self.Width
            self.iw = self.Height
        else:
            self.flag_rotate = False
            self.iw = self.Width
            self.ih = self.Height
        #print(f'in video reader with nframes, height, width = {self.FrameCount}, {self.ih}, {self.iw}')

    def next_frame(self):
        self.FrameIndex += 1
        self.offset = self.fileoffset + self.FrameIndex * self.count * self.infilebytes #MattC track offset
      
        if self.SER_flag: #MattC
            if self.buffer_remaining == 0:
                self.buf = np.fromfile(
                    self.file,
                    dtype = self.infiledatatype,
                    count = self.count * max(0, min(self.buffer_size, self.FrameCount - self.FrameIndex)),
                    offset=self.offset)
                self.buffer_remaining = self.buffer_size
            i = self.buffer_size - self.buffer_remaining
            img = self.buf[self.count*i : self.count*(i+1)]
            self.buffer_remaining -= 1

            
        elif self.AVI_flag:
            ret, img = self.file.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise Exception('error input file is neither is SER nor AVI')

        img = np.reshape(img, (self.Height, self.Width))
        
        if self.flag_rotate:
            img = np.rot90(img)
        if self.infiledatatype == 'uint8':
            img = np.asarray(img, dtype='uint16')*256 #upscale 8-bit to 16-bit
        return img

    def has_frames(self):
        return self.FrameIndex + 1 < self.FrameCount
