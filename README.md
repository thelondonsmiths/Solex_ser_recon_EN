Solar disk reconstruction from SHG (spectroheliography) video files. Both 16bit and 8bit files are accepted. SER and AVI files are accepted.
If no spectral line can recognised in the video file, the program will stop.

- Install the most recent version of Python from python.org. During Windows installation, check the box to update the PATH.

- For Windows, double click the windows_setup batch file to install the needed Python libraries.
If you are installing an update of this software, double click on window_update batch file to make sure you are using the most recent Python libraries. 
It is important that the PIP package manager itself is up to date for some of the libraries to install correctly. 
This is particularly the case if using an older (unmaintained) version of Python (e.g. Windows 7). 
In this case, run `pip install -U pip` in a Command Prompt window and follow the specific instructions.

### **Usage**

**Graphical user interface**: launch SHG_MAIN (by double clicking under Windows). A Windows Desktop shortcut can also be created.
In the Python GUI window, enter the name of the video file(s) to be processed. Batch processing is possible but will halt if a file is unsuitable.

**Command line interface example**: `python SHG_MAIN.py serfile1.SER` [serfile2.SER ... if batch processing]

**Command line options**:
- d : display all graphics
- c : only the CLAHE image is saved
- f : all FITS files are saved
- w : a,b,c will produce images at a, b and c ; x:y:w will produce images starting at x, finishing at y, every w pixels
- p : disable black disk on protuberance images
- t : disable transversalium correction
- m : mirror flip in the x-direction
- s : crop width to make square
- r : crop width to a constant number of pixels

Check the "Show graphics" box for a 'live view' display of the reconstruction and a peek at the final png images.
This will increase processing time significantly. This feature is not recommended for batch processing.

Note that all output files will be saved in the same directory as the video file. If the program is run a second time, it will overwrite the original output files.

If the "Save fits files" box is checked, the following files will be stored:

- _filename_mean.fits_: average image of all the frames in the video of the spectral line
- _filename_raw.fits_: raw image reconstruction
- _filename_circular.fits_: geometrically corrected image
- _filename_detransversaliumed.fits_: image corrected for line defects
- _filename_clahe.fits_: final image, with Contrast Limited Adaptive Histogram Equalization

If the "Save clahe.png only" box is checked, then only the png image with Contrast Limited Adaptive Histogram Equalization will be saved.
This is the most useful output file for stacking purposes.

If the "Crop width square" box is checked, the width is cropped to be the same as the height, with the Sun centred.
This feature is particularly helpful for stacking frames (which typically require them all to be the same dimensions) and creating animations and mosaics.
The crop square feature is only useful for full-disk images.
If the width is smaller than the height, the Sun is centred and some dark space is added on each side to make the image square.

If the "Fixed image width" box is specified, the width of the image is cropped to that number of pixels. 
The "Fixed image width" overrides the "Crop square" function and works in the same way by centering and then cropping or filling with dark space.
This feature is particularly useful for stacking frames of partial (non-full-disk) scans and creating animations and mosaics.
The value of the fixed width is remembered.

If "Mirror X" is checked, the image is reversed after the geometric correction to compensate for scanning in the reverse direction. This applies to all files (png and fits).
The choice of "Mirror X" is deliberately not remembered.

The "Rotate png images" silder applies only to the final png output images. Rotation is counterclockwise in degrees (0, 90, 180, 270)

If "Correct transversalium lines" is not checked, then the program makes no attempt to fix line defects.
The function for fixing line defects works well if they are small and well-defined. Wide lines may require a more manual process.
In this case, turning off the automated function may be helpful. The default should be for this box to be checked.
For protuberance images, the ones with the black disc, turning off the transversalium is recommended. This because the transversalium filter effects the dark sky background but is useless if the surface is masked.

The slider for "Transversalium correction strength" can be adjusted from low (weak) to high (strong).
The number corresponds to the width of a window in hundreds of pixels for creation of a "flat".
The middle range (around 2 to 3) is typically a good value for Hydrogen alpha. Very wide defects may require a higher value, while very "clean" images may retain more features with a lower setting.
Calcium images (with very strong contrast) seem to work better with a low value (around 1).
Hydrogen beta images (with typically low contrast) seem to work better with a higher value (around 4).
Continuum images, because they have weaker features, can be done with a higher setting.
For stacking, it is recommended to use a lower correction strength than for a single frame. This is because the stacking itself will tend to average out the transversalium noise.

Y/X ratio: enter a specific Y/X ratio, if this is known. Leave blank for auto-correction. Enter 1 if no correction desired.

Tilt angle: enter a specific tilt angle in degrees. Leave blank for auto-correction. Enter 0 if no tilt correction desired.

Pixel offset: offset in pixels from the minimum of the line to reconstruct the image on another wavelength (displaced from the central minimum).
- For no shift, leave the "Pixel offset" box at the default of '0'
- Specify the output of a particular shift by entering a single number or particular values with commas: 'a,b,c,d,e'
- For a range x to y with an interval of w, use colons: 'x:y:w'
- If 'w' not specified, the default is 1 so  'x:y' will produce the range x, x+1, x+2, ... y-2, y-1, y
- x, y, a, b, c can be positive or negative integers; the number w can only be a positive integer
- Batch pixel shift processing of a batch of files is allowed

Protus adjustment: make the black circle larger or smaller in radius by inputting a positive or negative integer (typically between -10 and +10).
If you want to turn off the black disk altogether, then enter a negative number greater than the radius (e.g. -9999).
The proftus adjustment setting is remembered.

Geometry correction may fail under certain circumstances (one example being a partial eclipse). In this case, enter the Y/X ratio and Tilt angle manually (try 1, 0 initially).

For rapid processing during data acquisition, make sure "Show graphics" is off.
If Y/X is set to 1, distortion due to inappropriate scanning speed vs frame rate can be recognised and optimised.
Similarly, if Tilt is set to 0, instrument misalignment can be recognised and corrected.

The composite results window should be killed by pushing any key on the keyboard.
By default, the Processing GUI will reappear after each run.
The prior file location and several other GUI states are now saved in the _SHG_config_ file (previously in a _SHG.ini_ file).
In CLI mode, the parameters in the _SHG_config_ file are ignored.

A file _serfile_log_ is generated with a number of useful parameters. In particular:
- **Y/X ratio**: in general, this should be close to 1. If it is larger than 1.1, then the data is likely being undersampled and so a higher FPS or slower scan speed may be helpful.
If it is smaller than 0.9, then oversampling is probably occurring and the scan speed could be increased.
- **Unrotation**: this approximately corresponds to the misorientation of the SHG instrument with the scan direction (i.e. RA or DEC).
It should be possible to reduce this to around 0.5 degrees without too much difficulty, at which point the raw scan will show very little instrument tilt.
- **Disk radius**: this figure is useful for a number of post-processing steps. If doing a "fixed image width" crop, then chose a value at least 2.2 times the radius.