Solar disk reconstruction from SHG (spectroheliography) video files. Both 16bit and 8bit files are accepted. SER and AVI files are accepted.
If no spectral line can recognised in the video file, the program will stop.

- Install the most recent version of Python from python.org. During Windows installation, check the box to update the PATH.

- For Windows, double click the windows_setup batch file to install the needed Python libraries.
If you are installing an update of this software, double click on window_update batch file to make sure you are using the most recent Python libraries. 
It is important that the PIP package manager itself is up to date for some of the libraries to install correctly. 
This is particularly the case if using an older (unmaintained) version of Python (e.g. Windows 7). 
In this case, run `pip install -U pip` in a Command Prompt window and follow the specific instructions.

### **Usage**

**Graphical User Interface**: launch SHG_MAIN (by double clicking under Windows). A Windows Desktop shortcut can also be created.
In the Python UI window, enter the name of the video file(s) to be processed. Batch processing is possible but will halt if a file is unsuitable.

**Command Line Interface example**: `python SHG_MAIN.py serfile1.SER` [serfile2.SER ... if batch processing]

**Command line options**:

- d : display all graphics
- c : only the CLAHE image is saved
- f : all FITS files are saved
- h : displays help menu
- m : mirror flip in the x-direction
- p : disable black disk on protuberance image
- r : crop width to a constant number of pixels
- s : crop width to make square
- t : disable transversalium correction
- w : a,b,c will produce images at a, b and c ; x:y:w will produce images starting at x, finishing at y, every w pixels
- x : disable ellipse fit

Check the "Show graphics" box for a 'live view' display of the reconstruction and a peek at the final png images.
This will increase processing time significantly. This feature is not recommended for batch processing.
The composite png peek window can be killed early by pushing any key on the keyboard (default is 30 sec in single file mode and 5 sec in batch mode).
For rapid processing during data acquisition, make sure "Show graphics" is off.

Note that by default all output files will be saved in the same directory as the video file. If the program is run a second time, it will overwrite the original output files.
A different output folder can also be chosen using the UI. This destination will be remembered for subsequent processing.

In "File input mode", you can choose one or more files. The working directory is remembered for the next time the program runs.
In "Folder input mode", you can choose a particular folder to process all the files in batch mode (all with the same settings). The input direction is remembered.
If "Continuous detect mode" is selected, the program will process all the files already in the folder then wait until a new file arrives, at which point it will also be processed.
A 600x600 pixel CLAHE image of the last file processed will be displayed while the program is waiting.
If "Continuous detect mode" is on, then generally "Show graphics" should be off.

When the program starts, it will always be in "File input mode" rather than "Folder input mode".

If the "Save fits files" box is checked, the following files will be stored:

- _filename_mean.fits_: average image of all the frames in the video of the spectral line
- _filename_raw.fits_: raw image reconstruction
- _filename_circular.fits_: geometrically corrected image
- _filename_detransversaliumed.fits_: image corrected for line defects
- _filename_clahe.fits_: final image, with Contrast Limited Adaptive Histogram Equalization

If the "Save clahe.png only" box is checked, then only the png image with Contrast Limited Adaptive Histogram Equalization will be saved.
This is typically the most useful output file for stacking purposes.

If the "Crop width square" box is checked, the width is cropped to be the same as the height, with the Sun centred.
This feature is particularly helpful for stacking frames (which typically require them to all be the same dimensions) and creating animations and mosaics.
The crop square feature is only useful for full-disk images.
If the width is smaller than the height, the Sun is centred and some dark space is added on each side to make the image square.

If the "Fixed image width" box is specified, the width of the image is cropped to that number of pixels. 
The "Fixed image width" overrides the "Crop square" function and works in the same way by centering and then cropping or filling with dark space.
This feature is particularly useful for stacking frames of partial (non-full-disk) scans and creating animations and mosaics.
The value of the fixed width is remembered.

If "Mirror X" is checked, the image is reversed after the geometric correction to compensate for scanning in the reverse direction. This applies to all files (png and fits).
The choice of "Mirror X" is deliberately not remembered.

The "Rotate png images" silder applies only to the final png output images. Rotation is counterclockwise in degrees (0, 90, 180, 270).

If "Correct transversalium lines" is not checked, then the program makes no attempt to fix line defects. The default should be for this box to be checked.
The function for fixing line defects works well if they are fairly narrow. Wide, dark lines may require a more manual process.
For protuberance images, the ones with the black disc, turning off the transversalium is recommended. This because the transversalium filter affects the dark sky background but is irrelevant if the surface is masked.

The slider for "Transversalium correction strength" can be adjusted from low (weak) to high (strong).
The number corresponds to the width of a window in hundreds of pixels for the creation of a "flat".
The default setting of 3 seems to work well for most spectral lines. Very wide defects may be improved by using a higher value.

Y/X ratio: enter a specific Y/X ratio, if this is known. Leave blank for auto-correction. Enter 1 if no correction is desired.

Tilt angle: enter a specific tilt angle in degrees. Leave blank for auto-correction. Enter 0 if no tilt correction is desired.

Pixel offset: offset in pixels from the minimum of the spectral line to reconstruct the image on a wavelength displaced from the central minimum.
For the direction of positive and negative offsets, check the _spectral_line_data_ file. Negative is to the left and positive is to the right.
The spectral line will typically be curved and the apex of the parabolic curve will point towards the blue.
- For no shift, leave the "Pixel offset" box at the default of '0'
- Specify the output of a particular shift by entering a single number or particular values separated by commas: 'a,b,c,d,e' etc
- For a range x to y with an interval of w, use colons: 'x:y:w'
- If 'w' not specified, the default is 1 so  'x:y' will produce the range x, x+1, x+2, ... y-2, y-1, y
- x, y, a, b, c can be positive or negative integers; the number w can only be a positive integer
- Batch pixel shift processing of a batch of files is allowed

Protus adjustment: make the black circle larger or smaller in radius by inputting a positive or negative integer (typically between -10 and +10).
If you want to turn off the black disk altogether, then enter a negative number greater than the radius (e.g. -9999). The protus adjustment setting is remembered.

Geometry correction may fail under certain circumstances (one example being a partial eclipse). In this case, enter the Y/X ratio and Tilt angle manually (try 1, 0 initially).

By default, the Processing UI will reappear after each run.
The prior input file location and several other UI states are saved in the _SHG_config_ file.
In CLI mode, the UI parameters in the _SHG_config_ file are ignored.

A file _serfile_log_ is generated with a number of useful parameters. In particular:
- **Y/X ratio**: in general, this should be close to 1. If it is larger than 1.1, then the data is likely being undersampled and so a higher FPS or slower scan speed may be helpful.
If it is smaller than 0.9, then oversampling is probably occurring and the scan speed could be increased.
- **Unrotation**: this approximately corresponds to the misorientation of the SHG instrument with the scan direction (i.e. RA or DEC).
It should be possible to reduce this to around 0.5 degrees without too much difficulty, at which point the raw scan will show very little instrument tilt.
- **Disk radius**: this figure is useful for a number of post-processing steps. If doing a "fixed image width" crop, then chose a value at least 2.2 times the radius.