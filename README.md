Solar disk reconstruction from SHG (spectroheliography) SER files. Only 16bit files are accepted. No AVI files are accepted.
If no spectral line can recognised in the SER file, the program will stop.

Install the most recent version of Python from python.org. During Windows installation, check the box to update the PATH.

For Windows, double click the windows_setup file to install the needed Python libraries.

Usage:

Graphical user interface: launch SHG_MAIN (by double clicking under Windows). A Windows Desktop shortcut can also be created.
In the Python GUI window, enter the name of the SER file(s) to be processed. Batch processing is possible but will halt if a file is unsuitable.

Command line interface: python SHG_MAIN.py your_ser_file1.SER [your_ser_file2.SER ... if batch processing]

Command line options:
- d : display all graphics
- c : only the CLAHE image is saved
- f : all FITS files are saved
- p : save the protuberance image
- w: a,b,c will produce images at a, b and c ; x:y:w will produce images starting at x, finishing at y, every w pixels

Check the "Show graphics" box for a 'live view' reconstruction display, a graphic of the geometry correction and a quick view of the final images.
This will increase processing time significantly. This feature is not recommended for batch processing.

If the "Save .fits files" box is checked, the following files will be stored in the same directory as the SER file:

- xx_mean.fits: average image of all the frames in the SER video of the spectral line
- xx_img.fits: raw image reconstruction
- xx_flat.fits: corrected flat image
- xx_recon.fits: final image, corrected for tilt
- xx_clahe.fits: final image, with Contrast Limited Adaptive Histogram Equalization

If the "Save CLAHE.png only" box is checked, then only the final PNG image with Contrast Limited Adaptive Histogram Equalization will be saved.

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

Geometry correction may fail under certain circumstances. In this case, enter the Y/X ratio and Tilt angle manually (try 1, 0 initially).

For rapid processing during data acquisition, make sure "Show graphics" is off.
If Y/X is set to 1, distortion due to inappropriate scanning speed vs frame rate can be recognised and optimised.
Similarly, if Tilt is set to 0, instrument misalignment can be recognised and corrected.

Geometry graphics window can be killed by killed by pushing 'X'.
Composite results window should be killed by pushing any key on the keyboard.
