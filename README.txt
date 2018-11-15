README.txt

Developed on: macOS High Sierra (10.13.6)
Language: Python 2.7.14
Packages:
	os      (included in python)
	math    (included in python)
	pickle  (included in python)
	numpy (1.14.0)
	opencv-python (3.4.3.18)
	matplotlib (2.1.2)
	scipy (1.0.0)

Setup: 
	- Ensure Python 2.7 is installed.
	- Run the following command to install dependencies for the project:
		pip install -r requirements.txt
	- NOTE: WhiteImages.py MUST be ran before HDR.py, as the former generates .pickle files that are used by the latter.

The two program files WhiteImages.py and HDR.py can be found in the root folder.

When running both files and an image is displayed, simply close it for the program to advance.

Running WhiteImages.py will generate and display all graphs for part 1. The images for part one can be found in the WhiteImages folder. 

NOTE: The HDR algorithms are relatively slow, and may take a minute or two to complete each.
Running HDR.py will generate and display all graphs for parts 2, 3, 4 as well as the tone mapped HDR images. The full HDR stack and the tone mapped HDR images are available in the HDRStack folder.

All graphs for all parts of the project are located within the report PDF as well as the PlotsAndHists folder.