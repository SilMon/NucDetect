[![PyPI version](https://badge.fury.io/py/NucDetect.svg)](https://badge.fury.io/py/NucDetect) [![Downloads](https://pepy.tech/badge/nucdetect)](https://pepy.tech/project/nucdetect)

NucDetect - A python package for Detection and Quantification of DNA Doublestrand Breaks
============

NucDetect is a Python package for the detection and quantification of γH2AX and 53BP1 foci inside nuclei. Its written in 
pure Python 3.7, obeys the PEP 8 style guidelines and includes PEP 484 type hints as well as Epytext docstrings.

![Result](https://github.com/SilMon/NucDetect_Additional_Data/blob/main/WIKI/result.png)

Requirements
============

NucDetect is compatible with Windows, Mac OS X and Linux operating systems. It requires 
the following packages:

* tensorflow-cpu>=2.17.0
* scikit-image>=0.16.2
* matplotlib>=3.1.3
* seaborn>=0.13.2
* statannotations>=0.7.2
* pyqt5>=5.14.1
* numba>=0.48.0
* pillow>=11.13.0
* qtawesome==1.3.1
* piexif>=1.1.3
* pyqtgraph>=0.14.0
* pandas>=2.1.4
* imagecodecs>=2026.1.1
* openpyxl>=3.1.5
* PyWavelets>=1.9.0

### Important note
While higher package versions than listed might be okay, due to the fragile nature of some of the used packaged, there's
a good probability that the program might break. In this case, try first to use listed minimal versions.

Installation
============
Run the following commands to clone and install from GitHub

```console
$ git clone https://github.com/SilMon/NucDetect.git
```

or pypi
```console
python3 -m pip install NucDetect
```

### For Windows users
Download the packed program that can be found under "Releases" and place the folder where ever you desire. The program 
can be started by running the *NucDetect.exe* file.

Start
============
The program can be started by running the NucDetectAppQT.py:
```console
cd %UserProfile%/AppData/local/Programs/Python/python37/Lib/site-packages/gui
python -m NucDetectAppQT
```
*First start*: Switch to the created NucDetect Folder, which will be created in User directory. Then place images you
want to analyse into the images folder and click the reload button. This will load all images and create a thumbnail for
each (needed to decrease the memory footprint of QT). This can take several minutes, depending on the number of images
and used hardware (e.g. around 5 min for 2200 images on a Ryzen 3700X processor). Progress will be displayed in the
command prompt.

### Supported Image Formats

Following image formats are supported by NucDetect:
* TIFF
* PNG
* JPG
* BMP

### Not supported

* Grayscale images
* Binary images

### Wiki
Detailed information about the program can be found on the ![wiki](https://github.com/SilMon/NucDetect/wiki)

### Supplementary Data
https://github.com/SilMon/NucDetect_Additional_Data
___

Author: Romano Weiss

Co-Author: Stefan Rödiger
