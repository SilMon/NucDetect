[![PyPI version](https://badge.fury.io/py/NucDetect.svg)](https://badge.fury.io/py/NucDetect) [![Downloads](https://pepy.tech/badge/nucdetect)](https://pepy.tech/project/nucdetect)

NucDetect - A python package for Detection and Quantification of DNA Doublestrand Breaks
============

NucDetect is a Python package for the detection and quantification of γH2AX and 53BP1 foci inside nuclei. Its written in 
pure Python 3.7, obeys the PEP 8 style guidelines and includes PEP 484 type hints as well as Epytext docstrings.

### Note

The current release is a very early alpha version. Please report report any detected bugs and/or improvement suggestions.

Requirements
============

NucDetect is compatible with Windows, Mac OS X and Linux operating systems. It requires 
the following packages:

* tensorflow>=2.1.0
* numpy>=1.18.1
* scikit-image>=0.16.2
* matplotlib>=3.1.3
* pyqt5>=5.14.1
* numba>=0.48.0
* pillow>=7.0.0
* qtawesome>=0.6.1
* piexif>=1.1.3

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

___

Author: Romano Weiss

Co-Author: Stefan Rödiger
