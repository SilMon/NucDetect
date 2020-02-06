[![PyPI version](https://badge.fury.io/py/NucDetect.svg)](https://badge.fury.io/py/NucDetect)

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

* scipy>=0.19.0
* numpy>=1.13.3
* scikit-image>=0.15
* matplotlib >= 3.0.2
* PyQT5 >= 5.11.3
* Pillow
* qtawesome
* piexif
* tensorflow >= 2.0
* numba >= 0.45.1

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

### Supported Image Formats

Following image formats are supported by NucDetect:
* TIFF
* PNG
* JPG
* BMP

___

Author: Romano Weiss

Co-Author: Stefan Rödiger
