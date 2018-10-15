<<<<<<< HEAD
from setuptools import setup, find_packages

setup(name="NucDetect",
      author="Romano Weiss",
      packages=find_packages(),
      version="0.0.1")
=======
'''
Created on 02.10.2018

@author: Romano Weiss
'''
import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="Nuclear Detector",
    version="0.0.1-1",
    description=("Module to identify intranuclear proteins on basis of"
                 "fluorescence images."),
    author="Romano Weiss",
    url="https://github.com/SilMon/NucDetect",
    long_description=read("README"),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7"
    ],
    packages=find_packages(),
)
>>>>>>> 8cb4461152deeaba96f216fd2150993a4a2c506d
