import setuptools

with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="NucDetect",
    version="0.20.0",
    description=("Module to quantify intranuclear foci on basis of "
                 "immunofluorescence images."),
    author="Romano Weiss",
    url="https://github.com/SilMon/NucDetect",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.9"
    ],
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=[
        "tensorflow>2.2.0",
        "numpy>1.18.1",
        "scikit-image>0.16.2",
        "matplotlib>3.1.3",
        "pyqt5>5.14.1",
        "numba>0.48.0",
        "pillow>7.0.0",
        "qtawesome>0.6.1",
        "piexif>=1.1.3",
        "pyqtgraph>=0.11.1"
    ]
)
