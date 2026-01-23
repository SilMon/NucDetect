import setuptools

with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="NucDetect",
    version="1.0",
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
        "Programming Language :: Python :: 3.12"
    ],
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=[
        "tensorflow-cpu>=2.17.0",
        "scikit-image>=0.16.2",
        "matplotlib>=3.1.3",
        "seaborn>=0.13.2",
        "statannotations>=0.7.2",
        "pyqt5>=5.14.1",
        "numba>=0.48.0",
        "pillow>=11.13.0",
        "qtawesome==1.3.1",
        "piexif>=1.1.3",
        "pyqtgraph>=0.14.0",
        "pandas>=2.1.4",
        "imagecodecs>=2026.1.1",
        "openpyxl>=3.1.5",
        "PyWavelets>=1.9.0",
    ]
)
