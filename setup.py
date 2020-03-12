import setuptools

with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="NucDetect",
    version="0.11.14.dev1",
    description=("Module to identify intranuclear proteins on basis of "
                 "fluorescence images."),
    author="Romano Weiss",
    url="https://github.com/SilMon/NucDetect",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7"
    ],
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=[
        "tensorflow==2.0.0",
        "numpy>=1.18.1",
        "scikit-image>=0.16.2",
        "matplotlib>=3.1.3",
        "pyqt5>=5.14.1",
        "numba>=0.48.0",
        "pillow>=7.0.0",
        "qtawesome>=0.6.1",
        "piexif>=1.1.3"
    ]
)
