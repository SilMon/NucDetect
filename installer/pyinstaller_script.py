import PyInstaller.__main__
import shutil

if __name__ == "__main__":
    # Clear the dist folder
    shutil.rmtree("dist")
    # Run the installer script with parameters
    PyInstaller.__main__.run([
        '../gui/NucDetectAppQT.py',
        "--name=NucDetect",
        "--workpath=..",
        "--add-data=../gui/definitions/css/*.css;gui/definitions/css/",
        "--add-data=../gui/definitions/images/*.png;gui/definitions/images/",
        "--add-data=../gui/definitions/ui/*.ui;gui/definitions/ui/",
        "--add-data=../gui/settings/*.ui;gui/settings",
        "--add-data=../gui/settings/*.json;gui/settings",
        "--add-data=../fcn/model/detector.h5;fcn/model/",
        "--add-data=../core/database/scripts/*.sql;core/database/scripts/",
        "--collect-submodules=tensorflow",
        "--hidden-import=imagecodecs._shared",
        "--hidden-import=imagecodecs._imcd",
        "--hidden-import=h5py.defs",
        "--hidden-import=h5py.utils",
        "--hidden-import=h5py.h5ac",
        "--hidden-import=h5py._proxy"
    ])
