# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = ['imagecodecs._shared', 'imagecodecs._imcd', 'h5py.defs', 'h5py.utils', 'h5py.h5ac', 'h5py._proxy']
hiddenimports += collect_submodules('tensorflow')


block_cipher = None


a = Analysis(
    ['..\\gui\\NucDetectAppQT.py'],
    pathex=[],
    binaries=[],
    datas=[('../gui/definitions/css/*.css', 'gui/definitions/css/'), ('../gui/definitions/images/*.png', 'gui/definitions/images/'), ('../gui/definitions/ui/*.ui', 'gui/definitions/ui/'), ('../gui/settings/*.ui', 'gui/settings'), ('../gui/settings/*.json', 'gui/settings'), ('../fcn/model/detector.h5', 'fcn/model/'), ('../core/database/scripts/*.sql', 'core/database/scripts/')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NucDetect',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='NucDetect',
)
