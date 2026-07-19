# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import runpy
import sys

from PyInstaller.utils.hooks import collect_all


ROOT = Path.cwd().resolve()
APP_NAME = "BL03U-MassSpectrumTool"
ICON_ICO = ROOT / "icons" / "icon.ico"
ICON_ICNS = ROOT / "icons" / "icon.icns"
APP_VERSION = runpy.run_path(
    str(ROOT / "src" / "bl03u_masstool" / "_version.py")
)["__version__"]

datas = [
    (str(ROOT / "icons"), "icons"),
    (str(ROOT / "src" / "bl03u_masstool" / "resources"), "resources"),
    (str(ROOT / "data" / "examples"), "data/examples"),
]
rdkit_datas, rdkit_binaries, rdkit_hiddenimports = collect_all("rdkit")
datas += rdkit_datas

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT / "src"), str(ROOT)],
    binaries=rdkit_binaries,
    datas=datas,
    hiddenimports=rdkit_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

if sys.platform == "darwin":
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=APP_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
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
        name=APP_NAME,
    )
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        version=APP_VERSION,
        bundle_identifier="cn.ihep.bl03u.mass-spectrum-tool",
        icon=str(ICON_ICNS),
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name=APP_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=str(ICON_ICO) if sys.platform == "win32" else None,
    )
