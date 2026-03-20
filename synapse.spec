# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect ALL torch_geometric .py source files.
# TorchScript's @torch.jit.script requires access to original .py sources at runtime.
torch_geo_datas = collect_data_files('torch_geometric', include_py_files=True)
torch_geo_hiddens = collect_submodules('torch_geometric')

# It is important to find dynamic imports to include them as hiddenimports
hidden_imports = [
    'src.utils.logging_setup',
    'ui.styles.theme_manager',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'src.infrastructure.postgres_manager',
    'sqlalchemy',
    'paho.mqtt',
    'torch',
    'transformers',
] + torch_geo_hiddens

a = Analysis(
    ['synapse.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('Model Vault/qwen3_1.7B', 'Model Vault/qwen3_1.7B'),
        ('ui/assets/images', 'ui/assets/images'),
        ('ui/translations', 'ui/translations'),
    ] + torch_geo_datas,
    hiddenimports=hidden_imports,
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
    name='synapse',
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
    icon='ui/assets/images/logo.png'
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='synapse',
)
