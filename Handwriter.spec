# Build with the Python environment described in requirements-lock.txt.
from pathlib import Path
import importlib.util
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

root = Path(SPECPATH)
hidden = []
transformers_root = Path(importlib.util.find_spec('transformers').origin).parent
peft_root = Path(importlib.util.find_spec('peft').origin).parent
vision_root = Path(importlib.util.find_spec('torchvision').origin).parent
# torchvision >= 0.29 loads these stable-ABI extensions by filename; the
# stock PyInstaller hook still looks for the former _C/image module names.
vision_binaries = [(str(file), 'torchvision') for file in vision_root.iterdir()
                   if file.suffix.lower() in {'.pyd', '.dll'}]
for package in (
    'transformers.models.trocr', 'transformers.models.vision_encoder_decoder',
    'transformers.models.vit', 'transformers.models.roberta',
    'transformers.models.gpt2', 'transformers.models.bert',
):
    package_root = transformers_root.joinpath(*package.split('.')[1:])
    for source in package_root.rglob('*.py'):
        relative = source.relative_to(transformers_root.parent).with_suffix('')
        parts = list(relative.parts)
        if parts[-1] == '__init__':
            parts.pop()
        if 'modeling_tf_' not in source.name and 'modeling_flax_' not in source.name:
            hidden.append('.'.join(parts))
hidden += [
    'transformers.models.auto.configuration_auto',
    'transformers.models.auto.modeling_auto',
    'transformers.models.auto.tokenization_auto',
    'transformers.models.auto.image_processing_auto',
    'transformers.models.auto.processing_auto',
]
for source in peft_root.rglob('*.py'):
    parts = list(source.relative_to(peft_root.parent).with_suffix('').parts)
    if parts[-1] == '__init__':
        parts.pop()
    hidden.append('.'.join(parts))
datas = collect_data_files('transformers', includes=['**/*.json'])
datas += collect_data_files('huggingface_hub', includes=['templates/*'])
datas += [(str(root / 'assets' / 'handwriter.png'), 'assets')]
for distribution in ('torch', 'torchvision', 'transformers', 'tokenizers', 'safetensors', 'huggingface-hub', 'peft', 'accelerate'):
    datas += copy_metadata(distribution)
analysis = Analysis(
    [str(root / 'run_app.py')], pathex=[str(root)],
    binaries=vision_binaries, datas=datas, hiddenimports=hidden,
    hookspath=[], hooksconfig={}, runtime_hooks=[str(root / 'tools' / 'runtime_hook.py')],
    excludes=[
        'pytest', 'reportlab', 'pypdf', 'matplotlib', 'scipy', 'sklearn',
        'tensorflow', 'keras', 'jax', 'flax', 'IPython', 'notebook',
        'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets',
        'PySide6.QtWebEngineQuick', 'PySide6.QtPdf', 'PySide6.QtPdfWidgets',
        'PySide6.QtQuick', 'PySide6.QtQml', 'PySide6.QtMultimedia',
        'tkinter', 'torchaudio', 'torchtext',
    ],
    noarchive=False, module_collection_mode={'handwriter': 'py', 'peft': 'pyz+py', 'accelerate': 'pyz+py'},
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz, analysis.scripts, [], exclude_binaries=True,
    name='Handwriter', debug=False, bootloader_ignore_signals=False,
    strip=False, upx=False, console=False,
    icon=str(root / 'assets' / 'handwriter.ico'),
)
app = COLLECT(
    exe, analysis.binaries, analysis.datas,
    strip=False, upx=False, name='Handwriter-Personal',
)
