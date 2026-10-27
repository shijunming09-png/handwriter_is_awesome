"""Keep optional runtime caches and progress output out of user directories."""
import os
import sys
from pathlib import Path

root = Path(sys.executable).resolve().parent
os.environ.setdefault('HF_HOME', str(root / 'models' / '.cache'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
# Model loading is always local; explicit model preparation uses urllib.
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
