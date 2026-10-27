"""Collect the installed distribution licenses for a reproducible release."""
from __future__ import annotations

import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import sys


def collect(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    inventory = []
    for distribution in metadata.distributions():
        name = distribution.metadata.get('Name', 'unknown')
        version = distribution.version
        licenses = []
        for relative in distribution.files or []:
            if not any(word in relative.name.lower() for word in ('license', 'copying', 'notice')):
                continue
            source = distribution.locate_file(relative)
            if not source.is_file() or source.suffix.lower() in {'.py', '.pyc', '.pyd', '.dll'}:
                continue
            if source.stat().st_size > 10 * 1024 * 1024:
                continue
            folder = destination / f'{name}-{version}'
            folder.mkdir(exist_ok=True)
            filename = str(relative).replace('\\', '_').replace('/', '_')
            shutil.copy2(source, folder / filename)
            licenses.append(str(Path(folder.name) / filename))
        inventory.append({'name': name, 'version': version, 'licenses': licenses})
    python_license = Path(sys.base_prefix) / 'LICENSE.txt'
    if python_license.is_file():
        shutil.copy2(python_license, destination / 'PYTHON-LICENSE.txt')
    (destination / 'DEPENDENCIES.json').write_text(
        json.dumps(sorted(inventory, key=lambda item: item['name'].lower()), ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


if __name__ == '__main__':
    collect(Path(sys.argv[1]))
