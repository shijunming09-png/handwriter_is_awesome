"""Use Windows 10/11 system ICU/UCRT, not DLLs found on another app's PATH.

PyInstaller can discover Poppler's renamed ICU DLL or libheif's old API-set
forwarders on the build host. Those filenames also name Windows system DLLs,
but their exported functions are not interchangeable. They must not shadow
the operating system's libraries in this Windows 10/11 application.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys


def fix_bundle(application: Path) -> list[str]:
    root = application.resolve()
    internal = root / '_internal'
    if not internal.is_dir() or not (root / 'Handwriter.exe').is_file():
        raise ValueError(f'Not a Handwriter portable folder: {root}')
    removed = []
    system_names = {'ucrtbase.dll', 'icuuc.dll', 'icuin.dll', 'icudt.dll'}
    for file in internal.rglob('*.dll'):
        name = file.name.lower()
        if name not in system_names and not name.startswith(('api-ms-win-', 'ext-ms-win-')):
            continue
        if not file.resolve().is_relative_to(internal.resolve()):
            raise ValueError('Refusing to modify a file outside the portable application.')
        removed.append(str(file.relative_to(root)))
        file.unlink()
    (root / 'bundle-fixes.json').write_text(json.dumps({
        'target': 'Windows 10/11 x64',
        'reason': 'Use OS ICU/UCRT instead of unrelated PATH DLLs.',
        'removed_shadowing_system_libraries': removed,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    return removed


if __name__ == '__main__':
    files = fix_bundle(Path(sys.argv[1]))
    print(f'Bundle compatibility: removed {len(files)} shadowing system DLLs.')
