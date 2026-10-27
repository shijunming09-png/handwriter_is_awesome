$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$environmentPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $environmentPython)) {
    py -3.12 -m venv (Join-Path $projectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python 3.12 environment.' }
}
& $environmentPython -m pip install --upgrade pip
& $environmentPython -m pip install -r (Join-Path $projectRoot 'requirements-lock.txt') --extra-index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $environmentPython (Join-Path $projectRoot 'tools\make_icon.py')
$basePythonDirectory = & $environmentPython -c 'import sys; print(sys.base_prefix)'
# Keep unrelated Poppler/libheif/other application DLLs off the build search path.
$env:PATH = @((Split-Path -Parent $environmentPython), $basePythonDirectory, (Join-Path $env:SystemRoot 'System32'), $env:SystemRoot) -join ';'
$env:PYINSTALLER_CONFIG_DIR = Join-Path $projectRoot 'build\cache'
& $environmentPython -m PyInstaller --noconfirm --distpath (Join-Path $projectRoot 'dist') --workpath (Join-Path $projectRoot 'build') (Join-Path $projectRoot 'Handwriter.spec')
if ($LASTEXITCODE -ne 0) { throw 'Packaging failed. See build logs.' }
$portableRoot = Join-Path $projectRoot 'dist\Handwriter-Personal'
& $environmentPython (Join-Path $projectRoot 'tools\fix_windows_bundle.py') $portableRoot
if ($LASTEXITCODE -ne 0) { throw 'Bundle compatibility checks failed.' }
Copy-Item -LiteralPath (Join-Path $projectRoot 'README.md') -Destination $portableRoot
Copy-Item -LiteralPath (Join-Path $projectRoot 'THIRD_PARTY_NOTICES.md') -Destination $portableRoot
& $environmentPython (Join-Path $projectRoot 'tools\collect_licenses.py') (Join-Path $portableRoot 'licenses')
Write-Output "Application: $portableRoot\Handwriter.exe"
