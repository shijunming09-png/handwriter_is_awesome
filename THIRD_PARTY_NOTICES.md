# Third-party components

Handwriter uses the following open-source components. This overview is not a substitute for the full licenses bundled with a binary distribution. Component versions are recorded in the dependency lock/installed-package manifest produced during the build.

| Component | License / notice | Upstream |
|---|---|---|
| Python | PSF License; bundled third-party notices apply | https://www.python.org/doc/copyright/ |
| PySide6 / Qt for Python | LGPLv3 / GPLv3 / commercial, as applicable to the distributed modules | https://doc.qt.io/qtforpython-6/licenses.html |
| PyTorch | BSD-style license and third-party notices | https://github.com/pytorch/pytorch/blob/main/LICENSE |
| Transformers | Apache-2.0 | https://github.com/huggingface/transformers/blob/main/LICENSE |
| Hugging Face Hub | Apache-2.0 | https://github.com/huggingface/huggingface_hub/blob/main/LICENSE |
| Microsoft TrOCR handwritten model | MIT (model repository metadata; retain accompanying notices) | https://huggingface.co/microsoft/trocr-base-handwritten |
| PaddleOCR / PP-OCR model and DB postprocessing algorithm | Apache-2.0 (retain notices supplied with the selected model). The application uses ONNX Runtime directly and does not bundle a PaddlePaddle inference runtime. | https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE |
| ONNX Runtime | MIT | https://github.com/microsoft/onnxruntime/blob/main/LICENSE |
| OpenCV | Apache-2.0 for current OpenCV releases; bundled-code notices also apply | https://github.com/opencv/opencv/blob/4.x/LICENSE |
| pypdfium2 / PDFium | Apache-2.0 OR BSD-3-Clause for pypdfium2; PDFium and bundled third-party notices apply | https://github.com/pypdfium2-team/pypdfium2/blob/main/LICENSE |
| python-docx | MIT | https://github.com/python-openxml/python-docx/blob/master/LICENSE |
| Pillow | MIT-CMU / historical PIL notice, plus bundled-code notices | https://github.com/python-pillow/Pillow/blob/main/LICENSE |
| NumPy | BSD-3-Clause and bundled-code notices | https://github.com/numpy/numpy/blob/main/LICENSE.txt |
| pyclipper / Clipper | MIT for pyclipper; accompanying Clipper and other notices apply | https://github.com/fonttools/pyclipper |
| PyInstaller (build tool) | GPL with bootloader distribution exception | https://pyinstaller.org/en/stable/license.html |

The complete distribution may also contain torchvision and transitive dependencies (for example tokenizers, safetensors, lxml, requests and packaging). The build collects installed license files and retains native-library notices, including NVIDIA CUDA redistributable notices where CUDA libraries are bundled. Development-only tools such as pytest, pypdf and ReportLab are used to create and inspect test fixtures; they are not required by the app's PDF renderer or document exporter.

Qt remains dynamically linked in the portable folder. Users must be able to replace the LGPL-covered Qt/PySide libraries under the applicable terms. This project does not impose restrictions that override third-party license rights.

## Evaluation-only sample

The optional IAM handwriting example is governed by the IAM Handwriting Database's own terms, not the model's MIT license. It is downloaded only for local evaluation and is excluded from the portable application. See `tests/fixtures/SOURCES.md` for provenance.

## Personal handwriting training additions

PEFT (Apache-2.0): https://github.com/huggingface/peft ; Accelerate (Apache-2.0): https://github.com/huggingface/accelerate . Full installed distribution licenses and versions are collected into the binary release. eScriptorium inspired the review/training workflow; this release does not bundle eScriptorium or Kraken code or models.
