# 墨迹 Handwriter

Windows 本地英文手写扫描 PDF 识别与个人笔迹训练工具。

Local English handwritten PDF transcription and writer adaptation with TrOCR.

本项目采用 AI 辅助开发。当前版本已完成识别、校对、训练和导出流程的功能验证；尚未证明对某位真实作者的识别准确率有所提升。

## 可以做什么

- 导入 PDF，渲染页面、校正轻微倾斜并自动检测文字行。
- 使用 TrOCR 识别英文手写，支持原稿对照校对、行框调整、拆分、合并和选定行重新识别。
- 保存任务并继续处理，导出按原页序排列的 Word 和 UTF-8 TXT。
- 为一位作者积累逐字确认的训练样本，手动进行 LoRA 微调。
- 查看通用模型与候选模型的 CER/WER 和逐行差异，手动启用个人模型或回退通用模型。
- 训练支持进度、取消和安全检查点恢复；模型准备完成后可离线使用，文稿保留在本机。

优先支持单栏、基本水平排列的英文手写扫描稿。复杂多栏、表格还原、多作者管理、模型训练以外的版面重建和可搜索 PDF 不在当前范围内。

## 技术

| 环节 | 实现 |
| --- | --- |
| 桌面界面 | Python 3.12、PySide6 |
| PDF 与页面处理 | pypdfium2、OpenCV |
| 文字行检测 | PP-OCRv5 mobile detector、ONNX Runtime CPU |
| 手写识别 | microsoft/trocr-base-handwritten、Transformers、PyTorch |
| 作者适配 | PEFT LoRA，冻结基座，训练 decoder 注意力的 q_proj/v_proj |
| 文档导出 | python-docx、UTF-8 文本 |
| Windows 打包 | PyInstaller，独立便携文件夹 |

识别优先使用可用的 NVIDIA GPU，失败时回退 CPU。当前训练功能需要 NVIDIA GPU。

## 从源码运行

仓库只包含源码和说明，模型与运行库不在 Git 仓库中。首次准备模型需要网络连接。

在 Windows 安装 Python 3.12 后，在仓库目录执行 PowerShell：

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe run_app.py
~~~

启动后点击“准备 / 检查模型”，程序会从指定的公开模型仓库下载文件并校验权重。完成后识别和训练可离线进行，无需云端 OCR 账号。

依赖包含 PyTorch CUDA 构建，下载和磁盘占用较大。GPU 需要兼容的 NVIDIA 驱动，无需另装 CUDA Toolkit。

## 使用流程

1. 导入英文手写 PDF，点击“开始 / 继续识别”。
2. 对照原稿检查识别文字，必要时调整行框并重新识别。
3. 导出 Word 或 TXT；任务与校对会保存到本机。
4. 需要训练时，打开“个人笔迹训练”，建立一位作者并把任务明确归入作者。
5. 逐字核对后勾选“已逐字校对，可用于训练”；也可在完整检查后确认本页。
6. 数据达到门槛后建立快照，手动训练、查看对比结果，再决定是否启用候选模型。

修改文字、移动或拆合行框、旋转页面会使相关训练确认失效。识别成功或修改过文字不会自动视为人工确认。模型切换和重新识别保留已有人工校对。

详细操作、旧任务兼容与便携包说明见 [使用说明](docs/PORTABLE.md)。

## 训练与评估

实验门槛为至少 150 条有效确认行、5 个原稿页面，训练 / 验证 / 测试组分别至少 90 / 20 / 20 行。

- 按完整页面约 60% / 20% / 20% 划分，重复页面和裁图不跨集合。
- 首次正式快照固定保留验证与测试页面，此后新增页面进入训练组。
- 排除空白、不确定、失效、多行转写和超过 128 token 的标签，避免静默截断。
- LoRA 参数为 r=8、alpha=16、dropout=0.05；批量1、梯度累积8、学习率1e-4、最多20轮。
- 依据验证集 CER 选择最佳检查点，连续3轮不改善停止；测试集不用于优化器训练。
- 候选模型不会自动启用，保留适配器、通用模型、正式版本和回退入口。

每个正式合并模型另需约1.3 GB，训练前至少预留4 GiB。150行只是实验门槛，不代表效果保证；应以同一作者未参与训练的页面评估。

## 验证范围

对应 Handwriter 2.0 交付的本地验证记录：

- 68 项自动测试通过，包括确认失效、旧任务迁移、去重与按页划分、GUI线程、恢复与导出。
- RTX 5060 Laptop GPU 完成真实前向、反向和参数更新，适配器保存、恢复及合并重载通过。
- 恢复训练与连续两步训练的96个适配器张量完全一致。
- 打包 EXE 在进程级禁网条件下完成小样本训练，以及通用 / 个人模型的 PDF 识别和 Word/TXT 导出。
- 生产门槛使用合成5页150行验证；真实训练流程使用公开 IAM 小样本内部测试，显式绕过正常数据门槛。
- 小样本测试未观察到准确率提升，不能推断实际作者效果。

详见 [验证报告](验证报告.txt)。上述是本地交付验证记录，不是 GitHub Actions 的持续集成状态。

运行源码测试：

~~~powershell
$env:QT_QPA_PLATFORM = 'offscreen'
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp .\build\test-temp
~~~

合成素材由测试自动生成。IAM 素材和转写不随仓库分发；如自行获取，请遵循 [素材来源说明](tests/fixtures/SOURCES.md)。

## 打包

~~~powershell
.\build.ps1
~~~

产物为 dist/Handwriter-Personal/Handwriter.exe。此步骤生成应用运行库，通用模型可随后在应用中准备；预装模型的便携交付约6.5 GiB，需保留完整文件夹。

## 数据与许可

models、projects、writers、logs 等运行目录由 .gitignore 排除。请勿提交个人原稿、作者档案、训练数据、模型权重或凭据。

第三方组件与模型许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 licenses/。项目原创源码的发布许可证尚未选择。发布选择记录见 [PUBLISHING.md](docs/PUBLISHING.md)。

本项目借鉴 eScriptorium 的“校对—训练—评估”流程，采用现有 TrOCR 的 LoRA 微调，不包含 eScriptorium/Kraken 服务或其模型。

