"""Assemble an independent Handwriter v2 release from verified evidence."""
from __future__ import annotations
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def assemble(app_dir, evidence_dir, output_dir):
    source = Path(__file__).resolve().parents[1]
    app_dir, evidence_dir, output_dir = [p.resolve() for p in (app_dir, evidence_dir, output_dir)]
    required = ('tests', 'packaged-training', 'packaged-base-ocr', 'packaged-personal-ocr', 'dataset', 'backend', 'merged-reload')
    data = {name: json.loads((evidence_dir / (name + '.json')).read_text(encoding='utf-8')) for name in required}
    tests, training, dataset, backend = [data[k] for k in ('tests', 'packaged-training', 'dataset', 'backend')]
    if tests['passed'] < 1 or tests.get('failed', 0):
        raise ValueError('Automated tests did not pass.')
    if training['status'] != 'completed' or not training.get('adapter_updated') or not training.get('offline_verified') or training.get('network_attempts'):
        raise ValueError('Packaged training did not pass.')
    if not dataset.get('production_gate_enforced') or not dataset.get('worker_validate_dataset_passed') or not dataset.get('page_groups_do_not_cross_splits'):
        raise ValueError('Production dataset acceptance did not pass.')
    if not backend.get('success') or not data['merged-reload'].get('success'):
        raise ValueError('Resume/model reload did not pass.')
    for name in ('packaged-base-ocr', 'packaged-personal-ocr'):
        r = data[name]
        if not r['success'] or r['network_attempts'] or not r.get('desktop_classes_imported') or not r.get('export_content_and_order_match'):
            raise ValueError('Packaged OCR/export did not pass: ' + name)
    if not (app_dir / 'Handwriter.exe').is_file() or not (app_dir / 'licenses').is_dir():
        raise ValueError('Application or licenses are missing.')
    if any((app_dir / name).exists() for name in ('writers', 'projects')):
        raise ValueError('Refusing to distribute QA writer/project data.')
    output_dir.mkdir(parents=True, exist_ok=True)
    app = output_dir / 'Handwriter-Personal'
    if app.resolve() != app_dir:
        if app.exists():
            raise FileExistsError('Refusing to overwrite an existing application.')
        shutil.copytree(app_dir, app)
    for name in ('README.md', 'THIRD_PARTY_NOTICES.md'):
        shutil.copy2(source / name, app / name)
    size = sum(p.stat().st_size for p in app.rglob('*') if p.is_file())
    base, candidate = training['metrics']['baseline'], training['metrics']['candidate']
    counts = dataset['counts']
    report = f"""墨迹 Handwriter 2.0 · 个人英文笔迹训练版验证报告
日期：{date.today().isoformat()}

交付
独立便携目录 Handwriter-Personal，约 {size / 1024**3:.2f} GiB，包含运行库和通用模型。
无需安装 Python/CUDA Toolkit；训练需要 NVIDIA GPU，识别支持 CPU 回退。
不含测试作者档案、原稿、QA 训练模型或 IAM 图片/PDF。

自动测试
{tests['passed']} 项通过，耗时 {tests['elapsed_seconds']:.2f} 秒，失败 {tests.get('failed', 0)} 项。
覆盖旧任务迁移备份、人工确认与失效、按页划分与去重、固定测试集、
训练检查点/取消恢复、模型校验切换、校对保留、GUI线程回收和Word/TXT导出。
GUI使用源码Qt offscreen自动交互；不是人工操作完整桌面的验收。

生产数据门槛
合成5页/150行经实际300 DPI渲染、真实TrOCR分词器与裁切流程：
训练/验证/测试 = {counts['train']}/{counts['validation']}/{counts['test']}行。
生产门槛校验通过、页组不跨集合；此项是合成功能测试，不代表手写准确率。
正常界面要求至少150条有效确认行、5页，各组至少90/20/20行。

打包EXE的真实训练计算
设备：{training['device']}；混合精度：{training['amp']}。
冻结基座，48个目标模块，{training['trainable_parameters']}个LoRA可训练参数。
公开IAM手写样本内部小测试每组1行，显式绕过正常数据门槛：
优化器更新{training['global_step']}步，参数已改变；峰值显存{training['peak_gpu_mb']:.2f} MiB。
适配器、优化器检查点、标准合并模型保存成功，完整性校验通过。
取消/恢复阶段：{' / '.join(backend['stages'])}；完成后没有额外训练。
通用/候选模型使用相同行图、分词器、解码参数进行比较：
验证CER {base['validation']['cer']:.2%} → {candidate['validation']['cer']:.2%}，
验证WER {base['validation']['wer']:.2%} → {candidate['validation']['wer']:.2%}；
测试CER {base['test']['cer']:.2%} → {candidate['test']['cer']:.2%}，
测试WER {base['test']['wer']:.2%} → {candidate['test']['wer']:.2%}。
该小测试未观察到提升。样本少且不是用户的同一作者，不能推广为准确率结论。
IAM来源：https://huggingface.co/datasets/Teklia/IAM-line

打包EXE识别与导出
通用/个人模型均完成真实PDF渲染、检测3行、识别、保存和Word/TXT导出。
两种导出的内容与行序一致；Qt {data['packaged-base-ocr']['qt_version']}和桌面类加载成功。
训练、通用识别、个人识别均在进程中阻止网络socket连接，连接尝试为0。
这是进程级禁网验证，未断开系统网络、未修改系统安全配置。
模型只从本地加载；交付已带齐通用模型。

功能与效果
已验证训练功能、恢复、合并/重载、显式启用/回退和导出流程。
尚无这位作者的真实样稿，尚未验证个人笔迹识别效果改善。
需要真实校对数据和从未参与训练的保留页，才能完成个人效果验收。
候选模型不会自动启用；历史版本和通用模型保留。
每个正式个人模型另需约1.3 GB，训练前至少预留4 GiB。
"""
    for name in ('resume-equivalence', 'personal-cpu-reload', 'source-parity'):
        path = evidence_dir / (name + '.json')
        if not path.is_file():
            continue
        extra = json.loads(path.read_text(encoding='utf-8'))
        if not extra.get('success'):
            raise ValueError('Optional acceptance failed: ' + name)
        if name == 'resume-equivalence':
            report += '\n恢复等价性：96个适配器张量与连续两步训练最大差为0，优化器与随机状态恢复通过。\n'
        elif name == 'personal-cpu-reload':
            report += 'CPU补测：源引擎中强制禁用CUDA检测，打包训练所得个人模型实际CPU单行识别成功；不是EXE的CPU实测。\n'
    start = """墨迹 Handwriter 2.0 · 开始使用

1. 打开Handwriter-Personal文件夹，双击Handwriter.exe，保留整个文件夹。
2. 导入英文手写PDF，识别、对照校对，导出Word/TXT。
3. 打开“个人笔迹训练”，保存一位作者，把任务明确归入该作者。
4. 逐字检查后勾选“已逐字校对，可用于训练”，修改文字或行框后需重新确认。
5. 至少150条有效确认行、5页后，手动建立快照并训练；查看对比后再决定启用。
6. 可取消/恢复、切换个人模型和回退通用模型，已有人工校对保持优先。

无需安装Python；已含通用模型，可离线识别与训练，训练需要可用的NVIDIA GPU。
完整说明：Handwriter-Personal/README.md。
本版验证的是功能；没有真实作者样稿，尚不代表识别准确率已提升。
"""
    for folder in (output_dir, app):
        (folder / '验证报告.txt').write_text(report, encoding='utf-8')
        (folder / '开始使用.txt').write_text(start, encoding='utf-8')
    archive = output_dir / 'Handwriter-Personal-source.zip'
    allowed_roots = {'assets', 'handwriter', 'tests', 'tools'}
    allowed_files = {'README.md', 'THIRD_PARTY_NOTICES.md', 'Handwriter.spec', 'build.ps1', 'pyproject.toml', 'requirements-lock.txt', 'run_app.py'}
    excluded = {'__pycache__', '.pytest_cache', '.git', '.venv', 'models', 'writers', 'projects', 'logs', 'build', 'dist'}
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(source.rglob('*')):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(source)
            if any(part in excluded for part in relative.parts):
                continue
            if relative.parts[0] not in allowed_roots and str(relative) not in allowed_files:
                continue
            if relative.parts[:2] == ('tests', 'fixtures') and relative.name != 'SOURCES.md':
                continue
            if relative.name.lower().startswith('iam_') or path.suffix.lower() in {'.pyc', '.log'}:
                continue
            bundle.write(path, 'Handwriter-source/' + relative.as_posix())
        for path in sorted((app / 'licenses').rglob('*')):
            if path.is_file():
                bundle.write(path, 'Handwriter-source/licenses/' + path.relative_to(app / 'licenses').as_posix())
        bundle.writestr('Handwriter-source/验证报告.txt', report)
    info = {'version': '2.0.0', 'portable_folder': app.name, 'portable_size_bytes': size,
            'exe_sha256': sha(app / 'Handwriter.exe'), 'source_archive': archive.name,
            'source_sha256': sha(archive), 'automated_tests': tests,
            'functional_validation': True, 'personal_writer_accuracy_improvement_validated': False,
            'evidence_sha256': {p.name: sha(p) for p in evidence_dir.glob('*.json')}}
    (output_dir / 'release-info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('app-dir', 'evidence-dir', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.app_dir, args.evidence_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

