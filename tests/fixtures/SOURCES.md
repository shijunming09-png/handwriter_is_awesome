# 测试素材来源

`synthetic_*.png`、`single_page.pdf`、`two_pages.pdf`、`layout_scenarios.pdf`、`blank.png`、`rotated_90.*`、中文路径下的 PDF，以及损坏、加密 PDF，由 `tools/make_fixtures.py` 在本地生成。文字使用系统字体，测试的是渲染、切行、排序及错误处理，**不是真实手写稿，也不能用来宣称手写识别准确率**。

可选的 `iam_industry.jpg` 是 Hugging Face 官方测试数据集提供的 IAM 手写单词示例（参考文字为 `industry`），用于 TrOCR 集成测试：

- 图片：https://huggingface.co/datasets/hf-internal-testing/fixtures_ocr/resolve/main/iam_picture.jpeg
- 官方测试数据集：https://huggingface.co/datasets/hf-internal-testing/fixtures_ocr
- 带参考文字的测试：https://github.com/huggingface/transformers/blob/main/tests/models/vision_encoder_decoder/test_modeling_vision_encoder_decoder.py
- IAM 数据库及使用条件：https://fki.tic.heia-fr.ch/databases/iam-handwriting-database

该示例用于本地评估，**不包含在便携应用交付包中**。IAM 数据集有其自身使用条件，模型的 MIT 许可不等同于图片的许可。`iam_three_lines.*` 仅是同一单词重复排版后的切行、页序测试输入，不能当作三个独立手写样本，也不能用来代表整页连笔识别质量。原 IAM 网站的完整示例证书验证失败，此处使用正常 TLS 验证的官方镜像，未绕过证书验证。

字符错误率使用人工参考文字和识别结果之间的 Levenshtein 距离除以参考字符数；词错误率以空白分词后同样计算。测试报告应同时说明样本数量和是否保留大小写、标点，避免把单行示例的表现推广到其他笔迹。

## 多词手写行

`tools/make_fixtures.py --download-iam-lines` 只通过公开 dataset viewer 取 `Teklia/IAM-line` 的 test split 前三条记录，不下载完整数据集。`iam_test_000.jpg` 至 `iam_test_002.jpg` 是真实手写行，对应 TXT 为数据集提供的人工转写标签；标签保留了 IAM 对标点分词产生的空格。

- 数据集：https://huggingface.co/datasets/Teklia/IAM-line
- 公开 viewer API：https://datasets-server.huggingface.co/first-rows?dataset=Teklia%2FIAM-line&config=default&split=test
- 来源、稳定行号和标签记录在 `iam_lines_manifest.json`；临时签名 URL 查询部分不持久化。

`iam_multiline.pdf` 和 PNG 将这三条不同的手写行排到同一张白页，用来验证渲染、自动切行和识别流程；它是人工重排的测试页，并非一张原始 IAM 手稿整页。素材仍只用于本地评估，不包含在便携应用中。与之配套的识别结果报告必须说明仅有三行，不能推广为通用准确率。
