# 基于卷积神经网络的文本情感分类

本项目完成 Rotten Tomatoes 句子级五分类情感分类任务，主模型为 TextCNN，并提供 FastText 作为轻量 baseline。代码面向课程作业提交，包含可运行源码、Docker 环境、实验输出保存、中文报告模板和演示脚本。

## 项目结构

```text
.
├── configs/                 # 实验配置
├── preprocessed_file/       # 已预处理数据
├── reports/                 # 报告模板和演示脚本
├── scripts/                 # 常用运行脚本
├── src/                     # 源代码
├── Dockerfile
├── requirements.txt
└── README.md
```

## 数据

数据集为 Rotten Tomatoes 电影评论句子，标签为五分类：

| 标签 | 含义 |
| --- | --- |
| 0 | very negative |
| 1 | negative |
| 2 | neutral |
| 3 | positive |
| 4 | very positive |

当前预处理数据规模：

| split | 样本数 |
| --- | ---: |
| train | 8544 |
| dev | 1101 |
| test | 2210 |

## 本地检查

```bash
PYTHONPYCACHEPREFIX=.pycache python -m compileall src
```

如果本地安装了依赖，也可以用 CPU 快速检查：

```bash
pip install -r requirements.txt
python -m src.train --config configs/fasttext.yaml --device cpu
```

## 服务器 Docker 运行

在服务器仓库根目录构建镜像：

```bash
docker build -t nlp-textcnn-sentiment .
```

使用 GPU 6 运行完整实验：

```bash
docker run --rm -it \
  --gpus '"device=6"' \
  -v "$PWD/outputs:/workspace/outputs" \
  -v "$PWD/checkpoints:/workspace/checkpoints" \
  nlp-textcnn-sentiment \
  bash scripts/run_all.sh
```

只训练 TextCNN：

```bash
docker run --rm -it \
  --gpus '"device=6"' \
  -v "$PWD/outputs:/workspace/outputs" \
  -v "$PWD/checkpoints:/workspace/checkpoints" \
  nlp-textcnn-sentiment \
  bash scripts/train_textcnn.sh
```

只评估 TextCNN 测试集：

```bash
docker run --rm -it \
  --gpus '"device=6"' \
  -v "$PWD/outputs:/workspace/outputs" \
  -v "$PWD/checkpoints:/workspace/checkpoints" \
  nlp-textcnn-sentiment \
  bash scripts/evaluate_textcnn.sh
```

运行后重点查看：

```text
outputs/textcnn_train_metrics.json
outputs/textcnn_test_metrics.json
outputs/textcnn_test_predictions.csv
outputs/fasttext_train_metrics.json
outputs/fasttext_test_metrics.json
checkpoints/textcnn_best.pt
checkpoints/fasttext_best.pt
```

## 报告与演示

- 报告模板：`reports/report.tex`
- 演示视频脚本：`reports/demo_script.md`

服务器跑完后，把 `outputs/*metrics.json` 的结果填入报告中的实验表格即可。

如果本地或服务器安装了 XeLaTeX，可以生成 PDF：

```bash
cd reports
xelatex report.tex
xelatex report.tex
```

## 可选服务器信息检查

如果 Docker 或 GPU 环境不确定，可以在服务器上运行：

```bash
nvidia-smi
docker --version
docker run --rm --gpus '"device=6"' nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```
