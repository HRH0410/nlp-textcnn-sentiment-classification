# 1-2 分钟程序演示视频脚本

## 0:00-0:15 项目介绍

本项目实现 Rotten Tomatoes 电影评论句子的五分类情感分类任务，类别包括 very negative、negative、neutral、positive、very positive。主模型采用 TextCNN，并加入 FastText baseline 作为对比。

## 0:15-0:35 代码结构展示

展示项目目录：

- `preprocessed_file/`：预处理后的训练、验证、测试数据和词表
- `src/data.py`：数据读取、词表加载、padding
- `src/models.py`：TextCNN 和 FastText 模型
- `src/train.py`：训练与验证
- `src/evaluate.py`：测试集评估与预测结果导出
- `configs/`：实验超参数
- `Dockerfile`：容器环境

## 0:35-0:55 模型方法说明

TextCNN 首先将 token id 映射为词向量，然后使用多个不同窗口大小的一维卷积提取 n-gram 局部特征，再经过 max pooling 得到句子级表示，最后用全连接层输出五类情感概率。

## 0:55-1:20 Docker 运行展示

展示服务器运行命令：

```bash
docker build -t nlp-textcnn-sentiment .
docker run --rm -it --gpus '"device=6"' \
  -v "$PWD/outputs:/workspace/outputs" \
  -v "$PWD/checkpoints:/workspace/checkpoints" \
  nlp-textcnn-sentiment bash scripts/run_all.sh
```

说明所有实验在 Docker 容器中运行，并指定使用 GPU 6。

## 1:20-1:45 实验结果展示

展示 `outputs/textcnn_test_metrics.json` 和 `outputs/fasttext_test_metrics.json` 中的 accuracy、macro F1 和混淆矩阵。说明 TextCNN 与 baseline 的对比结果，以及 TextCNN 在局部短语特征建模上的优势。

## 1:45-2:00 总结

总结项目完成了数据读取、模型训练、测试评估、结果保存、Docker 部署和报告整理，满足课程作业对代码、实验和文档的要求。
