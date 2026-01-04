# 南开大学《Python语言程序设计》- 虚假新闻检测

> **项目定位说明**：本项目实现效果**尚可**，但仍有较大调优空间（超参数调优、数据增强、模型融合等）。**核心优势在于代码结构工整精简、逻辑清晰**，适合学习参考或作为baseline快速迭代。

---

## 目录说明

```
Fake-news-detection/
├── src/                     # 源代码目录
│   ├── ML_model.py          # 经典机器学习模型（TF-IDF + NB/LR/RF）
│   ├── LSTM_model.py        # 基于预训练词向量 + BiLSTM 的分类器
│   ├── BERT_model.py        # hfl/chinese-roberta-wwm-ext 序列分类器
│   ├── data/                # 数据目录
│   │   ├── train.csv        # 训练集
│   │   ├── Atest.csv        # 测试集
│   │   └── prediction_*.csv # 各模型预测结果（运行后生成）
│   └── Word2Vec/            # 预训练词向量（需自行下载）
│       └── sgns.sogou.char  # 中文预训练词向量（LSTM 必需）
├── report/                  # 实验报告
│   └── report.pdf
├── ppt/                     # 汇报PPT
│   └── ppt.pptx
└── README.md
```

## 环境依赖

- Python 3.10

安装命令：

```
pip install pandas numpy scikit-learn gensim jieba transformers tqdm

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## 预训练词向量（LSTM 必需）

- 项目使用"中文词向量"发布的预训练向量（Sogou News 搜狗新闻的 *Word + Character* 300d向量）。
- 主页：<https://github.com/Embedding/Chinese-Word-Vectors>
- 百度网盘直链：<https://pan.baidu.com/s/1pUqyn7mnPcUmzxT64gGpSw>

下载后将相应的词向量文件解压/重命名为：

```
src/Word2Vec/sgns.sogou.char
```

确保 `LSTM_model.py` 能在上述路径加载到该文件。

## 运行三个模型

- 经典机器学习（TF‑IDF + NB/LR/RF，分别训练三个模型）

```bash
cd src
python ML_model.py
```

输出：`data/prediction_NB.csv`, `data/prediction_LR.csv`, `data/prediction_RF.csv`

- LSTM 模型（需要 `Word2Vec/sgns.sogou.char`）

```bash
cd src
python LSTM_model.py
```

输出：`data/prediction_LSTM.csv`

- BERT 模型（会自动下载 `hfl/chinese-roberta-wwm-ext`）

```bash
cd src
python BERT_model.py
```

输出：`data/prediction_BERT.csv`
