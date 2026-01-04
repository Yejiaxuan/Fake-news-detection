import pandas as pd
import numpy as np
import torch
import random
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
from gensim.models import KeyedVectors
import jieba
import warnings
import logging

warnings.filterwarnings("ignore")
jieba.setLogLevel(logging.INFO)
logging.getLogger('jieba').setLevel(logging.WARNING)

# 固定随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device} ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else f"设备: {device}")

# 1) 读取数据
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/Atest.csv")

x_train = train["text"].astype(str)
y_train = train["label"]
x_test = test["text"].astype(str)
test_ids = test["id"]


# 2) 文本预处理和编码
def cut(text):
    return [word.strip() for word in jieba.lcut(text) if word.strip()]


train_tokens = [cut(title) for title in x_train]
test_tokens = [cut(title) for title in x_test]

# 词向量
word2vec = KeyedVectors.load_word2vec_format("./Word2Vec/sgns.sogou.char", binary=False)
embedding_dim = word2vec.vector_size

# 词表
vocab = [word for word in set(word for seq in train_tokens for word in seq) if word in word2vec.key_to_index]

word2idx = {"<PAD>": 0, "<UNK>": 1}
word2idx.update({word: idx + 2 for idx, word in enumerate(vocab)})

# 计算最大长度
max_len = max(np.array([len(seq) for seq in train_tokens]))
print(f"最大序列长度: {max_len}")


# 编码 + padding
def encode(sequences):
    pad_idx, unk_idx = 0, 1
    encoded_array = np.zeros((len(sequences), max_len), np.int64)
    for idx, seq in enumerate(sequences):
        token_ids = [word2idx.get(word, unk_idx) for word in seq][:max_len]
        encoded_array[idx, :len(token_ids)] = token_ids
    return encoded_array


x_train_encoded = encode(train_tokens)
x_test_encoded = encode(test_tokens)

x_train_final, y_train_final = x_train_encoded, y_train

# 嵌入矩阵
embedding_matrix = np.zeros((len(word2idx), embedding_dim), np.float32)
vector_pool = []

for word, idx in word2idx.items():
    if idx < 2:
        continue
    if word in word2vec.key_to_index:
        embedding_matrix[idx] = word2vec[word]
        vector_pool.append(word2vec[word])

if vector_pool:
    embedding_matrix[1] = np.mean(vector_pool, axis=0)

# 3) 训练参数
batch_size = 32
hidden_dim = 128
num_classes = 2
epochs = 10
learning_rate = 5e-4

# 模型定义
class LSTMCls(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, embedding_matrix, num_layers=2, dropout_prob=0.3,
                 freeze=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = not freeze
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout_prob, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.drop = nn.Dropout(dropout_prob)

    def forward(self, input_ids):
        mask = (input_ids != 0).float().unsqueeze(-1)
        lstm_output, _ = self.lstm(self.embedding(input_ids))
        lstm_output = lstm_output * mask
        pooled_output = lstm_output.sum(1) / mask.sum(1).clamp_min(1)
        return self.fc(self.drop(pooled_output))


# 4) 数据加载器和模型初始化
train_loader = DataLoader(
    TensorDataset(torch.tensor(x_train_final), torch.tensor(y_train_final.values)),
    batch_size=batch_size, shuffle=True, pin_memory=True
)

model = LSTMCls(len(word2idx), embedding_dim, hidden_dim, num_classes, embedding_matrix).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam([param for param in model.parameters() if param.requires_grad], lr=learning_rate)

for epoch in range(epochs):
    model.train()
    probabilities, true_labels, losses = [], [], []

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        # 获取假新闻（标签1）的概率
        probs = torch.softmax(logits, dim=-1)[:, 1]
        probabilities.extend(probs.detach().cpu().numpy())
        true_labels.extend(batch_y.cpu().numpy())

    train_auc = roc_auc_score(true_labels, probabilities)
    print(f'Epoch {epoch + 1}/{epochs} - Training AUC: {train_auc:.4f}')

# 5) 使用最后一轮模型预测测试集并保存
test_loader = DataLoader(
    TensorDataset(torch.tensor(x_test_encoded)),
    batch_size=batch_size, pin_memory=True
)

model.eval()
probabilities = []
with torch.no_grad():
    for (batch_x,) in test_loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        # 获取假新闻（标签1）的概率
        probs = torch.softmax(logits, dim=-1)[:, 1]
        probabilities.extend(probs.cpu().numpy())

probabilities = np.array(probabilities)
submission = pd.DataFrame({"id": test_ids, "prob": probabilities})
submission.to_csv("./data/prediction_LSTM.csv", index=False)
print("保存: ./data/prediction_LSTM.csv")
