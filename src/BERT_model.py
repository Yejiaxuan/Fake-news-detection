import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

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

# 2) BERT文本编码
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')

# 计算最大长度
all_texts = list(x_train) + list(x_test)
max_len = max(len(tokenizer.tokenize(text)) + 2 for text in all_texts)
print(f"最大序列长度: {max_len}")


# 编码文本
def encode(texts):
    return tokenizer(texts.tolist(), add_special_tokens=True, max_length=max_len,
                     padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')


train_encoded = encode(x_train)
test_encoded = encode(x_test)
train_ids, train_mask, train_labels = train_encoded['input_ids'], train_encoded['attention_mask'], y_train

# 3) 训练参数
batch_size = 32
epochs = 10
learning_rate = 1e-5
num_labels = 2

# 4) 数据加载器
train_data = TensorDataset(train_ids, train_mask, torch.tensor(train_labels.values, dtype=torch.long))
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# 模型训练
model = BertForSequenceClassification.from_pretrained('hfl/chinese-roberta-wwm-ext', num_labels=num_labels).to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

for epoch in range(epochs):
    # 训练
    model.train()
    train_loss, train_probabilities, train_true_labels = 0, [], []
    for batch in tqdm(train_loader, desc=f"Train {epoch + 1}/{epochs}"):
        input_ids, attention_mask, labels = [item.to(device) for item in batch]
        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # 获取假新闻（标签1）的概率
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
        train_probabilities.extend(probs.detach().cpu().numpy())
        train_true_labels.extend(labels.cpu().numpy())
    train_auc = roc_auc_score(train_true_labels, train_probabilities)
    print(f'Epoch {epoch + 1}/{epochs} - Training AUC: {train_auc:.4f}')

# 5) 使用最后一轮模型预测测试集并保存
test_data = TensorDataset(test_encoded['input_ids'], test_encoded['attention_mask'])
test_loader = DataLoader(test_data, batch_size=batch_size)

model.eval()
probabilities = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask = [item.to(device) for item in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # 获取假新闻（标签1）的概率
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
        probabilities.extend(probs.cpu().numpy())

probabilities = np.array(probabilities)
submission = pd.DataFrame({"id": test_ids, "prob": probabilities})
submission.to_csv("./data/prediction_BERT.csv", index=False)
print("保存: ./data/prediction_BERT.csv")
