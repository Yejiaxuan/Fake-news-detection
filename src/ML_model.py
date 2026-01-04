import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

# 1) 读取数据
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/Atest.csv")

x_train = train["text"].astype(str)
y_train = train["label"]
x_test = test["text"].astype(str)
test_ids = test["id"]

# 2) 文本转 TF-IDF 特征 (使用jieba进行中文分词)
vectorizer = TfidfVectorizer(tokenizer=lambda x: list(jieba.cut(x)))
x_train_features = vectorizer.fit_transform(x_train)
x_test_features = vectorizer.transform(x_test)

# 3) 训练多个模型并分别预测

# 3.1 朴素贝叶斯
naive_bayes = MultinomialNB()
naive_bayes.fit(x_train_features, y_train)
nb_train_probabilities = naive_bayes.predict_proba(x_train_features)[:, 1]
nb_auc = roc_auc_score(y_train, nb_train_probabilities)
print(f"NB 训练 AUC: {nb_auc:.4f}")

# 获取假新闻（标签1）的概率
nb_probabilities = naive_bayes.predict_proba(x_test_features)[:, 1]
submission_nb = pd.DataFrame({"id": test_ids, "prob": nb_probabilities})
submission_nb.to_csv("./data/prediction_NB.csv", index=False)
print("保存: ./data/prediction_NB.csv")

# 3.2 逻辑回归
logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(x_train_features, y_train)
lr_train_probabilities = logistic_regression.predict_proba(x_train_features)[:, 1]
lr_auc = roc_auc_score(y_train, lr_train_probabilities)
print(f"LR 训练 AUC: {lr_auc:.4f}")

# 获取假新闻（标签1）的概率
lr_probabilities = logistic_regression.predict_proba(x_test_features)[:, 1]
submission_lr = pd.DataFrame({"id": test_ids, "prob": lr_probabilities})
submission_lr.to_csv("./data/prediction_LR.csv", index=False)
print("保存: ./data/prediction_LR.csv")

# 3.3 随机森林
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train_features, y_train)
rf_train_probabilities = random_forest.predict_proba(x_train_features)[:, 1]
rf_auc = roc_auc_score(y_train, rf_train_probabilities)
print(f"RF 训练 AUC: {rf_auc:.4f}")

# 获取假新闻（标签1）的概率
rf_probabilities = random_forest.predict_proba(x_test_features)[:, 1]
submission_rf = pd.DataFrame({"id": test_ids, "prob": rf_probabilities})
submission_rf.to_csv("./data/prediction_RF.csv", index=False)
print("保存: ./data/prediction_RF.csv")
