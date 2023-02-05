import sys
sys.path.append("../")
import pandas as pd
import numpy as np
import utils.CommonUtils as CommonUtils
import utils.NLPUtils as NLPUtils
import FeatureExtractor.extractors as Extractors
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from models.text_cnn import TextCNN
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline

def encode_onehot(label_num_map, labels):
    classes_dict = {label: np.identity(len(label_num_map))[label_num_map[label], :] for label in label_num_map}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


print("开始读取数据")
df_data = pd.read_csv('../output/datasource_0205_class.csv')


idx_name_map, name_idx_map = CommonUtils.get_idx_name_map(df_data["name"])
num_label_map, label_num_map = CommonUtils.get_num_label_map(df_data["label"])

onehot_labels = encode_onehot(label_num_map, list(df_data["label"]))
texts = list(df_data["text"].apply(lambda x: NLPUtils.preprocess_text(x)))
x_train, x_val, x_test, y_train, y_val, y_test, idx_train, idx_test = CommonUtils.get_sample_splits(texts, onehot_labels, sample_size=100)

# tf-idf 特征
tv = TfidfVectorizer(max_features=9999, stop_words="english")
x_train = tv.fit_transform(x_train)

# 卡方检验选择最佳特征
skb = SelectKBest(chi2, k=1000)#选择k个最佳特征
x_train = skb.fit_transform(x_train, np.argmax(y_train, axis=1))

rfc = RandomForestClassifier(random_state=0)
rfc = rfc.fit(x_train, np.argmax(y_train, axis=1))

pipe = make_pipeline(tv, skb, rfc)
preds = np.argmax(pipe.predict_proba(x_test), axis=1)
print(preds)

df_result = df_data[["name", "label", "summary", "description"]]
df_train = df_result.loc[idx_train]
df_result = df_result.loc[idx_test]
df_result["predict"] = [num_label_map[pred] for pred in preds]
df_error = df_result.loc[df_result["predict"] != df_result["label"]]
df_error = df_error.loc[:,["name", "label", "predict", "summary", "description"]]   #交换列的顺序
df_result =df_result.loc[:,["name", "label", "predict", "summary", "description"]] 
df_error.to_csv("RandomForest_result_error.csv", index=False)
df_result.to_csv("RandomForest_result.csv", index=False)
df_train.to_csv("RandomForest_trainset.csv", index=False)
report = classification_report(df_result["label"], df_result["predict"])
print(report)
