import sys
sys.path.append("../")
import utils.textrank as textrank
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import classification_report


data_dir = "../output/datasource_1128.csv"

# load data
print("Loading data from {}".format(data_dir))
src_data = pd.read_csv(data_dir)

src_list = src_data.values.tolist()

pkg_data_list = []

for data in src_list:
    pkg_data = {"pkg": data[0], "label": data[1], "text":data[2]}
    pkg_data_list.append(pkg_data)

print("Loading data finished, load {} records".format(len(pkg_data_list)))

name_list = [d['pkg'] for d in pkg_data_list]
label_list = [d['label'] if isinstance(d['label'], str) else "none" for d in pkg_data_list]
text_list = [d['text'] if isinstance(d['text'], str) else "" for d in pkg_data_list]

_label_map = {
        "基础环境": "基础环境",
        "核心库": "核心库",
        "核心工具": "核心工具",
        "系统服务": "系统服务",
        "系统库": "系统库",
        "系统工具": "系统工具",
        "应用服务": "应用服务",
        "应用库": "应用库",
        "应用工具": "应用工具",
}

def _convert_label(x):
    if x in _label_map:
        return _label_map[x]
    else:
        return "其它"

for i, label in enumerate(label_list):
    label_list[i] = _convert_label(label)

# 建立词袋模型
cv = CountVectorizer(binary=False)
cv_matrix = cv.fit_transform(text_list)
word_vecs = cv_matrix.toarray()
bag = cv.get_feature_names_out()
print("词袋大小： {}".format(len(bag)))

label_vec = {}

# 计算类向量
for i, vec in enumerate(word_vecs):
    label = label_list[i]
    if label in label_vec:
        for j, count in enumerate(label_vec[label]):
            label_vec[label][j] += vec[j]
    else:
        label_vec[label] = vec


scaler = MinMaxScaler()

for label in label_vec:
    _vec = label_vec[label]
    _vec = _vec / (max(_vec) - min(_vec))
    label_vec[label] = _vec

print("label_vec 构建完毕")
print(label_vec)
predict_list = [""] * len(label_list)

print("word_vecs 构建完毕")
for i, vec in enumerate(word_vecs):
    divided = max(vec) - min(vec)
    if divided == 0:
        print("you 0")
    vec = vec / divided if divided != 0 else vec
    dist = 10000
    for label in label_vec:
        _dist = np.linalg.norm(vec - label_vec[label])
        if _dist < dist:
            dist = _dist
            predict_list[i] = label
    if i % 500 == 0:
        print("计算完成{}".format(i))


t = classification_report(label_list, predict_list)

print(t)