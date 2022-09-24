import sys
sys.path.append("../")
import utils.textrank as textrank
import pandas as pd
import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

data_dir = "../data_resources/data_0915.csv"

# load data
print("Loading data from {}".format(data_dir))
src_data = pd.read_csv(data_dir)

src_list = src_data.values.tolist()

pkg_data_list = []

for data in src_list:
    pkg_data = {"pkg": data[1], "label": data[0], "text":str(data[2]) + str(data[3])}
    pkg_data_list.append(pkg_data)

print("Loading data finished, load {} records".format(len(pkg_data_list)))

name_list = [d['pkg'] for d in pkg_data_list]
label_list = [d['label'] if isinstance(d['label'], str) else "none" for d in pkg_data_list]
text_list = [d['text'] if isinstance(d['label'], str) else "" for d in pkg_data_list]

# 建立词袋模型
cv = CountVectorizer(min_df= 5, binary=True)
cv_matrix = cv.fit_transform(text_list)
word_vecs = cv_matrix.toarray()

# 建立关键词特征
key_list = []
key_words = ["library", "language", "service", "tool", "util"]
for text in text_list:
    keys = []
    for key in key_words:
        keys.append(0 if text.find(key) == -1 else 1)
    key_list.append(keys)

with open("../output/name_label_feature.csv", "w", newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    lines = []
    for i, pkg in enumerate(name_list):
        x_str = ' '.join(str(x) for x in word_vecs[i])
        key_str = ' '.join(str(x) for x in key_list[i])
        lines.append("{} {} {} {}".format(pkg, label_list[i], x_str, key_str))
    lines = [line.split(" ") for line in lines]
    writer.writerows(lines)

count_list = [list(vec).count(1) for vec in word_vecs]

print("词袋模型建立完毕，词袋大小为: {}, 单词出现平均占比: {}".format(len(word_vecs[0]), np.mean(count_list) / len(word_vecs[0])))