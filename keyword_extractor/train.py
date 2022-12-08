import sys
sys.path.append("../")
import utils.textrank as textrank
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import silhouette_score, silhouette_samples
import pandas as pd
import tensorflow_hub as hub
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer

def load_data(data_dir):
    print("Loading data from {}".format(data_dir))
    src_data = pd.read_csv(data_dir, header=None)

    src_list = src_data.values.tolist()

    pkg_data_list = []

    for data in src_list:
        pkg_data = {"pkg": data[3], "label": data[1] if data[1] == "编程语言" else data[2], "text":str(data[4]) + str(data[5]) + str(data[6]) + str(data[7])}
        pkg_data_list.append(pkg_data)

    print("Loading data finished, load {} records".format(len(pkg_data_list)))


    return pkg_data_list


def build_dataset(data_list, n_keywords=8, method="bert"):

    def concat_keyword(key_list):
        res = ""
        for k in key_list:
            res = res + k[0] + " "
        return res
    print("Start extracting keywords")

    name_list = [d['pkg'] for d in data_list]
    label_list = [d['label'] if isinstance(d['label'], str) else "none" for d in data_list]
    text_list = [textrank.preprocess_text(d['text']) if isinstance(d['text'], str) else "" for d in data_list]

    key_list = []
    key_kv = []

    if method == "none":
        key_list = text_list
    else:
        # 这里使用循环以防 oom
        print("start extracting keywords")
        for i, text in enumerate(text_list):
            if (method == "textrank"):
                keys = textrank.extract_keyword(text, n_keywords)
            elif (method == "bert"):
                keys = kw_model.extract_keywords(text,
                                            keyphrase_ngram_range=(1, 1),
                                            stop_words='english',
                                            use_mmr=True,
                                            diversity=0.3,
                                            top_n=9999)

            key_list.append(keys)
        
            kv_map = {}
            for k in keys:
                kv_map[k[0]] = k[1]
            key_kv.append(kv_map)
            if (i % 100 == 0):
                print("extracting {}/{}".format(i, len(text_list)))
            
            key_list = [concat_keyword(d) for d in key_list]

    print("Extracting finished")

    return name_list, label_list, key_list, key_kv


data_dir = "../data_resources/data_1008.csv"

# 超参数

n_clusters = 11
n_keyword = 8
n_pca_output_dementions = 64

# compile model
sentence2vec_model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
# transformer based
kw_model = KeyBERT()
# universal sentence encoder
sentence2vec_model = hub.load(sentence2vec_model_url)
pca = PCA(n_components=n_pca_output_dementions)


# load data from csv
datas = load_data(data_dir)

# idx_name: 下标 -> 包名, X: 下标 -> 关键词组
idx2name, label_list, feature, key_kv = build_dataset(datas, n_keyword, method="none")


# 建立词袋模型
# cv = CountVectorizer(min_df= 1, binary=True, stop_words="english")
# cv_matrix = cv.fit_transform(feature)
# X = cv_matrix.toarray()
# bag = cv.get_feature_names_out()
# print("词袋大小： {}".format(len(bag)))

# print(bag[0:100])
# print(X[0][0:100])

# feat = []

# for i, line in enumerate(X):
#     f = []
#     for j, letter in enumerate(line):
#         if letter > 0:
#             f.append(key_kv[i][bag[j]])
#         else:
#             f.append(0)
#     feat.append(f)
# X = feat
# print(X[0][0:100])

# sentence embedding -> 512d vector
X = sentence2vec_model(feature)
X = X.numpy()
# 512d -> 16d
# X = pca.fit_transform(X)


with open("../output/name_label_feature.csv", "w", newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    lines = []
    for i, pkg in enumerate(idx2name):
        x_str = ' '.join(str(x) for x in X[i])
        lines.append("{} {} {}".format(pkg, label_list[i], x_str))
    lines = [line.split(" ") for line in lines]
    writer.writerows(lines)

print("数据处理完毕")