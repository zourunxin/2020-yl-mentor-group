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

def load_data(data_dir):
    print("Loading data from {}".format(data_dir))
    src_data = pd.read_csv(data_dir)

    src_list = src_data.values.tolist()

    pkg_data_list = []

    for data in src_list:
        pkg_data = {"pkg": data[1], "summary": data[2],
                    "description": data[3], "label": data[0]}
        pkg_data_list.append(pkg_data)

    
    print("Loading data finished, load {} records".format(len(pkg_data_list)))

    return pkg_data_list


def build_dataset(data_list, n_summary_keyword=3, n_description_keyword=5, method="bert"):
    print("Start extracting keywords")

    name_list = [d['pkg'] for d in data_list]
    summary_list = [d['summary'] if isinstance(
        d['summary'], str) else "" for d in data_list]
    description_list = [d['description'] if isinstance(
        d['description'], str) else "" for d in data_list]
    label_list = [d['label'] if isinstance(
        d['label'], str) else "none" for d in data_list]

    summary_key_list = []
    description_key_list = []
    summary_textrank_key_list = []
    description_textrank_key_list = []

    # 这里使用循环以防 oom
    print("start extracting summary")
    for i, summary in enumerate(summary_list):
        keys = kw_model.extract_keywords(summary,
                                         keyphrase_ngram_range=(1, 1),
                                         stop_words='english',
                                         diversity=0.3,
                                         use_mmr=True,
                                         top_n=n_summary_keyword)
        summary_key_list.append(keys)
        summary_textrank_key_list.append(
            textrank.extract_keyword(summary, n_summary_keyword))
        if (i % 100 == 0):
            print("extracting {}/{}".format(i, len(summary_list)))

    print("start extracting description")
    for i, description in enumerate(description_list):
        keys = kw_model.extract_keywords(description,
                                         keyphrase_ngram_range=(1, 1),
                                         stop_words='english',
                                         diversity=0.3,
                                         use_mmr=True,
                                         top_n=n_description_keyword)
        description_key_list.append(keys)
        description_textrank_key_list.append(
            textrank.extract_keyword(description, n_description_keyword))
        if (i % 100 == 0):
            print("extracting {}/{}".format(i, len(description_list)))

    def concat_keyword(key_list):
        res = ""
        for k in key_list:
            res = res + k[0] + " "
        return res

    summary_key_list = [concat_keyword(d) for d in summary_key_list]
    description_key_list = [concat_keyword(d) for d in description_key_list]
    summary_textrank_key_list = [concat_keyword(
        d) for d in summary_textrank_key_list]
    description_textrank_key_list = [concat_keyword(
        d) for d in description_textrank_key_list]

    final_list = []

    if (method == "textrank"):
        for s, d in zip(summary_key_list, description_key_list):
            final_list.append(s + d)
    elif (method == "bert"):
        for s, d in zip(summary_key_list, description_key_list):
            final_list.append(s + d)

    print("Extracting finished")

    return name_list, label_list, final_list, summary_key_list, description_key_list, summary_textrank_key_list, description_textrank_key_list, summary_list, description_list


data_dir = "../data_resources/data_0915.csv"

# 超参数

n_clusters = 11
n_summary_keyword = 3
n_description_keyword = 5
n_pca_output_dementions = 16

# compile model
sentence2vec_model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
# transformer based
kw_model = KeyBERT()
# universal sentence encoder
sentence2vec_model = hub.load(sentence2vec_model_url)
pca = PCA(n_components=n_pca_output_dementions)

clf = KMeans(n_clusters=n_clusters)

# load data from csv
datas = load_data(data_dir)

# idx_name: 下标 -> 包名, X: 下标 -> 关键词组
idx2name, label_list, feature, summary_key_list, description_key_list, summary_textrank_key_list, description_textrank_key_list, summary_list, description_list = build_dataset(
    datas, n_summary_keyword, n_description_keyword)

# sentence embedding -> 512d vector
X = sentence2vec_model(feature)
# 512d -> 16d
X = pca.fit_transform(X)

# 下面代码通过轮廓系数评估分多少类
# n_clusters = [8 ,9, 10, 11]

# for i in range(len(n_clusters)):
#     # 实例化k-means分类器
#     clf = KMeans(n_clusters= n_clusters[i])
#     y_predict = clf.fit_predict(X)

#     s = silhouette_score(X, y_predict)
#     print("When cluster= {}\nThe silhouette_score= {}".format(n_clusters[i], s))


clf.fit(X)

labels = clf.labels_

result = []
for i, pkg in enumerate(idx2name):
    result.append([pkg, summary_list[i], description_list[i], summary_key_list[i], description_key_list[i],
                  summary_textrank_key_list[i], description_textrank_key_list[i], labels[i], label_list[i]])

result.sort(key=lambda x: (x[7], x[8]))

with open("../output/result.csv", "w", newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["pkg", "summary", "description", "summary_bert_key", "description_bert_key",
                    "summary_textrank_key", "description_textrank_key", "predict_label", "real_label"])
    writer.writerows(result)

with open("../output/feature.csv", "w", newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    lines = []
    for i, pkg in enumerate(idx2name):
        x_str = ' '.join(str(x) for x in X[i])
        lines.append("{} {} {}".format(pkg, label_list[i], x_str))
    lines = [line.split(" ") for line in lines]
    writer.writerows(lines)


# 可视化

vis_pca = PCA(n_components=2)
x_vis = vis_pca.fit_transform(X)
x_clusters = []
real_labels = list(set(label_list))

# # 聚类前
# plt.figure()
# plt.title("before classification")
# for l in range(real_labels):
#     x_cluster = np.array(x_vis)[[i for i, x in enumerate(label_list) if x == l]]
#     color = (np.random.random(), np.random.random(), np.random.random())
#     plt.scatter(x_cluster[:, 0], x_cluster[:, 1], c = color)
# plt.xlabel('feature1')
# plt.ylabel('feature2')
# plt.legend()

# 聚类后
plt.figure()
plt.title("after classification")
for i in range(n_clusters):
    x_cluster = x_vis[labels == i]
    color = (np.random.random(), np.random.random(), np.random.random())
    plt.scatter(x_cluster[:, 0], x_cluster[:, 1], c=color)
plt.xlabel('feature1')
plt.ylabel('feature2')
plt.legend()


plt.show()
