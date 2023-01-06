import sys
sys.path.append("../")
import utils.textrank as textrank
import pandas as pd
import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import extractors as ext

data_dir = "../output/datasource_1228.csv"

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
text_list = [textrank.preprocess_text(d['text']) if isinstance(d['text'], str) else "" for d in pkg_data_list]

# lazyload
label_feats = []
key_feats = []
bow_feats = []
tfidf_feats = []

# 配置特征提取器 ["name", "keyword", "bow", "tfidf"]
extractors = ["name", "keyword", "tfidf"]
feats = []

for extractor in extractors:    
    if extractor == "name":
        feats.append(ext.name_feat_extractor(name_list))
    elif extractor == "keyword":
        feats.append(ext.keyword_feat_extractor(text_list))
    elif extractor == "bow":
        feats.append(ext.bow_feat_extractor(text_list))
    elif extractor == "tfidf":
        feats.append(ext.tfidf_feat_extractor(text_list))
    else:
        print("WRANING: no extractor named {}".format(extractor))

with open("../output/name_label_feature.csv", "w", newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    lines = []
    if len(feats) == 0:
        print("WARNING: no extractor is given")
    for i, pkg in enumerate(name_list):
        line = [name_list[i], label_list[i]]
        for f in feats:
            line = line + list(f[i])
        if line.count(0) < len(line) - 2:
            lines.append(line)    
    writer.writerows(lines)
