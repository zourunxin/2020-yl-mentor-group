import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import NLPUtils
from utils.FileUtil import csv_reader, write_csv, write_excel
from utils.CommonUtils import convert_label


def del_illegal_edge():
    reader = csv_reader('/Users/zourunxin/Mine/Seminar/20Data/1228/datasource_1228_rpm.csv')
    pkg_layer = {}
    layer_digit = {'核心': 0, '系统': 1, '应用': 2, '其它': 3}
    for line in reader:
        pkg_layer[line[0]] = layer_digit[convert_label(line[1], mode='layer')]

    edges = []
    reader = csv_reader('/Users/zourunxin/Mine/Seminar/20Data/1228/rpm/edges.csv')
    for line in reader:
        if pkg_layer[line[0]] < pkg_layer[line[1]]:
            continue
        edges.append([line[0], line[1]])

    write_csv('/Users/zourunxin/Mine/Seminar/20Data/1228/rpm/legal_edges.csv', ['out', 'in'], edges)
    return


def layer_feat_extractor():
    """
    使用 tfidf 获得 unique word
    """
    reader = csv_reader('/Users/zourunxin/Mine/Seminar/20Data/1228/datasource_1228_rpm.csv')
    layer_text = {}
    for line in reader:
        layer = convert_label(line[1], mode='layer')
        text = " ".join(set(s.lower() for s in NLPUtils.preprocess_text(line[2]).split(' ')))
        lText = layer_text.get(layer, "") + " " + text
        layer_text[layer] = lText
    # 建立词袋模型
    print("开始提取 TF-IDF 特征(每个包的预料为一个文档)")
    tv = TfidfVectorizer(max_features=300, max_df=1, stop_words="english")
    tfidf = tv.fit_transform(layer_text.values())
    write_excel('/Users/zourunxin/Mine/Seminar/20Data/1228/analy_src_data/distinct_word.xlsx', '300feat', ['word'], tv.get_feature_names())

    return tv.get_feature_names()


if __name__ == '__main__':
    print(layer_feat_extractor())