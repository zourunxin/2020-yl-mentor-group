import pdb

import sys
sys.path.append("../")
from sklearn.ensemble import RandomForestClassifier
from GraphSAGE_embedding.main import graphsage
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import utils.CommonUtils as CommonUtils
import FeatureExtractor.extractors as extractors
from utils import NLPUtils, ClassificationReportAVG
from sklearn.metrics import classification_report


def encode_onehot(label_num_map, labels):
    classes_dict = {label: np.identity(len(label_num_map))[label_num_map[label], :] for label in label_num_map}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


def get_splits(y, strategy="all", sample_size=200):
    '''
    分割数据集
    '''
    idx_set = []
    for i in range(y.shape[1]):
        idx_set.append([])

    for i, label in enumerate(y):
        label = np.argmax(label)
        idx_set[label].append(i)

    idx_train = []
    idx_val = []
    idx_test = []

    for s in idx_set:
        # np.random.seed(1234)
        np.random.shuffle(s)
        if strategy == "all":
            idx_train = idx_train + s[0:int(len(s) * 0.7)]
            idx_val = idx_val + s[int(len(s) * 0.7):]
            idx_test = idx_test + s[int(len(s) * 0.8):]
            # idx_train = idx_train + s[int(len(s)*0.5):]
            # idx_val = idx_val + s[int(len(s) * 0.3):int(len(s) * 0.5)]
            # idx_test = idx_test + s[:int(len(s) * 0.5)]
        elif strategy == "sample":
            idx_train = idx_train + s[0:sample_size]
            idx_val = idx_val + s[int(len(s) * 0.8):]
            idx_test = idx_test + s[sample_size:]

    print("样本数量 train: {}, val: {}, test: {}".format(len(idx_train), len(idx_val), len(idx_test)))

    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    val_mask = sample_mask(idx_val, y.shape[0])
    test_mask = sample_mask(idx_test, y.shape[0])

    return y_train, y_val, y_test, train_mask, val_mask, test_mask

reports = []
for _ in range(5):
    df_data = pd.read_csv('../output/datasource_1228_without_isolate_node.csv')
    num_label_map, label_num_map = CommonUtils.get_num_label_map(df_data["label"])
    onehot_labels = encode_onehot(label_num_map, list(df_data["label"]))
    y_train, y_val, y_test, train_mask, val_mask, test_mask = get_splits(onehot_labels)
    y_train = [np.argmax(y) for y in y_train[train_mask]]
    y_test = [np.argmax(y) for y in y_test[test_mask]]


    # texts = list(df_data["text"].apply(lambda x: NLPUtils.preprocess_text(x)))
    # embedding = extractors.tfidf_feat_extractor(texts)
    sage = graphsage()
    embedding = sage.exec()
    clf = RandomForestClassifier()
    clf.fit(embedding[train_mask], y_train)
    predict_results = clf.predict(embedding[test_mask])
    report = classification_report(y_test, predict_results, digits=3)
    print(report)
    ClassificationReportAVG.save_cr(report, './log-{}.txt'.format('mean'))
    reports.append(report)

ClassificationReportAVG.cr_avg(reports)
