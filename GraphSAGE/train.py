import pdb
import sys
sys.path.append("../")
import tensorflow as tf
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.python.keras.optimizers import adam_v2
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import utils.CommonUtils as CommonUtils
import utils.NLPUtils as NLPUtils
import utils.ClassificationReportAVG as ClassificationReportAVG
import FeatureExtractor.extractors as Extractors
from models.GraphSAGEModel import GraphSAGE

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=str)
parser.add_argument('--strategy', type=str, default='all')
parser.add_argument('--backUp', type=str)
args = parser.parse_args()

def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    # adj = normalize_adj(adj, symmetric)
    return adj.todense()

def encode_onehot(label_num_map, labels):
    classes_dict = {label: np.identity(len(label_num_map))[label_num_map[label], :] for label in label_num_map}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)

def get_splits(y, strategy="all", sample_size = 200):
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
            idx_train = idx_train + s[0:int(len(s)*0.7)]
            idx_val = idx_val + s[int(len(s) * 0.7):]
            idx_test = idx_test + s[int(len(s) * 0.7):]
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

def sample_neighs(G, nodes, sample_num=None, self_loop=False, shuffle=True):  # 抽样邻居节点
    # np.random.seed(0)
    _sample = np.random.choice
    neighs = [list(G[int(node)]) for node in nodes]  # nodes里每个节点的邻居
    if sample_num:
        if self_loop:
            _sample_num = sample_num - 1
        samp_neighs = [
            (list(_sample(neigh, _sample_num, replace=False)) if len(neigh) >= _sample_num else list(
                _sample(neigh, _sample_num, replace=True))) if len(neigh) > 0 else list(np.array([])) for neigh in neighs]  # 采样邻居
        if self_loop:
            samp_neighs = [
                samp_neigh + list([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]  # gcn邻居要加上自己
        samp_neighs = [
                samp_neigh + [samp_neigh[-1]] * (sample_num - len(samp_neigh)) for samp_neigh in samp_neighs]   # 补齐邻居个数
        if shuffle:
            samp_neighs = [list(np.random.permutation(x)) for x in samp_neighs]
    else:
        samp_neighs = neighs
    return np.asarray(samp_neighs), np.asarray(list(map(len, samp_neighs)))

print("开始读取数据")
df_data = pd.read_csv('../output/datasource_1228.csv')
# df_data = df_data.sample(3000)
df_edges = pd.read_csv('../output/edges.csv')
# df_edges = df_edges.loc[lambda df : (df['out'].isin(df_data["name"])) & (df['in'].isin(df_data["name"]))]
idx_name_map, name_idx_map = CommonUtils.get_idx_name_map(df_data["name"])
num_label_map, label_num_map = CommonUtils.get_num_label_map(df_data["label"])
names = list(df_data["name"])
onehot_labels = encode_onehot(label_num_map, list(df_data["label"]))
texts = list(df_data["text"].apply(lambda x: NLPUtils.preprocess_text(x)))
name_features = Extractors.name_feat_extractor(names)
keyword_features = Extractors.keyword_feat_extractor(texts)
# tfidf_features = Extractors.tfidf_class_feat_extractor(list(df_data["label"]), texts)
tfidf_features = Extractors.tfidf_feat_extractor(texts, onehot_labels, feature_num=1000)
# bow_features = Extractors.bow_feat_extractor(texts)
layer_features = Extractors.unique_feat_extractor(texts, max_features=3000, speci_layer='all')
features = np.hstack((layer_features, tfidf_features))
# 构建邻接矩阵
adj = sp.coo_matrix((np.ones(len(df_edges)),
                    (df_edges['out'].apply(lambda x: name_idx_map[x]), df_edges['in'].apply(lambda x: name_idx_map[x]))),
                    shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)

print('数据集含有 {} 个结点, {} 条边, {} 个特征.'.format(adj.shape[0], len(df_edges), features.shape[1]))

reports = []
for _ in range(5):
    #分割数据集
    y_train, y_val, y_test, train_mask, val_mask, test_mask = get_splits(onehot_labels, strategy=args.strategy, sample_size=200)

    # 使用 networkx 从邻接矩阵创建图
    G = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())

    A = preprocess_adj(adj)

    indexs = np.arange(A.shape[0])
    # 两个隐藏层，定义每个层抽样邻居的个数
    neigh_number = [5, 10]
    neigh_maxlen = []

    model_input = [features, np.asarray(indexs, dtype=np.int32)]

    for num in neigh_number:
        sample_neigh, sample_neigh_len = sample_neighs(G, indexs, num, self_loop=True)
        model_input.extend([sample_neigh])
        neigh_maxlen.append(max(sample_neigh_len))


    # train
    model = GraphSAGE(feature_dim=features.shape[1],
                      neighbor_num=neigh_maxlen,
                      n_hidden=16,
                      n_classes=y_train.shape[1],
                      use_bias=True,
                      activation=tf.nn.relu,
                      aggregator_type='mean',
                      dropout_rate=0.5, l2_reg=2.5e-4)

    model.compile(adam_v2.Adam(0.0001), 'categorical_crossentropy', weighted_metrics=['categorical_crossentropy', 'acc'])

    val_data = (model_input, y_val, val_mask)

    # class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(np.argmax(onehot_labels, axis=1)), y=np.argmax(onehot_labels, axis=1))
    # class_weight = {i: class_weight[i] for i in range(len(class_weight))}
    print("start training")
    history = model.fit(model_input, y_train, sample_weight=np.array(train_mask).astype(np.int32), validation_data=val_data, batch_size=A.shape[0], epochs=800,
                        shuffle=False, verbose=1)

    eval_results = model.evaluate(model_input, y_test, sample_weight=test_mask, batch_size=A.shape[0])

    # predict
    predictions = model.predict(model_input, batch_size = A.shape[0])
    predictions = np.argmax(predictions, axis = 1)

    print(predictions)

    df_result = df_data[["name", "label", "summary", "description"]]
    df_result["predict"] = [num_label_map[pred] for pred in predictions]
    df_result = df_result[df_result["name"].map(lambda x: test_mask[name_idx_map[x]]) == 1]     # 取出测试集
    df_error = df_result.loc[df_result["predict"] != df_result["label"]]
    df_error = df_error.loc[:,["name", "label", "predict", "summary", "description"]]
    df_error.to_csv("../output/zrx/GraphSAGE_result{}{}.csv".format(args.epoch, args.strategy), index=False)
    df_result.to_csv("../output/zrx/all_GraphSAGE_result{}{}.csv".format(args.epoch, args.strategy), index=False)
    report = classification_report(df_result["label"], df_result["predict"], digits=3)
    print(report)
    ClassificationReportAVG.save_cr(report, './log-{}.txt'.format(args.backUp))
    reports.append(report)
    # CommonUtils.show_learning_curves(history)

ClassificationReportAVG.cr_avg(reports)
