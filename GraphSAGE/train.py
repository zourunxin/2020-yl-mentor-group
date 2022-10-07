from types import NoneType
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp
import pandas as pd

from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.models import  Model
from tensorflow.python.keras import backend
from tensorflow.python.keras.layers import Input, Dense, Dropout, Layer, LSTM
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.initializers import glorot_uniform, Zeros
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.metrics import Metric
import csv

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

import os

# =================================== load data =====================================
f = open('../output/idx_label.csv', 'r', encoding='utf-8-sig')
labels = f.readlines()[1:]
labels = [l.strip().split(",") for l in labels]
labels.sort(key=lambda x: int(x[0]))
f.close()

f = open('../output/feature.csv', 'r', encoding='utf-8-sig')
features = f.readlines()
features = [l.strip().split(",") for l in features]
f.close()

final_features = []
for i, feature in enumerate(features):
    final_features.append(labels[i] + features[i])

with open("../output/GAT_features.csv", "w", newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(final_features)

df_features = pd.read_csv('../output/GAT_features.csv', header=None)
df_edges = pd.read_csv("../output/edges.csv")
df_classes =  pd.read_csv("../output/idx_label.csv")

# =============================== utils =================================

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def convert_symmetric(X, sparse=True):
    if sparse:
        X += X.T - sp.diags(X.diagonal())
    else:
        X += X.T - np.diag(X.diagonal())
    return X


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    # adj = normalize_adj(adj, symmetric)
    return adj.todense()


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm

def cal_acc(predict: dict, actual: dict):
    """
    计算模型整体的准确率，分类正确的个数占总分类个数的比例
    :param predict: <pkg: str, pre_label: str>
    :param actual: <pkg: str, act_label: str>
    :return: accuracy
    """
    assert len(predict) == len(actual)
    ele_sum = len(predict)
    acc_sum = 0
    for k, v in predict.items():
        actual_label = actual[k]
        if actual_label == v:
            acc_sum += 1
    return acc_sum / ele_sum


def statistic(map1: dict, map2: dict):
    """
    map1 和 map2 需有相同 key，计算在 v1 下，各个 v2 的个数。（用于计算精确率、召回率时使用）
    :param map1: <pkg: str, v1: str>
    :param map2: <pkg: str, v2: str>
    :return: <v1 : <v2_1 : cnt1, v2_2 : cnt2, ...,> ...,>
    """
    res = dict()
    for k, v1 in map1.items():
        label_cnt = res.get(v1, dict())
        v2 = map2[k]
        cnt = label_cnt.get(v2, 0) + 1
        label_cnt[v2] = cnt
        res[v1] = label_cnt
    return res


def cal_precision(statistic_dict: dict):
    """
    计算每个聚类标签的精确率：成功被分出来的真实 label 数量占该聚类标签总 label 数量的比例
    :param statistic_dict: <predict_label(str) : <actual_label1(str) : cnt1(int), actual_label2(str) : cnt2(int), ...>, ...>
    :return: <predict_label1(str) : precision1(double), predict_label2 : precision2(double), ...>
    """
    res = dict()
    for k, v in statistic_dict.items():
        predict_cnt = sum(v.values())
        actual_cnt = v.get(k, 0)
        res[k] = round(actual_cnt / predict_cnt, 2)
        
    digit_label_map = get_digit_label_map()
    for k, v in digit_label_map.items():
        if k not in res.keys():
            res[k] = 0
            
    return res


def cal_recall(statistic_dict: dict):
    """
    计算每个聚类标签的召回率：真实 label 个数被成功分类出来的个数占该真实 label 总数的比例
    :param statistic_dict: <actual_label(str) : <predict_label1(str) : cnt1(int), predict_label2(str) : cnt2(int), ...>, ...>
    :return: <actual_label1(str) : recall1(double), actual_label2(str) : recall2(double), ...>
    """
    res = dict()
    for k, v in statistic_dict.items():
        actual_cnt = sum(v.values())
        predict_cnt = v.get(k, 0)
        res[k] = round(predict_cnt / actual_cnt, 2)
    
    digit_label_map = get_digit_label_map()
    for k, v in digit_label_map.items():
        if k not in res.keys():
            res[k] = 0
            
    return res


def cal_f1_score(class_precision: dict, class_recall: dict):
    """
    计算每个聚类标签的 f1-score：f1-score = 2 * precision * recall / (precision + recall)
    :param class_precision: <label1(str) : precision1(double), label2(str) : precision2(double), ...>
    :param class_recall: <label1(str) : recall1(double), label2(str) : recall2(double), ...>
    :return: <label1(str) : f1-score1(double), label2(str) : f1-score2(double),
    """
    assert len(class_precision) == len(class_recall)
    res = dict()
    for k, precision in class_precision.items():
        recall = class_recall[k]
        if (precision + recall) != 0:
            res[k] = round(2 * precision * recall / (precision + recall), 2)
        else:
            res[k] = None
    return res


def cal_label_cnt(statistic_dict: dict):
    """
    计算各真实标签的数量
    :param statistic_dict: <actual_label(str) : <predict_label1(str) : cnt1(int), predict_label2(str) : cnt2(int), ...>, ...>
    :return: <label1 : cnt1, label2 : cnt2, ...,>
    """
    res = dict()
    for k, v in statistic_dict.items():
        res[k] = sum(v.values())
        
    digit_label_map = get_digit_label_map()
    for k, v in digit_label_map.items():
        if k not in res.keys():
            res[k] = 0
            
    return res


def get_digit_label_map():
    """
    获取 数字 - label 之间的映射
    :return: <idx: str, label: str>
    """
    # label_list = ['基础环境', '核心库', '核心工具', '系统服务', '系统库', '系统工具', '应用服务', '应用库', '应用工具', '编程语言', '其它']
    label_list = ['库', '工具', '服务', '其它']
    label_dict = dict()
    for i, enum in enumerate(label_list):
        label_dict[i] = enum
    return label_dict


def cal_metrics(predict_map: dict, actual_map: dict):
    """
    抽象出来的用于计算 accuracy、precision、recall、f1-score 的方法
    :param predict_map: <pkg: str, pre_label: str>
    :param actual_map: <pkg: str, act_label: str>
    :return: [[label1, precision1, recall1, f1-score1], [label2, precision2, recall2, f1-score2], ...]，可直接写 csv
    """
    precision_map = cal_precision(statistic(predict_map, actual_map))
    recall_map = cal_recall(statistic(actual_map, predict_map))
    f1score_map = cal_f1_score(precision_map, recall_map)
    label_cnt_map = cal_label_cnt(statistic(actual_map, predict_map))
    digit_label_map = get_digit_label_map()
    res = list()
    for k in sorted(int(x) for x in precision_map.keys()):
#         k = str(k)
        label = digit_label_map[k]
        label_cnt = label_cnt_map[k]
        precision = precision_map[k]
        recall = recall_map[k]
        f1score = f1score_map[k]
        row = [label, label_cnt, precision, recall, f1score]
        res.append(row)
    res.append([])
    row = ['模型整体 accuracy: ' + str(cal_acc(predict_map, actual_map))]
    res.append(row)
    return res

# =============================== preprocess ==================================

def get_splits(y, strategy="all", sample_size = 100):
    idx_list = np.arange(len(y))
    idx_neg = []
    idx_pos = []
    
    idx_set = []
    
    for i in range(11):
        idx_set.append([])

    for i, label in enumerate(y):
        label = np.argmax(label)
        idx_set[label].append(i)
    
    idx_train = []
    idx_val = []
    idx_test = []
    np.random.seed(0)
    
    for s in idx_set:
        np.random.shuffle(s)
        if strategy == "all":
            idx_train = idx_train + s[0:int(len(s)*0.6)]
            idx_val = idx_val + s[int(len(s) * 0.6):int(len(s) * 0.8)]
            idx_test = idx_test + s[:]
        elif strategy == "sample":
            idx_train = idx_train + s[0:sample_size]
            idx_val = idx_val + s[int(len(s) * 0.8):]
            idx_test = idx_test + s[:]
    
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

    return y_train, y_val, y_test,train_mask, val_mask, test_mask

idx_feat_lables = np.array(df_features)
unordered_edges = np.array(df_edges)

# 分离feature、label
features = idx_feat_lables[:, 2:]
onehot_labels = encode_onehot(idx_feat_lables[:, 1])

# 构建邻接矩阵
idx = np.array(idx_feat_lables[:, 0], dtype=np.int32)
idx_map = {j: i for i, j in enumerate(idx)}
edges = np.array(list(map(idx_map.get, unordered_edges.flatten())),dtype=np.int32).reshape(unordered_edges.shape)
adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)
adj = convert_symmetric(adj)

print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

#分割数据集
y_train, y_val, y_test, train_mask, val_mask, test_mask = get_splits(onehot_labels, strategy="sample", sample_size=400)

# ============================ sample neighs =============================

def sample_neighs(G, nodes, sample_num=None, self_loop=False, shuffle=True):  # 抽样邻居节点
    np.random.seed(0)
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
                samp_neigh + [samp_neigh[-1]] * (sample_num - len(samp_neigh)) for samp_neigh in samp_neighs]
        if shuffle:
            samp_neighs = [list(np.random.permutation(x)) for x in samp_neighs]
    else:
        samp_neighs = neighs
    return np.asarray(samp_neighs), np.asarray(list(map(len, samp_neighs)))

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


# =================================== model =======================================

class MeanAggregator(Layer):

    def __init__(self, units, input_dim, neigh_max, concat=True, dropout_rate=0.0, activation=tf.nn.relu, l2_reg=0,
                 use_bias=False,
                 seed=1024, **kwargs):
        super(MeanAggregator, self).__init__()
        self.units = units
        self.neigh_max = neigh_max
        self.concat = concat
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_bias = use_bias
        self.activation = activation
        self.seed = seed
        self.input_dim = input_dim

    def build(self, input_shapes):

        self.neigh_weights = self.add_weight(shape=(self.input_dim, self.units),
                                             initializer=glorot_uniform(
                                                 seed=self.seed),
                                             regularizer=l2(self.l2_reg),
                                             name="neigh_weights")
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units), initializer=Zeros(),
                                        name='bias_weight')

        self.dropout = Dropout(self.dropout_rate)
        self.built = True

    def call(self, inputs, training=None):
        features, node, neighbours = inputs

        node_feat = tf.nn.embedding_lookup(features, node)
        neigh_feat = tf.nn.embedding_lookup(features, neighbours)

        node_feat = self.dropout(node_feat, training=training)
        neigh_feat = self.dropout(neigh_feat, training=training)

        concat_feat = tf.concat([neigh_feat, node_feat], axis=1)
        concat_mean = tf.reduce_mean(concat_feat, axis=1, keepdims=False)

        output = tf.matmul(concat_mean, self.neigh_weights)
        if self.use_bias:
            output += self.bias
        if self.activation:
            output = self.activation(output)

        # output = tf.nn.l2_normalize(output,dim=-1)
        output._uses_learning_phase = True

        return output

    def get_config(self):
        config = {'units': self.units,
                  'concat': self.concat,
                  'seed': self.seed
                  }

        base_config = super(MeanAggregator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PoolingAggregator(Layer):

    def __init__(self, units, input_dim, neigh_max, aggregator='meanpooling', concat=True,
                 dropout_rate=0.0,
                 activation=tf.nn.relu, l2_reg=0, use_bias=False,
                 seed=1024, ):
        super(PoolingAggregator, self).__init__()
        self.output_dim = units
        self.input_dim = input_dim
        self.concat = concat
        self.pooling = aggregator
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_bias = use_bias
        self.activation = activation
        self.neigh_max = neigh_max
        self.seed = seed

        # if neigh_input_dim is None:

    def build(self, input_shapes):

        self.dense_layers = [Dense(
            self.input_dim, activation='relu', use_bias=True, kernel_regularizer=l2(self.l2_reg))]

        self.neigh_weights = self.add_weight(
            shape=(self.input_dim * 2, self.output_dim),
            initializer=glorot_uniform(
                seed=self.seed),
            regularizer=l2(self.l2_reg),

            name="neigh_weights")

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=Zeros(),
                                        name='bias_weight')

        self.built = True

    def call(self, inputs, mask=None):

        features, node, neighbours = inputs

        node_feat = tf.nn.embedding_lookup(features, node)
        neigh_feat = tf.nn.embedding_lookup(features, neighbours)

        dims = tf.shape(neigh_feat)
        batch_size = dims[0]
        num_neighbors = dims[1]
        h_reshaped = tf.reshape(
            neigh_feat, (batch_size * num_neighbors, self.input_dim))

        for l in self.dense_layers:
            h_reshaped = l(h_reshaped)
        neigh_feat = tf.reshape(
            h_reshaped, (batch_size, num_neighbors, int(h_reshaped.shape[-1])))

        if self.pooling == "meanpooling":
            neigh_feat = tf.reduce_mean(neigh_feat, axis=1, keep_dims=False)
        else:
            neigh_feat = tf.reduce_max(neigh_feat, axis=1)

        output = tf.concat(
            [tf.squeeze(node_feat, axis=1), neigh_feat], axis=-1)

        output = tf.matmul(output, self.neigh_weights)
        if self.use_bias:
            output += self.bias
        if self.activation:
            output = self.activation(output)

        # output = tf.nn.l2_normalize(output, dim=-1)

        return output

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'concat': self.concat
                  }

        base_config = super(PoolingAggregator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def GraphSAGE(feature_dim, neighbor_num, n_hidden, n_classes, use_bias=True, activation=tf.nn.relu,
              aggregator_type='mean', dropout_rate=0.0, l2_reg=0):
    features = Input(shape=(feature_dim,))
    node_input = Input(shape=(1,), dtype=tf.int32)
    neighbor_input = [Input(shape=(l,), dtype=tf.int32) for l in neighbor_num]

    if aggregator_type == 'mean':
        aggregator = MeanAggregator
    else:
        aggregator = PoolingAggregator

    h = features
    for i in range(0, len(neighbor_num)):
        if i > 0:
            feature_dim = n_hidden
        if i == len(neighbor_num) - 1:
            activation = tf.nn.softmax
            n_hidden = n_classes
        h = aggregator(units=n_hidden, input_dim=feature_dim, activation=activation, l2_reg=l2_reg, use_bias=use_bias,
                       dropout_rate=dropout_rate, neigh_max=neighbor_num[i], aggregator=aggregator_type)(
            [h, node_input, neighbor_input[i]])  #

    output = h
    input_list = [features, node_input] + neighbor_input
    model = Model(input_list, outputs=output)
    return model

# ============================= train ================================

def create_f1():
    def f1_function(y_true, y_pred):
        y_pred_binary = tf.where(y_pred>=0.5, 1, 0)
        tp = tf.reduce_sum(y_true * y_pred_binary)
        predicted_positives = tf.reduce_sum(y_pred_binary)
        possible_positives = tf.reduce_sum(y_true)
        return tp, predicted_positives, possible_positives
    return f1_function


class F1_score(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) # handles base args (e.g., dtype)
        self.f1_function = create_f1()
        self.tp_count = self.add_weight("tp_count", initializer="zeros")
        self.all_predicted_positives = self.add_weight('all_predicted_positives', initializer='zeros')
        self.all_possible_positives = self.add_weight('all_possible_positives', initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        tp, predicted_positives, possible_positives = self.f1_function(y_true, y_pred)
        self.tp_count.assign_add(tp)
        self.all_predicted_positives.assign_add(predicted_positives)
        self.all_possible_positives.assign_add(possible_positives)
    def result(self):
        precision = self.tp_count / self.all_predicted_positives
        recall = self.tp_count / self.all_possible_positives
        f1 = 2*(precision*recall)/(precision+recall)
        return f1


model = GraphSAGE(feature_dim=features.shape[1],
                  neighbor_num=neigh_maxlen,
                  n_hidden=16,
                  n_classes=y_train.shape[1],
                  use_bias=True,
                  activation=tf.nn.relu,
                  aggregator_type='pooling',
                  dropout_rate=0.5, l2_reg=2.5e-4)

model.compile(adam_v2.Adam(0.01), 'categorical_crossentropy', weighted_metrics=['categorical_crossentropy', 'acc'])

val_data = (model_input, y_val, val_mask)

print("start training")
history = model.fit(model_input, y_train, sample_weight=train_mask, validation_data=val_data, batch_size=A.shape[0], epochs=2400,
                    shuffle=False, verbose=1)

eval_results = model.evaluate(model_input, y_test, sample_weight=test_mask, batch_size=A.shape[0])
print('Done.\n'
      'Test accuracy: {}\n'
      'Test F1-score: {}'.format(*eval_results))


# ======================= predict =======================
predictions = model.predict(model_input, batch_size = A.shape[0])
predictions = np.argmax(predictions, axis = 1)

df_node =  pd.read_csv("../output/nodes.csv")
nodes = np.array(df_node)
labels = idx_feat_lables[:, 1]
nodes_true_prediction = np.hstack((nodes, labels[:,np.newaxis].astype(np.int32), predictions[:,np.newaxis]))
result = nodes_true_prediction.tolist()

predict_map = {}
actual_map = {}

for res in result:
    predict_map[res[0]] = res[3]
    actual_map[res[0]] = res[2]

metrics = cal_metrics(predict_map, actual_map)


with open("../output/GraphSAGE_statistic.csv", "w", newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["label", "count", "precision", "recall", "f1-score"])
    writer.writerows(metrics)

digit_label_map = get_digit_label_map()
result = [[r[0], r[1], digit_label_map[r[2]], digit_label_map[r[3]]] for r in result]

with open("../output/GraphSAGE_result.csv", "w", newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["idx", "name", "y_true", "y_predict"])
    writer.writerows(result)

# show
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()