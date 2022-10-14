import sys
sys.path.append("../")
from random import sample
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp
import pandas as pd
import csv

from keras import activations, constraints, initializers, regularizers
from keras import backend as K
from keras.layers import Input, Layer, Dropout, LeakyReLU
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from utils.utils import get_label_list


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
    label_list = get_label_list()
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

# ================================ Model =================================

class GraphAttention(Layer):

    def __init__(self,
                 F_,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 dropout_rate=0.5,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.F_ = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.dropout_rate = dropout_rate  # Internal dropout rate
        self.activation = activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = []       # Layer kernels for attention heads
        self.biases = []        # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = self.F_ * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_

        super(GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][-1]

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, self.F_),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name='kernel_{}'.format(head))
            self.kernels.append(kernel)

            # # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.F_, ),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name='bias_{}'.format(head))
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(shape=(self.F_, 1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint,
                                               name='attn_kernel_self_{}'.format(head),)
            attn_kernel_neighs = self.add_weight(shape=(self.F_, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint,
                                                 name='attn_kernel_neigh_{}'.format(head))
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
        self.built = True

    def call(self, inputs):
        X = inputs[0]  # Node features (N x F)
        A = inputs[1]  # Adjacency matrix (N x N)

        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W in the paper (F x F')
            attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            features = K.dot(X, kernel)  # (N x F')

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(features, attention_kernel[0])    # (N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = K.dot(features, attention_kernel[1])  # (N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            dense = attn_for_self + K.transpose(attn_for_neighs)  # (N x N) via broadcasting

            # Add nonlinearty
            dense = LeakyReLU(alpha=0.2)(dense)

            # Mask values before activation (Vaswani et al., 2017)
            mask = -10e9 * (1.0 - A)
            dense += mask

            # Apply softmax to get attention coefficients
            dense = K.softmax(dense)  # (N x N)

            # Apply dropout to features and attention coefficients
            dropout_attn = Dropout(self.dropout_rate)(dense)  # (N x N)
            dropout_feat = Dropout(self.dropout_rate)(features)  # (N x F')

            # Linear combination with neighbors' features
            node_features = K.dot(dropout_attn, dropout_feat)  # (N x F')

            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])

            # Add output of attention head to final output
            outputs.append(node_features)

        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs)  # (N x KF')
        else:
            output = K.mean(K.stack(outputs), axis=0)  # N x F')

        output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape

# ============================= Build Model ===========================

# Prepared data
Y_train, Y_val, Y_test, idx_train, idx_val, idx_test = y_train, y_val, y_test, train_mask, val_mask, test_mask
A = preprocess_adj(adj)
X = features


# Parameters
N = X.shape[0]                # Number of nodes in the graph
F = X.shape[1]                # Original feature dimension
n_classes = Y_train.shape[1]  # Number of classes
F_ = 8                        # Output size of first GraphAttention layer
n_attn_heads = 8              # Number of attention heads in first GAT layer
dropout_rate = 0.6            # Dropout rate (between and inside GAT layers)
l2_reg = 5e-4/2               # Factor for l2 regularization
learning_rate = 1e-3          # Learning rate for Adam
epochs = 800                  # Number of training epochs
es_patience = 100             # Patience fot early stopping

# Model definition (as per Section 3.3 of the paper)
X_in = Input(shape=(F,))
A_in = Input(shape=(N,))

dropout1 = Dropout(dropout_rate)(X_in)
graph_attention_1 = GraphAttention(F_,
                                   attn_heads=n_attn_heads,
                                   attn_heads_reduction='concat',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l2(l2_reg),
                                   attn_kernel_regularizer=l2(l2_reg))([dropout1, A_in])
dropout2 = Dropout(dropout_rate)(graph_attention_1)
graph_attention_2 = GraphAttention(n_classes,
                                   attn_heads=1,
                                   attn_heads_reduction='average',
                                   dropout_rate=dropout_rate,
                                   activation='softmax',
                                   kernel_regularizer=l2(l2_reg),
                                   attn_kernel_regularizer=l2(l2_reg))([dropout2, A_in])

# Build model
model = Model(inputs=[X_in, A_in], outputs=graph_attention_2)

model.compile(optimizer=Adam(learning_rate),
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])

model.summary()

# Callbacks
mc_callback = ModelCheckpoint('./best_model.h5',
                              monitor='categorical_crossentropy',
                              save_best_only=True,
                              save_weights_only=True)


# ======================= Train =======================


validation_data = ([X, A], Y_val, idx_val)

history = model.fit([X, A],
          Y_train,
          sample_weight=idx_train,
          epochs=epochs,
          batch_size=N,
          validation_data=validation_data,
          shuffle=False,  # Shuffling data means shuffling the whole graph
          callbacks=[mc_callback])


# =================== Evaluate =======================

# Load best model
# model.load_weights('./best_model.h5')




# Evaluate model
eval_results = model.evaluate([X, A],
                              Y_test,
                              sample_weight=idx_test,
                              batch_size=N,
                              verbose=0)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))

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

predictions = model.predict([X, A], batch_size = N)
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


with open("../output/GAT_statistic.csv", "w", newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["label", "count", "precision", "recall", "f1-score"])
    writer.writerows(metrics)

digit_label_map = get_digit_label_map()
result = [[r[0], r[1], digit_label_map[r[2]], digit_label_map[r[3]]] for r in result]

with open("../output/GAT_result.csv", "w", newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["idx", "name", "y_true", "y_predict"])
    writer.writerows(result)