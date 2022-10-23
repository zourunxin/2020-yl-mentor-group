import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.sparse as sp
import csv
import pandas as pd
import argparse

# =============================== util.py =================================
from utils.utils import get_label_list


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
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

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


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
        res[k] = actual_cnt / predict_cnt

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
        res[k] = (predict_cnt / actual_cnt)

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
            res[k] = 2 * precision * recall / (precision + recall)
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


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class GCNLayer(nn.Module):
    def __init__(self,input_features,output_features,bias=False):
        super(GCNLayer,self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weights = nn.Parameter(torch.FloatTensor(input_features,output_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_features))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1./math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-std,std)
        if self.bias is not None:
            self.bias.data.uniform_(-std,std)

    def forward(self,adj,x):
        support = torch.mm(x,self.weights)
        output = torch.spmm(adj,support)
        if self.bias is not None:
            return output+self.bias
        return output

class GCN(nn.Module):
    def __init__(self,input_size,hidden_size,num_class,dropout,bias=False):
        super(GCN,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_class = num_class
        self.gcn1 = GCNLayer(input_size,hidden_size,bias=bias)
        self.gcn2 = GCNLayer(hidden_size,num_class,bias=bias)
        self.dropout = dropout
    def forward(self,adj,x):
        x = F.relu(self.gcn1(adj,x))
        x = F.dropout(x,self.dropout,training=self.training)
        x = self.gcn2(adj,x)
        return F.log_softmax(x,dim=1)

# =================================== load data =====================================

def load_data(strategy="all", sample_size = 100):
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
    df_classes = pd.read_csv("../output/idx_label.csv")

    idx_feat_lables = np.array(df_features)
    unordered_edges = np.array(df_edges)

    # 分离feature、label
    features = sp.csr_matrix(idx_feat_lables[:, 2:], dtype=np.float32)
    onehot_labels = encode_onehot(idx_feat_lables[:, 1])

    # 构建邻接矩阵
    idx = np.array(idx_feat_lables[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, unordered_edges.flatten())), dtype=np.int32).reshape(unordered_edges.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)
    adj = convert_symmetric(adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_set = []

    for i in range(5):
        idx_set.append([])

    for i, label in enumerate(onehot_labels):
        label = np.argmax(label)
        idx_set[label].append(i)

    idx_train = []
    idx_val = []
    idx_test = []
    np.random.seed(0)

    for s in idx_set:
        np.random.shuffle(s)
        if strategy == "all":
            idx_train = idx_train + s[0:int(len(s) * 0.6)]
            idx_val = idx_val + s[int(len(s) * 0.6):int(len(s) * 0.8)]
            idx_test = idx_test + s[:]
        elif strategy == "sample":
            idx_train = idx_train + s[0:sample_size]
            idx_val = idx_val + s[int(len(s) * 0.8):]
            idx_test = idx_test + s[:]

    print("样本数量 train: {}, val: {}, test: {}".format(len(idx_train), len(idx_val), len(idx_test)))

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(onehot_labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, idx_feat_lables



def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train_gcn(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(adj,features)
    loss = F.nll_loss(output[idx_train],labels[idx_train])
    acc = accuracy(output[idx_train],labels[idx_train])
    loss.backward()
    optimizer.step()
    loss_val = F.nll_loss(output[idx_val],labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss.item()),
          'acc_train: {:.4f}'.format(acc.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

def test():
    model.eval()
    output = model(adj,features)
    print(output[0:10])
    print(labels[0:10])
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    predictions = np.argmax(output.detach().numpy(), axis=1)

    df_node = pd.read_csv("../output/nodes.csv")
    nodes = np.array(df_node)

    label_list = idx_feat_lables[:, 1]
    nodes_true_prediction = np.hstack((nodes, label_list[:, np.newaxis].astype(np.int32), predictions[:, np.newaxis]))
    result = nodes_true_prediction.tolist()

    return result

if __name__ == '__main__':
    # 训练预设
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')

    args = parser.parse_args()
    np.random.seed(args.seed)
    adj, features, labels, idx_train, idx_val, idx_test, idx_feat_lables = load_data(strategy="sample", sample_size = 100)
    model = GCN(features.shape[1],args.hidden,labels.max().item() + 1,dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        train_gcn(epoch)

    result = test()

    predict_map = {}
    actual_map = {}

    for res in result:
        predict_map[res[0]] = res[3]
        actual_map[res[0]] = res[2]

    metrics = cal_metrics(predict_map, actual_map)

    with open("../output/GCN_statistic.csv", "w", newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["label", "count", "precision", "recall", "f1-score"])
        writer.writerows(metrics)

    digit_label_map = get_digit_label_map()
    result = [[r[0], r[1], digit_label_map[r[2]], digit_label_map[r[3]]] for r in result]

    with open("../output/GCN_result.csv", "w", newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["idx", "name", "y_true", "y_predict"])
        writer.writerows(result)

