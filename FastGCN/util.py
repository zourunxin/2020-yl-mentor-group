import torch
import numpy as np

from sklearn.metrics import precision_recall_fscore_support

from utils.utils import get_label_list


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def get_batches(train_ind, train_labels, batch_size=64, shuffle=True):
    """
    Inputs:
        train_ind: np.array
    """
    nums = train_ind.shape[0]
    if shuffle:
        np.random.shuffle(train_ind)
    i = 0
    while i < nums:
        cur_ind = train_ind[i:i + batch_size]
        cur_labels = train_labels[cur_ind]
        yield cur_ind, cur_labels
        i += batch_size

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def prf(output, labels):
    preds = output.max(1)[1].type_as(labels)
    p, r, f, s = precision_recall_fscore_support(labels, preds)
    return p[0], r[0], f[0]

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