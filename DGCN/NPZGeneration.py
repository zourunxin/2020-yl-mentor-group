import networkx as nx
import numpy as np
from utils.FileUtil import csv_reader, write_csv
from utils.CommonUtils import get_num_label_map
import csv
import scipy.sparse as sp


def generate_npz():
    # 构造图
    if di == 'di':
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    # 准备 nodes
    reader = csv_reader('/Users/zourunxin/Mine/Seminar/20Data/{}/{}/nodes.csv'.format(version, dataset))
    node_num = 0
    for _ in reader:
        node_num += 1
    g.add_nodes_from(range(0, node_num))
    # 准备 edges
    reader = csv_reader('/Users/zourunxin/Mine/Seminar/20Data/{}/{}/edges.csv'.format(version, dataset))
    edges = list()
    edge_num = 0
    for line in reader:
        edges.append([int(line[0]), int(line[1])])
        edge_num += 1
    g.add_edges_from(edges)
    adj = nx.adjacency_matrix(g)  # 构造稀疏矩阵

    # 准备 feature
    reader = csv.reader(open('/Users/zourunxin/Mine/Seminar/20Data/{}/{}/{}feat_name_label_feature_{}.csv'
                             .format(version, dataset, feat_num, feat_extr), encoding='utf-8-sig'))
    features = list()
    for line in reader:
        line = line[2:]
        tmp = [float(i) for i in line]
        features.append(tmp)
    attr = sp.csr_matrix(np.array(features))   # 获取 data、indices、indptr

    # 准备 label
    reader = csv.reader(open('/Users/zourunxin/Mine/Seminar/20Data/{}/{}/{}feat_name_label_feature_{}.csv'
                             .format(version, dataset, feat_num, feat_extr), encoding='utf-8-sig'))
    label = list()
    for line in reader:
        label.append(label_num.get(line[1], default_num))
    label = np.array(label)

    # 写 npz
    np.savez("/Users/zourunxin/Mine/Seminar/20Data/1228/DiGCN/data/{}/raw/{}{}{}{}.npz"
             .format(dataset, version, di, feat_extr, feat_num),
             adj_data=adj.data, adj_indices=adj.indices, adj_indptr=adj.indptr, adj_shape=adj.shape, labels=label,
             attr_data=attr.data, attr_indices=attr.indices, attr_indptr=attr.indptr, attr_shape=attr.shape)

    print_npz()
    return


def print_npz(file="/Users/zourunxin/Mine/Seminar/20Data/1228/DiGCN/data/{}/raw/{}{}{}{}.npz"):
    file = file.format(dataset, version, di, feat_extr, feat_num)
    reader = np.load(file, allow_pickle=True)
    cla = sorted(reader.files)
    print(cla)
    for i in cla:
        print(reader[i])
        print(' ')
    print('labels 分别是 {}'.format(set(reader['labels'])))


if __name__ == '__main__':
    version = '1228'
    mode = 'layer'
    di = 'di'
    dataset = 'rpm'
    feat_extr = 'tfidf'
    feat_num = 1500

    if mode == 'layer':
        label_num = {'基础环境': 0, '核心库': 0, '核心服务': 0, '核心工具': 0, '系统库': 1, '系统服务': 1, '系统工具': 1, '应用库': 2,
                 '应用服务': 2, '应用工具': 2}
        default_num = 3
    else:
        label_num = {'基础环境': 0, '核心库': 1, '核心服务': 1, '核心工具': 2, '系统库': 3, '系统服务': 4, '系统工具': 5, '应用库': 6,
                     '应用服务': 7, '应用工具': 8}
        default_num = 9
    generate_npz()
