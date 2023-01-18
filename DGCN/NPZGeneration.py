import networkx as nx
import numpy as np
import scipy.sparse as sp
import pandas as pd
import utils.NLPUtils as NLPUtils
import FeatureExtractor.extractors as Extractors


def generate_npz():
    # 构造图
    if di == 'di':
        g = nx.DiGraph()
    else:
        g = nx.Graph()

    df_data = pd.read_csv('/Users/zourunxin/Mine/Seminar/20Data/1228/datasource_1228_rpm.csv')
    df_edges = pd.read_csv('/Users/zourunxin/Mine/Seminar/20Data/1228/rpm/edges.csv')
    # 准备 nodes
    nodes = list(df_data["name"])
    # 准备 edges
    edges = []
    for edge1, edge2 in zip(df_edges['out'], df_edges['in']):
        edges.append([edge1, edge2])
    # 构造图
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    adj = nx.adjacency_matrix(g)

    # 准备 features
    texts = list(df_data["text"].apply(lambda x: NLPUtils.preprocess_text(x)))
    name_features = Extractors.name_feat_extractor(texts)
    keyword_features = Extractors.keyword_feat_extractor(texts)
    tfidf_features = Extractors.tfidf_feat_extractor(texts, max_features=2000)
    features = np.hstack((name_features, keyword_features, tfidf_features))
    # 构造图结点特征
    attr = sp.csr_matrix(np.array(features))
    # 准备 labels
    labels = list(df_data["label"].apply(lambda x: label_num.get(x, default_num)))

    # 写 npz
    np.savez("../output/DGCN/data/{}/raw/{}{}{}{}{}.npz"
             .format(rpm, version, mode, di, feat_extr, backup),
             adj_data=adj.data, adj_indices=adj.indices, adj_indptr=adj.indptr, adj_shape=adj.shape, labels=labels,
             attr_data=attr.data, attr_indices=attr.indices, attr_indptr=attr.indptr, attr_shape=attr.shape)

    print_npz()
    return


def print_npz(file="../output/DGCN/data/{}/raw/{}{}{}{}{}.npz"):
    file = file.format(rpm, version, mode, di, feat_extr, backup)
    reader = np.load(file, allow_pickle=True)
    cla = sorted(reader.files)
    print(cla)
    for i in cla:
        print(reader[i])
        print(' ')
    print('labels 分别是 {}'.format(set(reader['labels'])))


if __name__ == '__main__':
    rpm = 'rpm'
    version = '1228'
    mode = 'layer'
    di = 'di'
    feat_extr = 'tfidf'
    backup = '2000feature'

    if mode == 'layer':
        label_num = {'基础环境': 0, '核心库': 0, '核心服务': 0, '核心工具': 0, '系统库': 1, '系统服务': 1, '系统工具': 1, '应用库': 2,
                 '应用服务': 2, '应用工具': 2}
        default_num = 3
    else:
        label_num = {'基础环境': 0, '核心库': 1, '核心服务': 1, '核心工具': 2, '系统库': 3, '系统服务': 4, '系统工具': 5, '应用库': 6,
                     '应用服务': 7, '应用工具': 8}
        default_num = 9
    generate_npz()
