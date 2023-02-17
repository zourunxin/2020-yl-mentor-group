import random
from gensim.models import Word2Vec
import networkx as nx
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from utils import CommonUtils
from utils.FileUtil import csv_reader

# 从 start_node 开始随机游走
def deepwalk_walk(G, walk_length, start_node):
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = list(G.neighbors(cur))
        if len(cur_nbrs) > 0:
            walk.append(random.choice(cur_nbrs))
        else:
            break
    return walk

# 产生随机游走序列
def _simulate_walks(G, nodes, num_walks, walk_length):
    walks = []
    for _ in range(num_walks):
        random.shuffle(nodes)
        for v in nodes:
            walks.append(deepwalk_walk(G, walk_length=walk_length, start_node=v))
    return walks


def main():
    df_data = pd.read_csv('../output/datasource_1228_without_isolate_node.csv')
    idx_name_map, node_map = CommonUtils.get_idx_name_map(df_data["name"])
    G = nx.Graph()
    edges_reader = csv_reader('../output/edges.csv')
    for line in edges_reader:
        G.add_edge(node_map[line[0]], node_map[line[1]])
    # 得到所有节点
    nodes = list(G.nodes())
    # 得到序列
    walks = _simulate_walks(G, nodes, num_walks=10, walk_length=2)
    # 默认嵌入到100维
    w2v_model = Word2Vec(walks, sg=1, hs=1)
    # 打印其中一个节点的嵌入向量
    embedding = w2v_model.wv.vectors
    return embedding

