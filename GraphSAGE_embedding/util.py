import pdb
from collections import defaultdict
import numpy as np
import pandas as pd

import utils.CommonUtils as CommonUtils
import utils.NLPUtils as NLPUtils
from utils.FileUtil import csv_reader
import FeatureExtractor.extractors as Extractors


def load_data(cfg):
    num_nodes = cfg.num_nodes
    num_feats = cfg.num_features
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open(cfg.path + 'cora.content') as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            # feat_data[i,:] = map(float, info[1:-1])
            feat_data[i,:] = [float(x) for x in info[1:-1]]
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open(cfg.path + 'cora.cites') as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists, node_map


def load_data2(cfg):
    df_data = pd.read_csv('../output/datasource_1228.csv')
    processed_texts = list(df_data["text"].apply(lambda x: NLPUtils.preprocess_text(x)))
    feat_data = Extractors.tfidf_feat_extractor(processed_texts, feature_num=1000)
    idx_name_map, node_map = CommonUtils.get_idx_name_map(df_data["name"])
    num_label_map, label_num_map = CommonUtils.get_num_label_map(df_data["label"])
    labels = [label_num_map[label] for label in df_data["label"]]

    reader = csv_reader('../output/edges.csv')
    adj_lists = defaultdict(set)
    for line in reader:
        pkg1 = node_map[line[0]]
        pkg2 = node_map[line[1]]
        adj_lists[pkg1].add(pkg2)
        adj_lists[pkg2].add(pkg1)
    return feat_data, labels, adj_lists, node_map
