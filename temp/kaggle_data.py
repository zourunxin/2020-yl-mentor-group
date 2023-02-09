import sys
sys.path.append("../")
import pandas as pd
import utils.CommonUtils as CommonUtils
import utils.NLPUtils as NLPUtils
import FeatureExtractor.extractors as Extractors
import numpy as np
import pdb


def encode_onehot(label_num_map, labels):
    classes_dict = {label: np.identity(len(label_num_map))[label_num_map[label], :] for label in label_num_map}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

df_data = pd.read_csv('../output/datasource_0205_class.csv')
df_edges = pd.read_csv('../output/edges.csv')
df_graph_feat = pd.read_csv('./graph_feat.csv')
idx_name_map, name_idx_map = CommonUtils.get_idx_name_map(df_data["name"])
num_label_map, label_num_map = CommonUtils.get_num_label_map(df_data["label"])

num_classes = len(num_label_map)
onehot_labels = encode_onehot(label_num_map, list(df_data["label"]))
texts = list(df_data["text"].apply(lambda x: NLPUtils.preprocess_text(x)))
tfidf_features = Extractors.tfidf_feat_extractor(texts, onehot_labels, feature_num=100)
tfidf_features = np.array(tfidf_features)
name_features = Extractors.name_feat_extractor(list(df_data["name"]))
meta_features = Extractors.meta_feat_extractor(df_data, df_edges)
arti_feats = np.hstack((tfidf_features, name_features, meta_features))


arti_feats = list(arti_feats)
for i ,_ in enumerate(arti_feats):
    arti_feats[i] = [str(num) for num in list(arti_feats[i])]

arti_feats = [" ".join(feats) for feats in arti_feats]

df_data["arti"] = arti_feats
df_data["graph_feat"] = df_graph_feat["graph_feat"]
df_data.to_csv("datasource.csv", index=False)

