import sys
sys.path.append("../")
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import utils.CommonUtils as CommonUtils
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pdb

def encode_onehot(label_num_map, labels):
    classes_dict = {label: np.identity(len(label_num_map))[label_num_map[label], :] for label in label_num_map}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

df_data = pd.read_csv('datasource.csv')
idx_name_map, name_idx_map = CommonUtils.get_idx_name_map(df_data["name"])
num_label_map, label_num_map = CommonUtils.get_num_label_map(df_data["label"])

df_train, df_test = train_test_split(df_data, test_size=0.3, random_state=666, stratify=df_data["label"])

graph_feat_train = df_train["graph_feat"].values.tolist()
graph_feat_train = [[float(str_num) for str_num in str_feat.split(" ")] for str_feat in graph_feat_train]
graph_feat_test = df_test["graph_feat"].values.tolist()
graph_feat_test = [[float(str_num) for str_num in str_feat.split(" ")] for str_feat in graph_feat_test]

arti_feat_train = df_train["arti"].values.tolist()
arti_feat_train = [[float(str_num) for str_num in str_feat.split(" ")] for str_feat in arti_feat_train]
arti_feat_test = df_test["arti"].values.tolist()
arti_feat_test = [[float(str_num) for str_num in str_feat.split(" ")] for str_feat in arti_feat_test]

onehot_labels = encode_onehot(label_num_map, list(df_train["label"]))

rfc = RandomForestClassifier(random_state=0)
rfc = rfc.fit(arti_feat_train, np.argmax(onehot_labels, axis=1))

# pdb.set_trace()
preds = rfc.predict(arti_feat_test)
print(preds)

df_test["predict"] = [num_label_map[pred] for pred in preds]

report = classification_report(df_test["label"], df_test["predict"], digits=3)
print(report)





