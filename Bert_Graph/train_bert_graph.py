import sys
sys.path.append("../")
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import bert_graph
import matplotlib.pyplot as plt
import time
import pdb
import utils.CommonUtils as CommonUtils
import utils.NLPUtils as NLPUtils
from transformers import AutoTokenizer
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import transformers
import FeatureExtractor.extractors as Extractors

def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    # adj = normalize_adj(adj, symmetric)
    return adj.todense()

def encode_onehot(label_num_map, labels):
    classes_dict = {label: np.identity(len(label_num_map))[label_num_map[label], :] for label in label_num_map}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


bert_path = "prajjwal1/bert-tiny"

tokenizer = AutoTokenizer.from_pretrained(bert_path)

# ========== dataset: ==========
print("开始读取数据")
df_data = pd.read_csv('../output/datasource_0205_class.csv')
# df_data = df_data.sample(4000, random_state=666)
df_edges = pd.read_csv('../output/edges.csv')
# df_edges = df_edges.loc[lambda df : (df['out'].isin(df_data["name"])) & (df['in'].isin(df_data["name"]))]
processed_texts = list(df_data["text"].apply(lambda x: NLPUtils.preprocess_text(x)))
idx_name_map, name_idx_map = CommonUtils.get_idx_name_map(df_data["name"])
num_label_map, label_num_map = CommonUtils.get_num_label_map(df_data["label"])
num_classes = len(num_label_map)
onehot_labels = encode_onehot(label_num_map, list(df_data["label"]))


adj = sp.coo_matrix((np.ones(len(df_edges)), 
                    (df_edges['out'].apply(lambda x: name_idx_map[x]), df_edges['in'].apply(lambda x: name_idx_map[x]))),
                    shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)
print('数据集含有 {} 个结点, {} 条边'.format(adj.shape[0], len(df_edges)))



df_temp = df_data[["name", "label"]]
df_temp["idx"] = df_temp["name"].apply(lambda x: name_idx_map[x])
df_temp_train, df_temp_test = train_test_split(df_temp, test_size=0.3, random_state=666, stratify=df_temp["label"])

idx_train = list(df_temp_train["idx"])
idx_test = list(df_temp_test["idx"])
train_mask = sample_mask(idx_train, onehot_labels.shape[0])
test_mask = sample_mask(idx_test, onehot_labels.shape[0])
y_train = np.zeros(onehot_labels.shape, dtype=np.int32)
y_test = np.zeros(onehot_labels.shape, dtype=np.int32)
y_train[idx_train] = onehot_labels[idx_train]
y_test[idx_test] = onehot_labels[idx_test]


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


# ========== parameters: ==========
maxlen = 32
hidden_size = 768
graph_batch_size = A.shape[0]
bert_epochs = 50
graph_epochs = 3000
all_epochs = 200
mode = "both"
with_tfidf = False
tf_idf_feat_num = 1000
end2end = True

def tokenize(df):
    t = time.time()
    X_name = []
    X_input_ids = []
    X_token_type_ids = []
    X_attention_mask = []
    X_offset_mapping = []
    y = []
    i = 0
    for content,label,name in zip(list(df.text),list(df.label),list(df.name)):
        bert_input = tokenizer.encode_plus(content, 
            add_special_tokens = True, # add [CLS], [SEP]
            max_length = maxlen, # max length of the text that can go to BERT
            pad_to_max_length = True, # add [PAD] tokens
            return_offsets_mapping = True,
            return_attention_mask = True, # add attention mask to not focus on pad tokens
            truncation=True)
        
        X_input_ids.append(bert_input['input_ids'])
        X_token_type_ids.append(bert_input['token_type_ids'])
        X_attention_mask.append(bert_input['attention_mask'])
        # X_offset_mapping.append(bert_input['offset_mapping'])
        y.append(label_num_map[label])
        X_name.append(name)
    X_input_ids = np.array(X_input_ids)
    X_token_type_ids = np.array(X_token_type_ids)
    X_attention_mask = np.array(X_attention_mask)
    # X_offset_mapping = np.array(X_offset_mapping)
    y = np.array(y)
    print('tokenizing time cost:',time.time()-t,'s.')

    return X_input_ids, X_token_type_ids, X_attention_mask, y, X_name

indexs = np.arange(A.shape[0])
# 两个隐藏层，定义每个层抽样邻居的个数
neigh_number = [5, 10]
neigh_maxlen = []

X_input_ids, X_token_type_ids, X_attention_mask, y, name = tokenize(df_data)
X_tfidf = Extractors.tfidf_feat_extractor(processed_texts, onehot_labels, feature_num=tf_idf_feat_num)

model_input = [X_input_ids, X_token_type_ids, X_attention_mask, X_tfidf, np.asarray(indexs, dtype=np.int32)]

for num in neigh_number:
    sample_neigh, sample_neigh_len = sample_neighs(G, indexs, num, self_loop=True)
    model_input.extend([sample_neigh])
    neigh_maxlen.append(max(sample_neigh_len))


model = bert_graph.BERT_GraphSAGE(bert_path, maxlen, hidden_size, num_classes, neigh_maxlen, 
                                  end2end=end2end, tfidf_features=tf_idf_feat_num,
                                  n_hidden=16, use_bias=True, with_tfidf=with_tfidf)

y = [y_train, y_test]
mask = [train_mask, test_mask]

bert_prob, graph_prob, all_prob, bert_history, graph_history, all_history = model.train_val(
                                                    model_input, y, mask,
                                                    graph_batch_size, bert_epochs, graph_epochs, all_epochs,
                                                    save_best=True, mode=mode)

name_pred_map = {}
name_test = [idx_name_map[idx] for idx in idx_test]

if end2end:
    for idx in idx_test:
        predict_label = num_label_map[np.argmax(all_prob[idx])]
        name_pred_map[idx_name_map[idx]] = predict_label

    df_test_list = []
    for name in name_pred_map:
        true_label = df_data.loc[df_data["name"] == name]["label"].values.tolist()[0]
        summary = df_data.loc[df_data["name"] == name]["summary"].values.tolist()[0]
        description = df_data.loc[df_data["name"] == name]["description"].values.tolist()[0]
        df_test_list.append([name, true_label, name_pred_map[name], summary, description])

    df_test = pd.DataFrame(df_test_list,columns=['name', 'label', 'bert_graph_pred', 'summary', 'description'])
    df_test.to_csv("bert_graph_end2end_result.csv", index=False)
    df_all_error = df_test.loc[df_test["label"] != df_test["bert_graph_pred"]]
    df_all_error.to_csv("bert_graph_end2end_result_error.csv", index=False)

    all_report = classification_report(df_test["label"], df_test["bert_graph_pred"], digits=3)
    print("all result:")
    print(all_report)


    with open('./log.txt', 'a+', encoding='utf-8') as f:
        print("all result:", file=f)
        print(all_report, file=f)
else:
    for idx in idx_test:
        bert_label = num_label_map[np.argmax(bert_prob[idx])]
        graph_label = num_label_map[np.argmax(graph_prob[idx])]
        name_pred_map[idx_name_map[idx]] = {'bert': bert_label, 'graph': graph_label}

    df_test_list = []
    for name in name_pred_map:
        true_label = df_data.loc[df_data["name"] == name]["label"].values.tolist()[0]
        summary = df_data.loc[df_data["name"] == name]["summary"].values.tolist()[0]
        description = df_data.loc[df_data["name"] == name]["description"].values.tolist()[0]
        df_test_list.append([name, true_label, name_pred_map[name]['bert'], name_pred_map[name]['graph'], summary, description])


    df_test = pd.DataFrame(df_test_list,columns=['name', 'label', 'bert_pred', 'graph_pred', 'summary', 'description'])
    df_test.to_csv("bert_graph_result.csv", index=False)
    df_bert_error = df_test.loc[df_test["label"] != df_test["bert_pred"]]
    df_graph_error = df_test.loc[df_test["label"] != df_test["graph_pred"]]
    df_bert_error.to_csv("bert_result_error.csv", index=False)
    df_graph_error.to_csv("bert_graph_result_error.csv", index=False)

    bert_report = classification_report(df_test["label"], df_test["bert_pred"], digits=3)
    print("bert result:")
    print(bert_report)

    bert_graph_report = classification_report(df_test["label"], df_test["graph_pred"], digits=3)
    print("bert_graph result:")
    print(bert_graph_report)


    with open('./log.txt', 'a+', encoding='utf-8') as f:
        print("bert result:", file=f)
        print(bert_report, file=f)
        print("bert_graph result:", file=f)
        print(bert_graph_report, file=f)
# matrix = confusion_matrix(df_test["label"], df_test["pred"])
# print(matrix)
    


