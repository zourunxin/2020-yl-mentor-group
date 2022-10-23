import time
import torch
import argparse
import csv
import pandas as pd
import numpy as np
import scipy.sparse as sp

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import pickle

from sklearn.metrics import precision_recall_fscore_support

from sampler import Sampler_FastGCN
from layers import GraphConvolution
from util import get_batches, accuracy, prf
from util import cal_metrics, get_digit_label_map


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)

def convert_symmetric(X, sparse=True):
    if sparse:
        X += X.T - sp.diags(X.diagonal())
    else:
        X += X.T - np.diag(X.diagonal())
    return X

def nontuple_preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    ep = 1e-10
    r_inv = np.power(rowsum + ep, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def nontuple_preprocess_adj(adj):
    adj_normalized = normalize_adj(sp.eye(adj.shape[0]) + adj)
    # adj_normalized = sp.eye(adj.shape[0]) + normalize_adj(adj)
    return adj_normalized.tocsr()


def load_data(withs=False, cuda=False):
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_feat_lables = _load_data(strategy="sample", sample_size = 400)

    train_index = np.where(train_mask)[0]

    adj_train = adj[train_index, :][:, train_index]
    y_train = y_train[train_index]
    val_index = np.where(val_mask)[0]
    test_index = np.where(test_mask)[0]
    y_test = y_test[test_index]

    num_train = adj_train.shape[0]

    features = nontuple_preprocess_features(features).todense()

    # withs
    if cuda:
        device = torch.device("cuda")
        print("use cuda")
    else:
        device = torch.device("cpu")
    features = torch.FloatTensor(features).to(device)

    if withs:
        print('withs-------')
        tensor_adj = sparse_mx_to_torch_sparse_tensor(adj)
        agr_features = torch.mm(tensor_adj, features)
        features = torch.cat((features, agr_features), 1)
    train_features = features[train_index]

    norm_adj_train = nontuple_preprocess_adj(adj_train)
    norm_adj = nontuple_preprocess_adj(adj)

    return norm_adj, features, norm_adj_train, train_features, y_train, y_test, test_index, idx_feat_lables

def prf(output, labels):
    preds = output.max(1)[1].type_as(labels)
    p, r, f, s = precision_recall_fscore_support(labels, preds)
    return p[0], r[0], f[0]






def _load_data(strategy="all", sample_size = 100):
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

    y_train = np.zeros(onehot_labels.shape, dtype=np.int32)
    y_val = np.zeros(onehot_labels.shape, dtype=np.int32)
    y_test = np.zeros(onehot_labels.shape, dtype=np.int32)
    y_train[idx_train] = onehot_labels[idx_train]
    y_val[idx_val] = onehot_labels[idx_val]
    y_test[idx_test] = onehot_labels[idx_test]
    train_mask = sample_mask(idx_train, onehot_labels.shape[0])
    val_mask = sample_mask(idx_val, onehot_labels.shape[0])
    test_mask = sample_mask(idx_test, onehot_labels.shape[0])

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_feat_lables




class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, sampler, skip=False):
        super().__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass, skip)
        self.dropout = dropout
        self.sampler = sampler
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self, x, adj):
        outputs1 = F.relu(self.gc1(x, adj[0], x))
        outputs1 = F.dropout(outputs1, self.dropout, training=self.training)
        outputs2 = self.gc2(outputs1, adj[1], x)
        return F.log_softmax(outputs2, dim=1)
        # return self.out_softmax(outputs2)

    def sampling(self, *args, **kwargs):
        return self.sampler.sampling(*args, **kwargs)

    def embedding(self, x, adj):
        outputs1 = F.relu(self.gc1(x, adj[0], x))
        outputs1 = F.dropout(outputs1, self.dropout, training=self.training)
        return F.log_softmax(outputs1, dim=1)

def aggr_feats(node_list, adj, features):
    node_adj = adj[node_list, :]
    return torch.mm(sparse_mx_to_torch_sparse_tensor(node_adj), features)

def train(train_ind, train_labels, batch_size, train_times, adj, features, extend=False):
    t = time.time()
    model.train()
    for epoch in range(train_times):
        for batch_inds, batch_labels in get_batches(train_ind,
                                                    train_labels,
                                                    batch_size):
            sampled_feats, sampled_adjs, sampled_nodes, var_loss = model.sampling(
                batch_inds)
            if extend:
                aggred_feats = aggr_feats(sampled_nodes, adj, features)
                sampled_feats = aggred_feats

            optimizer.zero_grad()
            output = model(sampled_feats, sampled_adjs)
            loss_train = loss_fn(output, batch_labels) + 0.5 * var_loss
            acc_train = accuracy(output, batch_labels)
            loss_train.backward()
            optimizer.step()
    # just return the train loss of the last train epoch
    return loss_train.item(), acc_train.item(), time.time() - t

def test(test_adj, test_feats, test_labels, epoch, embed=False):
    t = time.time()
    model.eval()
    outputs = model(test_feats, test_adj)
    loss_test = loss_fn(outputs, test_labels)
    acc_test = accuracy(outputs, test_labels)
    if embed:
        embedding = model.embedding(test_feats, test_adj)
        print('embedding', embedding.shape)
        embedding_file = open('data/embedding_file','wb')
        pickle.dump(embedding, embedding_file)
        embedding_file.close()



    p, r, f = prf(outputs, test_labels)
    return loss_test.item(), acc_test.item(), time.time() - t, p, r, f


def predict(adj,features):
    output = model(features, adj)
    predictions = np.argmax(output.detach().numpy(), axis=1)
    df_node = pd.read_csv("../output/nodes.csv")
    nodes = np.array(df_node)

    label_list = idx_feat_lables[:, 1]
    nodes_true_prediction = np.hstack((nodes, label_list[:, np.newaxis].astype(np.int32), predictions[:, np.newaxis]))
    result = nodes_true_prediction.tolist()

    predict_map = {}
    actual_map = {}

    for res in result:
        predict_map[res[0]] = res[3]
        actual_map[res[0]] = res[2]

    metrics = cal_metrics(predict_map, actual_map)

    with open("../output/FASTGCN_statistic.csv", "w", newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["label", "count", "precision", "recall", "f1-score"])
        writer.writerows(metrics)

    digit_label_map = get_digit_label_map()
    result = [[r[0], r[1], digit_label_map[r[2]], digit_label_map[r[3]]] for r in result]

    with open("../output/FASTGCN_result.csv", "w", newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["idx", "name", "y_true", "y_predict"])
        writer.writerows(result)

def get_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--withs', action='store_true', default=False,
                        help='extend s')
    parser.add_argument('--embed', action='store_true', default=False,
                        help='embed')
    parser.add_argument('--skip', action='store_true', default=False,
                        help='extend skip')
    parser.add_argument('--extend', action='store_true', default=False,
                        help='extend')
    parser.add_argument('--dataset', type=str, default='cora',
                        help='dataset name.')
    # model can be "Fast" or "AS"
    parser.add_argument('--model', type=str, default='Fast',
                        help='model name.')
    parser.add_argument('--test_gap', type=int, default=10,
                        help='the train epochs between two test')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batchsize', type=int, default=256,
                        help='batchsize for train')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args




if __name__ == '__main__':

    args = get_args()
    # load data
    adj, features, adj_train, train_features, y_train, y_test, test_index, idx_feat_lables = load_data()

    layer_sizes = [128, 128]
    input_dim = features.shape[1]
    train_nums = adj_train.shape[0]
    test_gap = args.test_gap
    nclass = y_train.shape[1]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # set device
    if args.cuda:
        device = torch.device("cuda")
        print("use cuda")
    else:
        device = torch.device("cpu")

    # data for train and test
    # features = torch.FloatTensor(features).to(device)
    train_features = torch.FloatTensor(train_features).to(device)
    y_train = torch.LongTensor(y_train).to(device).max(1)[1]

    test_adj = [adj, adj[test_index, :]]
    test_feats = features
    test_labels = y_test
    test_adj = [sparse_mx_to_torch_sparse_tensor(cur_adj).to(device)
                for cur_adj in test_adj]
    test_labels = torch.LongTensor(test_labels).to(device).max(1)[1]

    sampler = Sampler_FastGCN(None, train_features, adj_train,
                              input_dim=input_dim,
                              layer_sizes=layer_sizes,
                              device=device)

    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=nclass,
                dropout=args.dropout,
                sampler=sampler,
                skip=args.skip).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = F.nll_loss

    start_time = time.time()

    for epochs in range(0, args.epochs // test_gap):
        train_loss, train_acc, train_time = train(np.arange(train_nums),
                                                  y_train,
                                                  args.batchsize,
                                                  test_gap,
                                                  adj,
                                                  features,
                                                  args.extend
                                                  )
        test_loss, test_acc, test_time, p, r, f = test(test_adj,
                                                       test_feats,
                                                       test_labels,
                                                       args.epochs,
                                                       args.embed)
        print(f"epchs:{epochs * test_gap}~{(epochs + 1) * test_gap - 1} "
              f"train_loss: {train_loss:.3f}, "
              f"train_acc: {train_acc:.3f}, "
              f"train_times: {train_time:.3f}s "
              f"test_loss: {test_loss:.3f}, "
              f"test_acc: {test_acc:.3f}, "
              f"test_times: {test_time:.3f}s, "
              f"precision: {p:.3f}, "
              f"recall: {r:.3f}, "
              f"f1_score: {f:.3f}, ")

    print("ALL TIME:", time.time() - start_time)

    predict_adj = [adj, adj[: , :]]

    predict_adj = [sparse_mx_to_torch_sparse_tensor(cur_adj).to(device)
                for cur_adj in predict_adj]
    predict(predict_adj, features)


