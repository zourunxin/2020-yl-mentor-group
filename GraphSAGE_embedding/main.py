#-*-coding:utf-8-*-
import pdb

import tensorflow as tf
import sys
sys.path.append("../")
tf.compat.v1.disable_v2_behavior()
import numpy as np
from sklearn.metrics import f1_score
from collections import defaultdict
import time
import random
import GraphSAGE_embedding.config as cfg
from GraphSAGE_embedding.aggregator import *
import networkx as nx
import itertools as it
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from utils.FileUtil import csv_reader
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
# tf.enable_eager_execution()

class graphsage():
    def __init__(self):
        self.cfg = cfg
        self.features = tf.Variable(self.cfg.features,dtype=tf.float32,trainable=False)     # tf.Variable 初始化变量
        if self.cfg.aggregator == 'mean':
            self.aggregator = mean_aggregator
        elif self.cfg.aggregator == 'pooling':
            self.aggregator = pooling_aggreagtor
        elif self.cfg.aggregator == 'lstm':
            self.aggregator = lstm_aggregator
        else:
            raise(Exception,"Invalid aggregator!")
        self.placeholders = self.build_placeholders()

    def build_placeholders(self):
        placeholders = {}
        if self.cfg.gcn:
            neigh_size = self.cfg.sample_num + 1
        else:
            neigh_size = self.cfg.sample_num
        placeholders['batchnodes']       = tf.compat.v1.placeholder(shape=(None),dtype=tf.int32)
        placeholders['samp_neighs_1st']  = tf.compat.v1.placeholder(shape=(None,neigh_size),dtype=tf.int32)
        if self.cfg.depth==2:
            placeholders['samp_neighs_2nd']  = tf.compat.v1.placeholder(shape=(None,neigh_size,neigh_size),dtype=tf.int32)
        if self.cfg.supervised:
            placeholders['labels'] = tf.compat.v1.placeholder(shape=(None),dtype=tf.int32)
        else:
            placeholders['input_1'] = tf.compat.v1.placeholder(shape=(None),dtype=tf.int32)
            placeholders['input_2'] = tf.compat.v1.placeholder(shape=(None),dtype=tf.int32)
            placeholders['input_3'] = tf.compat.v1.placeholder(shape=(None),dtype=tf.int32)
        return placeholders

    def construct_feed_dict_sup(self,nodes=None,samp_neighs_1st=None,samp_neighs_2nd=None,labels=None):
        feed_dict = {}
        feed_dict.update({self.placeholders['batchnodes']:nodes})
        feed_dict.update({self.placeholders['samp_neighs_1st']:samp_neighs_1st})
        feed_dict.update({self.placeholders['labels']:labels})
        if self.cfg.depth==2:
            feed_dict.update({self.placeholders['samp_neighs_2nd']:samp_neighs_2nd})
        return feed_dict

    def construct_feed_dict_unsup(self,nodes=None,samp_neighs_1st=None,samp_neighs_2nd=None,input_1=None,input_2=None,input_3=None):
        ###Note here labels are used for evaluate rather than training###
        feed_dict = {}
        feed_dict.update({self.placeholders['batchnodes']:nodes})
        feed_dict.update({self.placeholders['samp_neighs_1st']:samp_neighs_1st})
        feed_dict.update({self.placeholders['input_1']:input_1})
        feed_dict.update({self.placeholders['input_2']:input_2})
        feed_dict.update({self.placeholders['input_3']:input_3})
        if self.cfg.depth==2:
            feed_dict.update({self.placeholders['samp_neighs_2nd']:samp_neighs_2nd})
        return feed_dict

    def sample_neighs(self,nodes):
        _sample = np.random.choice
        neighs = [list(self.cfg.adj_lists[int(node)]) for node in nodes]
        samp_neighs = [list(_sample(neighs,self.cfg.sample_num,replace=False)) if len(neighs)>=self.cfg.sample_num else list(_sample(neighs,self.cfg.sample_num,replace=True)) for neighs in neighs]
        if self.cfg.gcn:
            samp_neighs = [samp_neigh+list([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        if self.cfg.aggregator=='lstm':
            # for lstm we need to shuffle the node order
            samp_neighs = [list(np.random.permutation(x)) for x in samp_neighs]
        return samp_neighs

    def forward(self):
        ### Here we set the aggregate depth as 2 ###
        if self.cfg.depth==2:
            agg_2nd = tf.map_fn(fn = lambda x:self.aggregator(tf.nn.embedding_lookup(self.features,x[0]),tf.nn.embedding_lookup(self.features,x[1]),self.cfg.dims,'agg_2nd'),
                elems=(self.placeholders['samp_neighs_1st'],self.placeholders['samp_neighs_2nd']),dtype=tf.float32)
            node_features = self.aggregator(tf.nn.embedding_lookup(self.features,self.placeholders['batchnodes']),tf.nn.embedding_lookup(self.features,self.placeholders['samp_neighs_1st']),self.cfg.dims,'agg_2nd')
            agg_1st = self.aggregator(node_features,agg_2nd,self.cfg.dims,'agg_1st')
        else:
            agg_1st = self.aggregator(tf.nn.embedding_lookup(self.features,self.placeholders['batchnodes']),tf.nn.embedding_lookup(self.features,self.placeholders['samp_neighs_1st']),
                self.cfg.dims,'agg_1st')
        return agg_1st

    def sess(self):
        gpu_config = tf.compat.v1.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        sess = tf.compat.v1.InteractiveSession(config=gpu_config)
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        return sess

    def supervised(self,inputs,labels):
        preds = tf.compat.v1.layers.dense(inputs,units=self.cfg.num_classes,activation=None)
        labels = tf.one_hot(labels,depth=self.cfg.num_classes)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,logits=preds)
        accuray = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds,1),tf.argmax(labels,1)),tf.float32))
        return loss,accuray

    def random_walk(self,num_walks=1,walk_length=1):
        G = nx.Graph()
        node_map = self.cfg.node_map
        G.add_nodes_from(node_map.values())
        edges_reader = csv_reader('../output/edges.csv')
        for line in edges_reader:
            G.add_edge(node_map[line[0]], node_map[line[1]])
        nodes = list(G.nodes())
        degrees = [G.degree(x) for x in nodes]
        walk_pairs = []
        for n in nodes:
            if G.degree(n) == 0:
                continue
            for j in range(num_walks):
                current_n = n
                for k in range(walk_length+1):
                    neigs = list(G.neighbors(current_n))
                    if len(neigs)>0:
                        next_n = random.choice(neigs)
                    else:
                        break
                    if current_n != n:
                        walk_pairs.append((n,current_n))
                    current_n = next_n
        random.shuffle(walk_pairs)
        return walk_pairs,nodes,degrees

    def neg_sample(self, pos_nodes, nodes, p):
        sample_nodes = []
        while len(sample_nodes)<self.cfg.neg_num:
            x = np.random.choice(nodes,size=1,replace=False,p=p)[0]
            if (x not in pos_nodes) and (x not in sample_nodes):
                sample_nodes.append(x)
        return sample_nodes

    def unsupervised(self,input_1,input_2,input_3):
        ###for unsupervised training, we use the loss function like deepwalk###
        aff = tf.reduce_sum(tf.multiply(input_1, input_2), 1)
        neg_aff = tf.matmul(input_1, tf.transpose(input_3))
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(aff), logits=aff)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(neg_aff), logits=neg_aff)
        loss = tf.reduce_mean(true_xent) + tf.reduce_mean(negative_xent)
        # loss = loss / tf.cast(tf.shape(input_1)[0], tf.float32)
        return loss

    def exec(self):
        walk_pairs,nodes,degrees = self.random_walk()
        p = np.array(degrees)/sum(degrees)
        emb = self.forward()
        emb = tf.nn.l2_normalize(emb,1)
        input_1 = tf.nn.embedding_lookup(emb,self.placeholders['input_1'])
        input_2 = tf.nn.embedding_lookup(emb,self.placeholders['input_2'])
        input_3 = tf.nn.embedding_lookup(emb,self.placeholders['input_3'])
        loss = self.unsupervised(input_1,input_2,input_3)
        opt  = tf.compat.v1.train.GradientDescentOptimizer(self.cfg.lr).minimize(loss)
        sess = self.sess()
        for i in range(self.cfg.epochs):
            start = 0
            t = 0
            while start<len(walk_pairs):
                s = time.time()
                end = min(start+self.cfg.batchsize,len(walk_pairs))
                batchpairs = walk_pairs[start:end]
                input_1,input_2 = zip(*batchpairs)
                input_1 = list(input_1)
                input_2 = list(input_2)
                input_3 = self.neg_sample(input_2, nodes, p)
                unique_nodes = list(set(input_1+input_2+input_3))
                look_up = {x:i for i,x in enumerate(unique_nodes)}
                samp_neighs_1st = self.sample_neighs(unique_nodes)
                samp_neighs_2nd = [self.sample_neighs(neighs) for neighs in samp_neighs_1st]
                input_1 = [look_up[x] for x in input_1]
                input_2 = [look_up[x] for x in input_2]
                input_3 = [look_up[x] for x in input_3]
                feed_dict = self.construct_feed_dict_unsup(unique_nodes,samp_neighs_1st,samp_neighs_2nd,input_1,input_2,input_3)
                _,ls = sess.run([opt,loss],feed_dict=feed_dict)
                e = time.time()
                t = t + e - s
                print('\r Unsupervised Epoch = {:d} TrainLoss = {:.5f} Time = {:.3f}'.format(i+1,ls,t),end='\r')
                start = end
            print()
        ### test ###
        start = 0
        embedding = np.zeros((self.cfg.num_nodes,self.cfg.dims))
        while(start<self.cfg.num_nodes):
            end = min(start + self.cfg.batchsize,self.cfg.num_nodes)
            unique_nodes = list(range(start,end))
            samp_neighs_1st = self.sample_neighs(unique_nodes)
            samp_neighs_2nd = [self.sample_neighs(neighs) for neighs in samp_neighs_1st]
            x = sess.run(emb,feed_dict={
                self.placeholders['batchnodes']:unique_nodes,
                self.placeholders['samp_neighs_1st']:samp_neighs_1st,
                self.placeholders['samp_neighs_2nd']:samp_neighs_2nd
                })
            embedding[unique_nodes] = x
            start = end
        return embedding


if __name__ == '__main__':
    # 仅能用于全连通图，不能存在孤立节点
    sage = graphsage()
    embedding = sage.exec()
    print(embedding)