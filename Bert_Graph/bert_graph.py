import numpy as np
import keras
import tensorflow as tf
import time
import pdb
from keras.callbacks import ModelCheckpoint
# from tensorflow.python.keras.layers import Input, Dense, Dropout, Layer, LSTM
# from tensorflow.python.keras.models import Model
from keras.regularizers import l2
from keras.initializers import glorot_uniform, Zeros
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Embedding, Conv1D, MaxPooling1D, Dropout, Layer
from keras.layers import Flatten,Dropout,Concatenate,Lambda,Multiply,Reshape,Dot,Bidirectional
# from tensorflow.python.keras.optimizers import adam_v2
import keras.backend as K
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from transformers import TFAutoModel
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class MeanAggregator(Layer):
    
    def __init__(self, units, input_dim, neigh_max, concat=True, dropout_rate=0.0, activation=tf.nn.relu, l2_reg=0,
                 use_bias=False,
                 seed=1024, **kwargs):
        super(MeanAggregator, self).__init__()
        self.units = units
        self.neigh_max = neigh_max
        self.concat = concat
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_bias = use_bias
        self.activation = activation
        self.seed = seed
        self.input_dim = input_dim

    def build(self, input_shapes):
        self.neigh_weights = self.add_weight(shape=(self.input_dim, self.units),
                                             initializer=glorot_uniform(
                                                 seed=self.seed),
                                             regularizer=l2(self.l2_reg),
                                             name="neigh_weights")
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units), initializer=Zeros(),
                                        name='bias_weight')

        self.dropout = Dropout(self.dropout_rate)
        self.built = True

    def call(self, inputs, training=None):
        features, node, neighbours = inputs

        node_feat = tf.nn.embedding_lookup(features, node)
        neigh_feat = tf.nn.embedding_lookup(features, neighbours)

        node_feat = self.dropout(node_feat, training=training)
        neigh_feat = self.dropout(neigh_feat, training=training)

        concat_feat = tf.concat([neigh_feat, node_feat], axis=1)
        concat_mean = tf.reduce_mean(concat_feat, axis=1, keepdims=False)

        output = tf.matmul(concat_mean, self.neigh_weights)
        if self.use_bias:
            output += self.bias
        if self.activation:
            output = self.activation(output)

        # output = tf.nn.l2_normalize(output,dim=-1)
        output._uses_learning_phase = True

        return output

    def get_config(self):
        config = {'units': self.units,
                  'concat': self.concat,
                  'seed': self.seed
                  }

        base_config = super(MeanAggregator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PoolingAggregator(Layer):

    def __init__(self, units, input_dim, neigh_max, aggregator='meanpooling', concat=True,
                 dropout_rate=0.0,
                 activation=tf.nn.relu, l2_reg=0, use_bias=False,
                 seed=1024, ):
        super(PoolingAggregator, self).__init__()
        self.output_dim = units
        self.input_dim = input_dim
        self.concat = concat
        self.pooling = aggregator
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_bias = use_bias
        self.activation = activation
        self.neigh_max = neigh_max
        self.seed = seed

        # if neigh_input_dim is None:

    def build(self, input_shapes):

        self.dense_layers = [Dense(
            self.input_dim, activation='relu', use_bias=True, kernel_regularizer=l2(self.l2_reg))]

        self.neigh_weights = self.add_weight(
            shape=(self.input_dim * 2, self.output_dim),
            initializer=glorot_uniform(
                seed=self.seed),
            regularizer=l2(self.l2_reg),

            name="neigh_weights")

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=Zeros(),
                                        name='bias_weight')

        self.built = True

    def call(self, inputs, mask=None):

        features, node, neighbours = inputs

        node_feat = tf.nn.embedding_lookup(features, node)
        neigh_feat = tf.nn.embedding_lookup(features, neighbours)

        dims = tf.shape(neigh_feat)
        batch_size = dims[0]
        num_neighbors = dims[1]
        h_reshaped = tf.reshape(
            neigh_feat, (batch_size * num_neighbors, self.input_dim))

        for l in self.dense_layers:
            h_reshaped = l(h_reshaped)
        neigh_feat = tf.reshape(
            h_reshaped, (batch_size, num_neighbors, int(h_reshaped.shape[-1])))

        if self.pooling == "meanpooling":
            neigh_feat = tf.reduce_mean(neigh_feat, axis=1, keep_dims=False)
        else:
            neigh_feat = tf.reduce_max(neigh_feat, axis=1)

        output = tf.concat(
            [tf.squeeze(node_feat, axis=1), neigh_feat], axis=-1)

        output = tf.matmul(output, self.neigh_weights)
        if self.use_bias:
            output += self.bias
        if self.activation:
            output = self.activation(output)

        # output = tf.nn.l2_normalize(output, dim=-1)

        return output

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'concat': self.concat
                  }

        base_config = super(PoolingAggregator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class BERT_GraphSAGE:
    def __init__(self, bert_path, max_length, hidden_size, num_classes, neighbor_num, n_hidden,
                 with_tfidf=True, tfidf_features=500,
                 end2end=False,
                 use_bias=True, aggregator_type="pooling",
                 dropout_rate=0.0, l2_reg=0, activation=tf.nn.relu):
        self.num_classes = num_classes
        self.end2end = end2end


        def bert_loss(y_true, y_pred):
            pred_probs = y_pred[:,:num_classes]
            return K.categorical_crossentropy(tf.cast(y_true, dtype=tf.float32), pred_probs)


        if not end2end:
            # 任务1：bert finetune
            input_shape = (max_length, )
            input_ids = Input(input_shape, dtype=tf.int32)
            attention_mask = Input(input_shape, dtype=tf.int32)
            token_type_ids = Input(input_shape, dtype=tf.int32)
            text_inputs = [input_ids, attention_mask, token_type_ids] # 构造 bert 输入
            bert = TFAutoModel.from_pretrained(bert_path, from_pt=True)
            bert_out = bert(text_inputs)
            sequence_output, pooler_output = bert_out[0], bert_out[1] # 取出 [cls] 向量表示  n * bert_out_dim
            text_emb = Dense(hidden_size,activation='tanh')(pooler_output)  # n * hidden_size
            bert_pred_probs = Dense(num_classes, activation='softmax')(text_emb)  # n * num_classes
            bert_concat_output = Concatenate()([bert_pred_probs, text_emb])
            
            # 任务2：GraphSAGE 下游任务
            feature_dim = (hidden_size + tfidf_features) if with_tfidf else hidden_size
            graph_features = Input(shape=(hidden_size,))
            tfidf_features = Input(shape=(tfidf_features,))
            node_input = Input(shape=(1,), dtype=tf.int32)
            neighbor_input = [Input(shape=(l,), dtype=tf.int32) for l in neighbor_num]
            graph_input_list = [graph_features, tfidf_features, node_input] + neighbor_input
            if aggregator_type == 'mean':
                aggregator = MeanAggregator
            else:
                aggregator = PoolingAggregator
            if with_tfidf:
                h = Concatenate()([graph_features, tfidf_features])
            else:
                h = graph_features
            for i in range(0, len(neighbor_num)):
                if i > 0:
                    feature_dim = n_hidden
                if i == len(neighbor_num) - 1:
                    activation = tf.nn.softmax
                    n_hidden = num_classes
                h = aggregator(units=n_hidden, input_dim=feature_dim, activation=activation, l2_reg=l2_reg, use_bias=use_bias,
                                dropout_rate=dropout_rate, neigh_max=neighbor_num[i], aggregator=aggregator_type)(
                    [h, node_input, neighbor_input[i]])

            output = h

            self.bert_model = Model(text_inputs, bert_concat_output)
            AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
            self.bert_model.compile(loss=bert_loss, optimizer=AdamLR(learning_rate=1e-5, lr_schedule={1:1, 50:0.1, 100:0.001}))

            self.graph_model = Model(graph_input_list, outputs=output)
            self.graph_model.compile(optimizer=Adam(5e-4), loss='categorical_crossentropy', weighted_metrics=['categorical_crossentropy', 'acc'])

        else:
            # 任务1：bert finetune
            input_shape = (max_length, )
            input_ids = Input(input_shape, dtype=tf.int32)
            attention_mask = Input(input_shape, dtype=tf.int32)
            token_type_ids = Input(input_shape, dtype=tf.int32)
            text_inputs = [input_ids, attention_mask, token_type_ids] # 构造 bert 输入
            bert = TFAutoModel.from_pretrained(bert_path, from_pt=True)
            bert_out = bert(text_inputs)
            sequence_output, pooler_output = bert_out[0], bert_out[1] # 取出 [cls] 向量表示  n * bert_out_dim
            text_emb = Dense(hidden_size,activation='tanh')(pooler_output)  # n * hidden_size
            # text_emb = pooler_output  # n * hidden_size

            # 任务2：GraphSAGE 下游任务
            feature_dim = (hidden_size + tfidf_features) if with_tfidf else hidden_size
            graph_features = text_emb
            tfidf_features = Input(shape=(tfidf_features,))
            node_input = Input(shape=(1,), dtype=tf.int32)
            neighbor_input = [Input(shape=(l,), dtype=tf.int32) for l in neighbor_num]
            graph_input_list = [tfidf_features, node_input] + neighbor_input
            
            if aggregator_type == 'mean':
                aggregator = MeanAggregator
            else:
                aggregator = PoolingAggregator
            if with_tfidf:
                h = Concatenate()([graph_features, tfidf_features])
            else:
                h = graph_features
            for i in range(0, len(neighbor_num)):
                if i > 0:
                    feature_dim = n_hidden
                if i == len(neighbor_num) - 1:
                    activation = tf.nn.softmax
                    n_hidden = num_classes
                h = aggregator(units=n_hidden, input_dim=feature_dim, activation=activation, l2_reg=l2_reg, use_bias=use_bias,
                                dropout_rate=dropout_rate, neigh_max=neighbor_num[i], aggregator=aggregator_type)(
                    [h, node_input, neighbor_input[i]])

            output = h
            all_input_list = text_inputs + graph_input_list       
            AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
            self.all_model = Model(all_input_list, outputs=output)
            self.all_model.compile(loss='categorical_crossentropy', optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1:1, 200:0.1, 250:0.01, 300:0.001}), weighted_metrics=['categorical_crossentropy', 'acc'])
    

    def evaluate_bert(self, bert_model, inputs, y_true, mask):
        idxs = [m[0] for m in np.argwhere(mask)]
        inputs = [inp[idxs] for inp in inputs]
        y_true = y_true[idxs]
        outputs = bert_model.predict(inputs)
        pred_probs = outputs[:, :self.num_classes]
        predictions = np.argmax(pred_probs,axis=1)
        y_true = np.argmax(y_true, axis=1)
        acc = round(accuracy_score(y_true,predictions),5)
        return acc
    
    def evaluate_all(self, all_model, inputs, y_true, mask, batch_size):
        idxs = [m[0] for m in np.argwhere(mask)]
        # inputs = [inp[idxs] for inp in inputs]
        y_true = y_true[idxs]
        pred_probs = all_model.predict(inputs, batch_size=batch_size)
        pred_probs = pred_probs[idxs]
        predictions = np.argmax(pred_probs,axis=1)
        y_true = np.argmax(y_true, axis=1)
        acc = round(accuracy_score(y_true,predictions),5)
        return acc

    def predict_bert(self, model_input):
        self.bert_model.load_weights('./best_model_bert.h5')
        return self.bert_model.predict(model_input, verbose=1)

    def predict_graph(self, model_input, batch_size):
        self.graph_model.load_weights('./best_model_graphsage.h5')
        return self.graph_model.predict(model_input, verbose=1, batch_size=batch_size)

    def predict_all(self, model_input, batch_size):
        self.all_model.load_weights('./best_all_model.h5')
        return self.all_model.predict(model_input, verbose=1, batch_size=batch_size)

    def train_val(self, model_input, y, mask, graph_batch_size, bert_epochs, graph_epochs, all_epochs, save_best=True, mode="both"):
        y_train, y_test = y
        train_mask, test_mask = mask
        best_val_score = 0
        test_score = 0
        bert_train_score_list = []
        bert_val_socre_list = []
        all_train_score_list = []
        all_val_socre_list = []
        # return
        bert_prob, graph_prob, all_prob, bert_history, graph_history, all_history = [0, 0, 0, 0, 0, 0]

        """实验说明：
        每一轮train完，在val上测试，记录其accuracy，
        每当val-acc达到新高，就立马在test上测试，得到test-acc，
        这样最终保留下来的test-acc就是在val上表现最好的模型在test上的accuracy
        """

        if not self.end2end:
            bert_input = model_input[:3]
            if mode == "both" or mode == "bert":
                # bert
                for i in range(bert_epochs):
                    t1 = time.time()
                    self.bert_model.fit(bert_input, y_train, sample_weight=train_mask, batch_size=32, epochs=1, verbose=1)
                    # record train set result:
                    train_score = self.evaluate_bert(self.bert_model, bert_input, y_train, train_mask)
                    bert_train_score_list.append(train_score)
                    # validation:
                    val_score = self.evaluate_bert(self.bert_model, bert_input, y_test, test_mask)
                    bert_val_socre_list.append(val_score)
                    t2 = time.time()
                    print('(bert)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), ' | train acc:', train_score, ' | val acc:',
                            val_score)
                    # save best model according to validation & test result:
                    if val_score > best_val_score:
                        best_val_score = val_score
                        print('Current Best bert model!', 'current epoch:', i + 1)
                        # test on best model:
                        # test_score = self.evaluate_bert(self.bert_model, bert_input, y_test, test_mask)
                        # print('  Current Best model! Test score:', test_score)
                        if save_best:
                            self.bert_model.save_weights('best_model_bert.h5')
                            print('best bert model saved!')
            
            bert_predictions = self.predict_bert(bert_input)
            bert_prob = bert_predictions[:,:self.num_classes]

            if mode == "both" or mode == "graph":
                # graph
                graph_feats = bert_predictions[:,self.num_classes:]
                
                # # 用 rf 测试一下 embedding 出来的向量怎么样
                # idx_train = [m[0] for m in np.argwhere(train_mask)]
                # x_train = graph_feats[idx_train]
                # y_train = y_train[idx_train]
                # idx_test = [m[0] for m in np.argwhere(test_mask)]
                # x_test = graph_feats[idx_test]
                # y_test = y_test[idx_test]
                # rfc = RandomForestClassifier(random_state=0)
                # rfc = rfc.fit(x_train, np.argmax(y_train, axis=1))
                # pred = rfc.predict(x_test)
                # print("randomForestClassifier result:")
                # print(classification_report(pred, np.argmax(y_test, axis=1)))

                graph_input = [graph_feats] + model_input[3:]
                val_data = (graph_input, y_test, test_mask)
                checkpoint = ModelCheckpoint('best_model_graphsage.h5', monitor='val_acc', verbose=1, save_weights_only=True, save_best_only=True, mode='max', period=1)
                graph_history = self.graph_model.fit(graph_input, y_train, validation_data=val_data, sample_weight=train_mask, batch_size=graph_batch_size,
                epochs=graph_epochs, verbose=2, callbacks=[checkpoint], shuffle=False)
                graph_prob = self.predict_graph(graph_input, batch_size=graph_batch_size)

            bert_history = {"train_acc": bert_train_score_list, "val_acc": bert_val_socre_list}
        
        # end2end
        else:
            for i in range(all_epochs):
                t1 = time.time()
                self.all_model.fit(model_input, y_train, sample_weight=train_mask, batch_size=graph_batch_size, epochs=1, shuffle=False, verbose=1)
                # record train set result:
                train_score = self.evaluate_all(self.all_model, model_input, y_train, train_mask, batch_size=graph_batch_size)
                all_train_score_list.append(train_score)
                # validation:
                val_score = self.evaluate_all(self.all_model, model_input, y_test, test_mask, batch_size=graph_batch_size)
                all_val_socre_list.append(val_score)
                t2 = time.time()
                print('(all)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), ' | train acc:', train_score, ' | val acc:',
                        val_score)
                # save best model according to validation & test result:
                if val_score > best_val_score:
                    best_val_score = val_score
                    print('Current Best all model!', 'current epoch:', i + 1)
                    # test on best model:
                    # test_score = self.evaluate_bert(self.bert_model, bert_input, y_test, test_mask)
                    # print('  Current Best model! Test score:', test_score)
                    if save_best:
                        self.all_model.save_weights('best_all_model.h5')
                        print('best all model saved!')
        
            all_prob = self.predict_all(model_input, batch_size=graph_batch_size)
            all_history = {"train_acc": all_train_score_list, "val_acc": all_val_socre_list}

        return bert_prob, graph_prob, all_prob, bert_history, graph_history, all_history


