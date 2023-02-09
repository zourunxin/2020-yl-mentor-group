import sys 
sys.path.append("../")
import numpy as np
import keras
import tensorflow as tf
import time
from keras.models import Sequential,Model
from keras.layers import Input,Dense,LSTM,Embedding,Conv1D,MaxPooling1D
from keras.layers import Flatten,Dropout,Concatenate,Lambda,Multiply,Reshape,Dot,Bidirectional
import keras.backend as K
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from transformers import TFAutoModel
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from models.text_cnn import TextCNN
import pdb


class BERT_LCM:
    def __init__(self, bert_path ,max_length, hidden_size, num_classes, alpha, extra_feats_dim,
                wvdim=768, use_textcnn=True, use_arti_feat=True):
        self.num_classes = num_classes

        def lcm_loss(y_true,y_pred,alpha=alpha):
            pred_probs = y_pred[:,:num_classes]
            label_sim_dist = y_pred[:,num_classes:]
            simulated_y_true = K.softmax(label_sim_dist+alpha*y_true)
            loss1 = -K.categorical_crossentropy(simulated_y_true,simulated_y_true)
            loss2 = K.categorical_crossentropy(simulated_y_true,pred_probs)
            return loss1+loss2

        def ls_loss(y_true, y_pred, e=0.1):
            loss1 = K.categorical_crossentropy(y_true, y_pred)
            loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/num_classes, y_pred)
            return (1-e)*loss1 + e*loss2     

        # text_encoder:
        input_shape = (max_length, )
        input_ids = Input(input_shape, dtype=tf.int32)
        attention_mask = Input(input_shape, dtype=tf.int32)
        token_type_ids = Input(input_shape, dtype=tf.int32)
        extra_feats = Input((extra_feats_dim,), dtype=tf.float32)
        
        text_inputs = [input_ids, attention_mask, token_type_ids] # 构造 bert 输入
        bert = TFAutoModel.from_pretrained(bert_path, from_pt=True, output_hidden_states=True)
        bert_out = bert(text_inputs)
        sequence_output, pooler_output, hidden_states = bert_out[0], bert_out[1], bert_out[2] 
        # extra_emb = Dense(128,activation='tanh')(extra_feats)

        if use_textcnn:
            # 将每一层的第一个token(cls向量)提取出来，拼在一起当作textcnn的输入
            cls_embeddings = tf.expand_dims(hidden_states[1][:, 0, :], axis=1) # [bs, 1, hidden]
            for i in range(2, 9):
                cls_embeddings = Concatenate(axis=1)([cls_embeddings, tf.expand_dims(hidden_states[i][:, 0, :], axis=1)])
            textcnn = TextCNN(cls_embeddings.shape[1], num_features=cls_embeddings.shape[2], class_num=128).get_model()
            bert_output = textcnn(cls_embeddings)
        else:
            bert_output = pooler_output

        if use_arti_feat:
            text_emb = Concatenate()([bert_output, extra_feats])
        else:
            text_emb = bert_output

        text_emb = Dense(hidden_size,activation='relu')(text_emb)  # n * hidden_size
        dense_output = Dense(32,activation='relu')(text_emb)  # n * hidden_size
        
        pred_probs = Dense(num_classes,activation='softmax')(dense_output)  # n * num_classes
        self.basic_predictor = Model(text_inputs + [extra_feats], pred_probs)
        AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
        self.basic_predictor.compile(loss='categorical_crossentropy',optimizer=AdamLR(learning_rate=5e-6))

        # label_encoder:
        label_input = Input(shape=(num_classes,),name='label_input') # n * num_classes * num_classes
        label_emb = Embedding(num_classes,wvdim,input_length=num_classes ,name='label_emb1')(label_input) # n * num_classes * wvdim
        # label_emb = Bidirectional(LSTM(hidden_size,return_sequences=True),merge_mode='ave')(label_emb) # (n,d)
        label_emb = Dense(hidden_size,activation='tanh',name='label_emb2')(label_emb) # n * num_classes * hidden_size
                
        # similarity part:
        doc_product = Dot(axes=(2,1))([label_emb,text_emb]) # (num_classes,hidden_size) dot (hidden_size,1) --> (num_classes,1)
        label_sim_dict = Dense(num_classes,activation='softmax',name='label_sim_dict')(doc_product)

        # concat output:
        concat_output = Concatenate()([pred_probs, label_sim_dict])

        # compile；
        AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
        self.model = Model(text_inputs + [extra_feats] + [label_input], concat_output)
        self.model.compile(loss=lcm_loss, optimizer=AdamLR(learning_rate=5e-6))

    def lcm_evaluate(self,model,inputs,y_true):
        outputs = model.predict(inputs)
        pred_probs = outputs[:,:self.num_classes]
        predictions = np.argmax(pred_probs,axis=1)
        acc = round(accuracy_score(y_true,predictions),5)
        return acc

    def train_val(self, data_package, batch_size, epochs, lcm_stop=50,save_best=True):
        X_input_ids_train, X_token_type_ids_train, X_attention_mask_train, X_arti_feats_train, y_train, X_input_ids_test, X_token_type_ids_test, X_attention_mask_test, X_arti_feats_test, y_test = data_package
        best_val_score = 0
        test_score = 0
        train_score_list = []
        val_socre_list = []
        """实验说明：
        每一轮train完，在val上测试，记录其accuracy，
        每当val-acc达到新高，就立马在test上测试，得到test-acc，
        这样最终保留下来的test-acc就是在val上表现最好的模型在test上的accuracy
        """

        L_train = np.array([np.array(range(self.num_classes)) for i in range(len(y_train))])
        L_val = np.array([np.array(range(self.num_classes)) for i in range(len(y_test))])
        L_test = np.array([np.array(range(self.num_classes)) for i in range(len(y_test))])

        for i in range(epochs):
            t1 = time.time()
            if i < lcm_stop:
                self.model.fit([X_input_ids_train, X_attention_mask_train, X_token_type_ids_train, X_arti_feats_train, L_train], to_categorical(y_train), batch_size=batch_size, epochs=1)
                # record train set result:
                train_score = self.lcm_evaluate(self.model,[X_input_ids_train, X_attention_mask_train, X_token_type_ids_train, X_arti_feats_train, L_train], y_train)
                train_score_list.append(train_score)
                # validation:
                val_score = self.lcm_evaluate(self.model,[X_input_ids_test, X_attention_mask_test, X_token_type_ids_test, X_arti_feats_test, L_val], y_test)
                val_socre_list.append(val_score)
                t2 = time.time()
                print('(LCM)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), ' | train acc:', train_score, ' | val acc:',
                      val_score)
                # save best model according to validation & test result:
                if val_score > best_val_score:
                    best_val_score = val_score
                    print('Current Best model!', 'current epoch:', i + 1)
                    # test on best model:
                    # test_score = self.lcm_evaluate(self.model,[X_input_ids_test, X_attention_mask_test, X_token_type_ids_test, X_arti_feats_test, L_test],y_test)
                    # print('  Current Best model! Test score:', test_score)
                    if save_best:
                        self.model.save_weights('best_model_bert_lcm.h5')
                        print('best model saved!')
            else:
                self.basic_predictor.fit([X_input_ids_train, X_attention_mask_train, X_token_type_ids_train, X_arti_feats_train],to_categorical(y_train),batch_size=batch_size,epochs=1)
                # record train set result:
                pred_probs = self.basic_predictor.predict([X_input_ids_train, X_attention_mask_train, X_token_type_ids_train, X_arti_feats_train])
                predictions = np.argmax(pred_probs, axis=1)
                train_score = round(accuracy_score(y_train, predictions),5)
                train_score_list.append(train_score)
                # validation:
                pred_probs = self.basic_predictor.predict([X_input_ids_test, X_attention_mask_test, X_token_type_ids_test, X_arti_feats_test])
                predictions = np.argmax(pred_probs, axis=1)
                val_score = round(accuracy_score(y_test, predictions),5)
                val_socre_list.append(val_score)
                t2 = time.time()
                print('(LCM_stopped)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), ' | train acc:', train_score, ' | val acc:',
                      val_score)
                # save best model according to validation & test result:
                if val_score > best_val_score:
                    best_val_score = val_score
                    print('Current Best model!', 'current epoch:', i + 1)
                    # test on best model:
                    # pred_probs = self.basic_predictor.predict([X_input_ids_test, X_attention_mask_test, X_token_type_ids_test])
                    # predictions = np.argmax(pred_probs, axis=1)
                    # test_score = round(accuracy_score(y_test, predictions),5)
                    # print('  Current Best model! Test score:', test_score)
                    if save_best:
                        self.model.save_weights('best_model_bert_lcm.h5')
                        print('best model saved!')
        return train_score_list, val_socre_list, best_val_score, test_score

    def predict(self, bert_inputs):
        self.model.load_weights('./best_model_bert_lcm.h5')
        return self.basic_predictor.predict(bert_inputs, verbose=0)
