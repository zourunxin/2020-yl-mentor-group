import sys
sys.path.append("../")
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import bert
import matplotlib.pyplot as plt
import time
import utils.CommonUtils as CommonUtils
import utils.NLPUtils as NLPUtils
from transformers import AutoTokenizer
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import FeatureExtractor.extractors as Extractors
import transformers
import pdb



# ========== parameters: ==========
maxlen = 128
hidden_size = 64
batch_size = 32
epochs = 100
bert_path = "prajjwal1/bert-tiny"
alpha = 3
wvdim = 64
lcm_stop = 0
use_textcnn = False
use_arti_feat = True
params_str = 'a=%s, wvdim=%s, lcm_stop=%s'%(alpha,wvdim,lcm_stop)


# ========== bert config: ==========
# for English, use bert_tiny:
# bert_type = 'bert'
# config_path = 'bert_weights/bert_tiny_uncased_L-2_H-128_A-2/bert_config.json'
# checkpoint_path = 'bert_weights/bert_tiny_uncased_L-2_H-128_A-2/bert_model.ckpt'
# vocab_path = 'bert_weights/bert_tiny_uncased_L-2_H-128_A-2/vocab.txt'

# for Chinese, use albert_tiny:
# bert_type = 'albert'
# config_path = '../bert_weights/albert_tiny_google_zh_489k/albert_config.json'
# checkpoint_path = '../bert_weights/albert_tiny_google_zh_489k/albert_model.ckpt'
# vocab_path = '../bert_weights/albert_tiny_google_zh_489k/vocab.txt'
for use_arti_feat in [True, False]:
    tokenizer = AutoTokenizer.from_pretrained(bert_path)

    def encode_onehot(label_num_map, labels):
        classes_dict = {label: np.identity(len(label_num_map))[label_num_map[label], :] for label in label_num_map}
        labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return labels_onehot

    # ========== dataset: ==========
    print("开始读取数据")
    df_data = pd.read_csv('../output/datasource_0205_class.csv')
    df_edges = pd.read_csv('../output/edges.csv')
    idx_name_map, name_idx_map = CommonUtils.get_idx_name_map(df_data["name"])
    num_label_map, label_num_map = CommonUtils.get_num_label_map(df_data["label"])
    num_classes = len(num_label_map)
    onehot_labels = encode_onehot(label_num_map, list(df_data["label"]))
    texts = list(df_data["text"].apply(lambda x: NLPUtils.preprocess_text(x)))
    tfidf_features = Extractors.tfidf_feat_extractor(texts, onehot_labels, feature_num=20)
    tfidf_features = np.array(tfidf_features)
    df_data["idx"] = range(len(df_data))
    df_train, df_test = train_test_split(df_data, test_size=0.3, random_state=666, stratify=df_data["label"])


    def tokenize(df):
        t = time.time()
        X_name = []
        X_input_ids = []
        X_token_type_ids = []
        X_attention_mask = []
        y = []
        i = 0
        for content,label,name in zip(list(df.text),list(df.label),list(df.name)):
            bert_input = tokenizer.encode_plus(content, 
                add_special_tokens = True, # add [CLS], [SEP]
                max_length = maxlen, # max length of the text that can go to BERT
                pad_to_max_length = True, # add [PAD] tokens
                return_attention_mask = True, # add attention mask to not focus on pad tokens
                truncation=True)
            X_input_ids.append(bert_input['input_ids'])
            X_token_type_ids.append(bert_input['token_type_ids'])
            X_attention_mask.append(bert_input['attention_mask'])
            y.append(label_num_map[label])
            X_name.append(name)
        X_input_ids = np.array(X_input_ids)
        X_token_type_ids = np.array(X_token_type_ids)
        X_attention_mask = np.array(X_attention_mask)
        y = np.array(y)
        print('tokenizing time cost:',time.time()-t,'s.')

        return X_input_ids, X_token_type_ids, X_attention_mask, y, X_name

    # ========== model traing: ==========
    old_list = []
    lcm_list = []

    X_input_ids_train, X_token_type_ids_train, X_attention_mask_train, y_train, name_train = tokenize(df_train)
    X_input_ids_test, X_token_type_ids_test, X_attention_mask_test, y_test, name_test = tokenize(df_test)



    X_tfidf_train = tfidf_features[df_train['idx'].values.tolist()]
    X_name_features_train = Extractors.name_feat_extractor(list(df_train["name"]))
    X_meta_train = Extractors.meta_feat_extractor(df_train, df_edges)
    # X_keyword_features_train = Extractors.keyword_feat_extractor(list(df_train["text"].apply(lambda x: NLPUtils.preprocess_text(x))))

    X_arti_feats_train = np.hstack((X_name_features_train, X_tfidf_train, X_meta_train))

    X_tfidf_test = tfidf_features[df_test['idx'].values.tolist()]
    X_name_features_test = Extractors.name_feat_extractor(list(df_test["name"]))
    X_meta_test = Extractors.meta_feat_extractor(df_test, df_edges)
    # X_keyword_features_test = Extractors.keyword_feat_extractor(list(df_test["text"].apply(lambda x: NLPUtils.preprocess_text(x))))
    X_arti_feats_test = np.hstack((X_name_features_test, X_tfidf_test, X_meta_test))

    data_package = [X_input_ids_train, X_token_type_ids_train, X_attention_mask_train, X_arti_feats_train, y_train, X_input_ids_test, X_token_type_ids_test, X_attention_mask_test, X_arti_feats_test, y_test]
    extra_feats_num = X_arti_feats_test.shape[1]
    model = bert.BERT_LCM(bert_path, maxlen, hidden_size, num_classes, alpha, extra_feats_num, wvdim, 
                        use_textcnn=False, use_arti_feat=True)
    train_score_list, val_socre_list, best_val_score, test_score = model.train_val(data_package, batch_size, epochs, lcm_stop, save_best=True)
    # plt.plot(train_score_list, label='train')
    # plt.plot(val_socre_list, label='val')
    # plt.title('BERT with LCM')
    # plt.legend()
    # plt.show()
    old_list.append(test_score)
    print('\n*** Orig BERT with LCM (%s) ***:'%params_str)
    print('test acc:', str(test_score))
    print('best val acc:', str(best_val_score))
    print('train acc list:\n', str(train_score_list))
    print('val acc list:\n', str(val_socre_list), '\n')


    name_pred_map = {}

    model_input = [X_input_ids_test, X_attention_mask_test, X_token_type_ids_test, X_arti_feats_test]
    output = model.predict(model_input)
    for i, name in enumerate(name_test):
        L_label = np.array(range(num_classes))
        # ↓ 这个输入维度搞了我一晚上，气死我了
        model_input = [np.array([X_input_ids_test[i]]), np.array([X_attention_mask_test[i]]), np.array([X_token_type_ids_test[i]]), np.array([X_arti_feats_test[i]])]
        pred_probs = output[i]
        name_pred_map[name] = num_label_map[np.argmax(pred_probs)]

    df_test["pred"] = df_test["name"].apply(lambda x: name_pred_map[x])
    df_test = df_test.loc[:,["name", "label", "pred", "summary", "description"]]
    df_test.to_csv("bert_lcm_result.csv", index=False)
    df_error = df_test.loc[df_test["label"] != df_test["pred"]]
    df_error.to_csv("bert_lcm_result_error.csv", index=False)

    report = classification_report(df_test["label"], df_test["pred"], digits=3)
    print(report)

    matrix = confusion_matrix(df_test["label"], df_test["pred"])
    print(matrix)
        

    with open('./log.txt', 'a+', encoding='utf-8') as f:
        print("prarm: {}".format(qqq), file=f)
        print(report, file=f)
        print(matrix, file=f)
