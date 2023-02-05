import sys
sys.path.append("../")
import logging
logging.basicConfig(level=logging.ERROR)
# from transformers import TFBertPreTrainedModel,TFBertMainLayer,BertTokenizer
from transformers import TFBertForSequenceClassification,BertTokenizer
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
from sklearn.metrics import classification_report
import utils.CommonUtils as CommonUtils
import utils.NLPUtils as NLPUtils
from keras.models import load_model

def convert_example_to_feature(review):
      
  # combine step for tokenization, WordPiece vector mapping, adding special tokens as well as truncating reviews longer than the max length
	return tokenizer.encode_plus(review, 
	            add_special_tokens = True, # add [CLS], [SEP]
	            max_length = max_length, # max length of the text that can go to BERT
	            pad_to_max_length = True, # add [PAD] tokens
	            return_attention_mask = True, # add attention mask to not focus on pad tokens
		    truncation=True
	          )
# map to the expected input to TFBertForSequenceClassification, see here 
def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
    return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }, label

def encode_examples(_x, _y):
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []

    for i, text in enumerate(_x):
        review = _x[i]
        label = _y[i]
        bert_input = convert_example_to_feature(review)
  
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([np.argmax(label)])
    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)

def encode_onehot(label_num_map, labels):
    classes_dict = {label: np.identity(len(label_num_map))[label_num_map[label], :] for label in label_num_map}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


print("开始读取数据")
df_data = pd.read_csv('../output/datasource_0117_class.csv')
model_path = "bert-base-uncased" #模型路径，建议预先下载(https://huggingface.co/bert-base-chinese#)

idx_name_map, name_idx_map = CommonUtils.get_idx_name_map(df_data["name"])
num_label_map, label_num_map = CommonUtils.get_num_label_map(df_data["label"])
onehot_labels = encode_onehot(label_num_map, list(df_data["label"]))
texts = list(df_data["text"])
x_train, x_val, x_test, y_train, y_val, y_test, idx_train, idx_test = CommonUtils.get_sample_splits(texts, onehot_labels, sample_size=100)


max_length = 64
batch_size = 32
learning_rate = 1e-5
number_of_epochs = 4
num_classes = 4

model = TFBertForSequenceClassification.from_pretrained(model_path, num_labels=num_classes)
model.load_weights('bert_weights.h5')

# tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)
ds_test_encoded = encode_examples(x_test, y_test).batch(batch_size)

def predictor(texts):
    ds_test_encoded = encode_examples(x_test, y_test).batch(batch_size)


predictions = model.predict(ds_test_encoded, batch_size = batch_size)
predictions = np.argmax(predictions.logits, axis=1)

df_result = df_data[["name", "label", "summary", "description"]]
df_train = df_result.loc[idx_train]
df_result = df_result.loc[idx_test]
df_result["predict"] = [num_label_map[pred] for pred in predictions]
df_error = df_result.loc[df_result["predict"] != df_result["label"]]
df_error = df_error.loc[:,["name", "label", "predict", "summary", "description"]]   #交换列的顺序
df_result =df_result.loc[:,["name", "label", "predict", "summary", "description"]] 
df_error.to_csv("bert_result_error.csv", index=False)
df_result.to_csv("bert_result.csv", index=False)
df_train.to_csv("bert_trainset.csv", index=False)
report = classification_report(df_result["label"], df_result["predict"])
print(report)