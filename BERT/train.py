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

def encode_examples(ds, limit=-1):
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    if (limit > 0):
        ds = ds.take(limit)
  
    for index, row in ds.iterrows():
        review = row["text"]
        label = row["y"]
        bert_input = convert_example_to_feature(review)
  
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])
    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)



def split_dataset(df, sample_size=100):
    train_data = pd.DataFrame(columns=['name', 'label', 'text', 'y'])
    val_data = pd.DataFrame(columns=['name', 'label', 'text', 'y'])
    test_data = pd.DataFrame(columns=['name', 'label', 'text', 'y'])  

    grouped_df = df.groupby('label')
    for group in grouped_df:
        g_data = shuffle(group[1])
        train_data = train_data.append(g_data[0:sample_size])
        val_data = val_data.append(g_data[sample_size:sample_size + 100])
        test_data = test_data.append(g_data[sample_size:])

    train_data = shuffle(train_data)
    val_data = shuffle(val_data)
    test_data = shuffle(test_data)
    print(train_data)
    print(test_data)

    return train_data, val_data, test_data

# parameters
data_path = "../output/datasource_1228.csv" # 数据路径
model_path = "bert-base-uncased" #模型路径，建议预先下载(https://huggingface.co/bert-base-chinese#)

max_length = 128
batch_size = 32
learning_rate = 2e-5
number_of_epochs = 1
num_classes = 4

# read data
df_raw = pd.read_csv(data_path)
# transfer label
df_label = pd.DataFrame({"label":["库","工具","服务","其它"],"y":list(range(4))})
df_raw = pd.merge(df_raw,df_label,on="label",how="left")


# split data
train_data, val_data, test_data = split_dataset(df_raw)


# tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)
# train dataset
ds_train_encoded = encode_examples(train_data).shuffle(10000).batch(batch_size)
# val dataset
ds_val_encoded = encode_examples(val_data).batch(batch_size)
# test dataset
ds_test_encoded = encode_examples(test_data).batch(batch_size)

# model initialization
model = TFBertForSequenceClassification.from_pretrained(model_path, num_labels=num_classes)

# optimizer Adam recommended
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,epsilon=1e-08, clipnorm=1)
# we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
# fit model
bert_history = model.fit(ds_train_encoded, epochs=number_of_epochs, validation_data=ds_val_encoded)
# evaluate test_set
# print("# evaluate test_set:",model.evaluate(ds_test_encoded))

predictions = model.predict(ds_test_encoded, batch_size = batch_size)
predictions = np.argmax(predictions.logits, axis=1)
print(predictions)
print(test_data["y"])

t = classification_report(list(test_data["y"]), predictions)
print(t)

test_data["y_predict"] = predictions

test_data.to_csv('../output/bert_result.csv')