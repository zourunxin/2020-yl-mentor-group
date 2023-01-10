import sys
sys.path.append("../")
import pandas as pd
import numpy as np
import utils.CommonUtils as CommonUtils
import utils.NLPUtils as NLPUtils
import FeatureExtractor.extractors as Extractors

from models.text_cnn import TextCNN
from keras.optimizers import Adam
from sklearn.metrics import classification_report

def encode_onehot(label_num_map, labels):
    classes_dict = {label: np.identity(len(label_num_map))[label_num_map[label], :] for label in label_num_map}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)

print("开始读取数据")
df_data = pd.read_csv('../output/datasource_1228_new.csv')
wv_model_dir = "../saved_models/word2vec_model_128d.model"

max_length = 128    # 文本最大长度
learning_rate = 0.0001
epochs = 200

idx_name_map, name_idx_map = CommonUtils.get_idx_name_map(df_data["name"])
num_label_map, label_num_map = CommonUtils.get_num_label_map(df_data["label"])
names = list(df_data["name"])
onehot_labels = encode_onehot(label_num_map, list(df_data["label"]))
texts = list(df_data["text"].apply(lambda x: NLPUtils.preprocess_text(x)))
texts = [text.split(" ") for text in texts]

features = Extractors.word2vec_2d_extractor(wv_model_dir, texts, padding=True, max_length=max_length)

print(features.shape)
x_train, x_val, x_test, y_train, y_val, y_test, idx_test = CommonUtils.get_sample_splits(features, onehot_labels, sample_size=100)

model = TextCNN(max_length, num_features=x_train.shape[2], class_num=4).get_model()

model.compile(optimizer=Adam(learning_rate),
              loss='categorical_crossentropy',
              weighted_metrics=['categorical_crossentropy', 'acc'])

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=epochs, verbose=1)

preds = model.predict(x_test, batch_size=32)
preds = np.argmax(preds, axis = 1)

print(preds)

df_result = df_data[["name", "label", "summary", "description"]]
df_result = df_result.loc[idx_test]
df_result["predict"] = [num_label_map[pred] for pred in preds]
df_error = df_result.loc[df_result["predict"] != df_result["label"]]
df_error = df_error.loc[:,["name", "label", "predict", "summary", "description"]]   #交换列的顺序

df_error.to_csv("../output/w2v_textcnn_result.csv", index=False)
report = classification_report(df_result["label"], df_result["predict"])
print(report)

CommonUtils.show_learning_curves(history)