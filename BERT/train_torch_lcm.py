import sys
sys.path.append("../")
from transformers import BertModel,BertTokenizer
import pandas as pd
import utils.CommonUtils as CommonUtils
import torch
import numpy as np

from torch.optim import Adam
from tqdm import tqdm
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical

BERT_PATH = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
bert = BertModel.from_pretrained(BERT_PATH)

print("开始读取数据")
df_data = pd.read_csv('../output/datasource_0117_class.csv')
df_data = df_data.sample(200)
idx_name_map, name_idx_map = CommonUtils.get_idx_name_map(df_data["name"])
num_label_map, label_num_map = CommonUtils.get_num_label_map(df_data["label"])
num_classes = len(num_label_map)

def encode_onehot(label_num_map, labels):
    classes_dict = {label: np.identity(len(label_num_map))[label_num_map[label], :] for label in label_num_map}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.names = [name for name in df["name"]]
        # self.labels = [label_num_map[label] for label in ]
        self.labels = encode_onehot(label_num_map, df['label'])
        self.texts = [tokenizer(text, 
                                padding='max_length',
                                max_length = 128,
                                truncation=True,
                                return_tensors="pt")
                      for text in df['text']]
        self.c_labels = np.array([np.array(range(len(num_label_map))) for i in range(len(self.labels))])

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]
    
    def get_batch_names(self, idx):
        return self.names[idx]

    def get_batch_c_labels(self, idx):
        return self.c_labels[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        batch_name = self.get_batch_names(idx)
        batch_c_labels = self.get_batch_c_labels(idx)
        return batch_texts, batch_c_labels, batch_y, batch_name



class BertClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH)
        self.num_classes = num_classes

    def forward(self, input_id, mask, label_input):
        # text_encoder:
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False) # 768
        dropout_output = nn.Dropout(0.5)(pooled_output)
        text_emb = nn.Linear(768, 64)(dropout_output) # 64
        pred_probs = nn.Linear(64, self.num_classes)(text_emb)
        pred_probs = nn.Softmax()(pred_probs)
        text_emb = text_emb.reshape(text_emb.shape[0],text_emb.shape[1], 1) # 1 * 64
        # print("text_emb: ", text_emb.shape)  
        # label_encoder:
        label_emb = nn.Embedding(self.num_classes, 256)(label_input) # num_classes * 256
        label_emb = nn.Linear(256, 64)(label_emb) # num_classes * 64
        label_emb = nn.Tanh()(label_emb)  # num_classes * 64
        # print("label_emb: ", label_emb.shape)  

        # similarity part:
        label_sim_dict = torch.matmul(label_emb, text_emb).squeeze(-1) # (num_classes,64) dot (64,1) --> (num_classes,1)
        # print("label_sim_dict: ", label_sim_dict.shape)  
        label_sim_dict = nn.Softmax()(label_sim_dict)
        # concat output:
        # print("label_sim_dict: ", label_sim_dict.shape)
        # print("pred_probs: ", pred_probs.shape)
        concat_output = torch.cat((pred_probs, label_sim_dict), dim=1)
        # print("concat_output: ", concat_output.shape)

        return concat_output

def lcm_loss(y_pred, y_true, alpha=3):
    def pt_categorical_crossentropy(pred, label):
        """
        使用pytorch 来实现 categorical_crossentropy
        """
        # print(-label * torch.log(pred))
        return torch.sum(-label * torch.log(pred))

    pred_probs = y_pred[:,:num_classes]
    label_sim_dist = y_pred[:,num_classes:]
    # print("pred_probs: ", pred_probs.shape)
    # print("label_sim_dist: ", label_sim_dist.shape)
    # print("y_true", y_true.shape)
    # https://stackoverflow.com/questions/61437961/is-crossentropy-loss-of-pytorch-different-than-categorical-crossentropy-of-ker
    # simulated_y_true = nn.Softmax()(label_sim_dist+alpha*y_true)
    # nn.CrossEntropyLoss 和 keras categorical_crossentropyloss 有区别
    simulated_y_true = label_sim_dist+alpha*y_true
    loss1 = -pt_categorical_crossentropy(simulated_y_true,simulated_y_true)
    loss2 = pt_categorical_crossentropy(simulated_y_true,pred_probs)
    return loss1+loss2


def train(model, train_data, val_data, learning_rate, epochs):
  # 通过Dataset类获取训练和验证集
    train, val = Dataset(train_data), Dataset(val_data)
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)
  # 判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 定义损失函数和优化器
    criterion = lcm_loss
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()
    # 开始进入训练循环
    for epoch_num in range(epochs):
      # 定义两个变量，用于存储训练集的准确率和损失
            total_acc_train = 0
            total_loss_train = 0
      # 进度条函数tqdm
            for train_input, label_input, train_label, _ in tqdm(train_dataloader):
                train_label = train_label.to(device)
                label_input = label_input.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)
        # 通过模型得到输出
                output = model(input_id, mask, label_input)
                # 计算损失
                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()
                # 计算精度
                acc = (output[:, :num_classes].argmax(dim=1) == train_label.argmax(dim=1)).sum().item()
                total_acc_train += acc
        # 模型更新
                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            # ------ 验证模型 -----------
            # 定义两个变量，用于存储验证集的准确率和损失
            total_acc_val = 0
            total_loss_val = 0
      # 不需要计算梯度
            with torch.no_grad():
                # 循环获取数据集，并用训练好的模型进行验证
                for val_input, label_input, val_label, _ in val_dataloader:
          # 如果有GPU，则使用GPU，接下来的操作同训练
                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)
                    label_input = label_input.to(device)

                    output = model(input_id, mask, label_input)

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()
                    
                    acc = (output[:, :num_classes].argmax(dim=1) == val_label.argmax(dim=1)).sum().item()
                    total_acc_val += acc
            
            print(
                f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_data): .3f} 
              | Train Accuracy: {total_acc_train / len(train_data): .3f} 
              | Val Loss: {total_loss_val / len(val_data): .3f} 
              | Val Accuracy: {total_acc_val / len(val_data): .3f}''')   

def evaluate(model, test_data):
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():
        for test_input, input_label, test_label, _ in test_dataloader:
              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)
              output = model(input_id, mask, input_label)[:,:num_classes]
              acc = (output.argmax(dim=1) == test_label.argmax(dim=1)).sum().item()
              total_acc_test += acc   
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')


df_train, df_test = train_test_split(df_data, test_size=0.3, random_state=666, stratify=df_data["label"])
print("train len: ", len(df_train))
print("test len: ", len(df_test))



EPOCHS = 6
model = BertClassifier(num_classes=num_classes)
LR = 1e-3

train(model, df_train, df_test, LR, EPOCHS)
evaluate(model, df_test)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
    model = model.cuda()

result_map = {}
with torch.no_grad():
    test_list = df_test.values.tolist()
    for test_data in test_list:
        name = test_data[0]
        input_label = torch.tensor(np.array(range(num_classes))).to(device)
        test_input =  tokenizer(test_data[2], padding='max_length', 
                                        max_length = 128, 
                                        truncation=True,
                                        return_tensors="pt")
        mask = test_input['attention_mask'].to(device)
        input_id = test_input['input_ids'].squeeze(1).to(device)
        output = model(input_id, mask, input_label)[:,:num_classes]
        result_map[name] = num_label_map[output.argmax(dim=1).item()]

df_test["pred"] = df_test["name"].apply(lambda x: result_map[x])
df_test = df_test.loc[:,["name", "label", "pred", "summary", "description"]]
df_test.to_csv("bert_result.csv", index=False)
df_error = df_test.loc[df_test["label"] != df_test["pred"]]
df_error.to_csv("bert_result_errror.csv", index=False)

report = classification_report(df_test["label"], df_test["pred"])
print(report)

matrix = confusion_matrix(df_test["label"], df_test["pred"])
print(matrix)
