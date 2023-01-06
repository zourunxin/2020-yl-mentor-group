# 包名 label predict description summary
import pandas as pd
import csv

def trans_text(data):
    text = str(data)
    return text if text != "nan" else ""

layer_data = pd.read_csv("../result/12.17/人工_tfidf_分层.csv")
layer_data = layer_data.values.tolist()
class_data = pd.read_csv("../result/12.17/人工_tfidf_分类.csv")
class_data = class_data.values.tolist()

rpm_data = pd.read_csv("../data_resources/1128-bin.csv")
rpm_data = rpm_data.values.tolist()

out_list = []

for i, data in enumerate(layer_data):
    if layer_data[i][1] != class_data[i][1]:
        print(layer_data[i][1])
    # if layer_data[i][2] != layer_data[i][3] or class_data[i][2] != class_data[i][3]:
    if layer_data[i][2] != layer_data[i][3]:
        for d in rpm_data:
            if layer_data[i][1] == d[0]:
                description = trans_text(d[1]) + trans_text(d[3])
                summary = trans_text(d[2]) + trans_text(d[4])
                # out_list.append([layer_data[i][1], d[6], layer_data[i][3], class_data[i][3], summary, description])
                out_list.append([layer_data[i][1], layer_data[i][2], layer_data[i][3], summary, description])

# with open("../output/error_list_all.csv", "w", newline='', encoding="utf-8-sig") as csvfile:
with open("../output/error_list_layer.csv", "w", newline='', encoding="utf-8-sig") as csvfile:
    writer = csv.writer(csvfile)
    # writer.writerow(["name", "label", "layer_pred", "class_pred", "summary", "description"])  # 先写入columns_name
    writer.writerow(["name", "layer_true", "layer_pred", "summary", "description"])  # 先写入columns_name
    writer.writerows(out_list)  # 写入多行用writerows

print(len(out_list))
