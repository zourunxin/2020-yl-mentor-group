import pandas as pd
import csv

data_dir = "../data_resources/data_1008.csv"
label_dir = "../data_resources/1107.csv"

src_data = pd.read_csv(data_dir, header=None)
labels = pd.read_csv(label_dir, header=None)

src_list = src_data.values.tolist()
label_list = labels.values.tolist()

name_label_map = {}
for l in label_list:
    name_label_map[l[0]] = l[1]

pkg_data_list = []

def trans_text(data):
    text = str(data)
    return text if text != "nan" else ""

for data in src_list:
    # [name, label, text]
    pkg_data = [data[3], data[0] if data[1] == "编程语言" or data[2].find("语言") != -1 else data[2], trans_text(data[4]) +  trans_text(data[5]) +  trans_text(data[6]) +  trans_text(data[7])]
    if pkg_data[0] in name_label_map:
        pkg_data[1] = name_label_map[pkg_data[0]]
    pkg_data_list.append(pkg_data)

with open("../output/datasource_1107.csv", "w", newline='', encoding="utf-8-sig") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["name", "label", "text"])  # 先写入columns_name
    writer.writerows(pkg_data_list)  # 写入多行用writerows
