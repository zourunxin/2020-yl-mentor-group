import pandas as pd
import csv

src_dir = "../data_resources/1128-src.csv"
rpm_dir = "../data_resources/1128-bin.csv"

src_data = pd.read_csv(src_dir)
rpm_data = pd.read_csv(rpm_dir)

src_list = src_data.values.tolist()
rpm_list = rpm_data.values.tolist()



name_label_text_list = []

def trans_text(data):
    text = str(data)
    return text if text != "nan" else ""

for data in rpm_list:
    # [name, label, text]
    name = data[0]
    # rpm:6 src:9
    label = data[6].split("/")[0].strip()
    text = trans_text(data[1]) + trans_text(data[2]) + trans_text(data[3]) + trans_text(data[4])
    # for rpm_data in rpm_list:
    #     if rpm_data[5] == name:
            # text += trans_text(rpm_data[1]) + trans_text(rpm_data[2]) + trans_text(rpm_data[3]) + trans_text(rpm_data[4])
    name_label_text_list.append([name, label, text])

with open("../output/datasource_1128.csv", "w", newline='', encoding="utf-8-sig") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["name", "label", "text"])  # 先写入columns_name
    writer.writerows(name_label_text_list)  # 写入多行用writerows
