import pandas as pd
import csv


src_data = pd.read_csv("../output/datasource_1128.csv")
predict_data = pd.read_csv("../output/GraphSAGE_result_label_key.csv")

src_data = src_data.values.tolist()
predict_data = predict_data.values.tolist()

keywords = ["library", "libraries", "command", "utility", "utilities", "tool", "tools", "application", "commands", "service"]

errors = []
for data in predict_data:
    if data[2] != data[3]:
        keys = []
        for d in src_data:
            if d[0] == data[1]:
                for key in keywords:
                    if d[2].find(key) != -1:
                        keys.append(key)
        errors.append([data[1], data[2], data[3], keys])
        print("name:{}, y_true:{}, y_predict:{}, keys:{}".format(data[1], data[2], data[3], keys))


with open("../output/artificial_feat_error_analyze.csv", "w", newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["name", "y_true", "y_predict", "keys"])
    writer.writerows(errors)