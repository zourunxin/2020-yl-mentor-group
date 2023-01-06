# 包名 label predict description summary
import pandas as pd
import csv

def trans_text(data):
    text = str(data)
    return text if text != "nan" else ""

bert_data = pd.read_csv("../result/12.31/bert_result_class.csv")
bert_data = bert_data.values.tolist()
graph_data = pd.read_csv("../result/12.31/GraphSAGE_result_class.csv")
graph_data = graph_data.values.tolist()

# filtered_bert_data = list(filter(lambda x: x[4] != x[5], l))
bert_set = set([d[1] for d in bert_data])
graph_set = set([d[1] for d in graph_data])

common_data = list(bert_set.intersection(graph_set))

bert_only = []
graph_only = []
both = []

label_map = {
    0: "库",
    1: "工具",
    2: "服务",
    3: "其它",
}

for pkg in common_data:
    bert_flag = False
    graph_flag = False
    bert_info = []
    graph_info = []
    for b_d in bert_data:
        if b_d[1] == pkg:
            if b_d[4] != b_d[5]:
                bert_info = [pkg, label_map[b_d[4]], label_map[b_d[5]]]
                bert_flag = True
    for g_d in graph_data:
        if g_d[1] == pkg:
            if g_d[2] != g_d[3]:
                graph_info = [pkg, g_d[2], g_d[3]]
                graph_flag = True
    if bert_flag and graph_flag:
        both.append([pkg, bert_info[1], bert_info[2], graph_info[2]])
    if bert_flag:
        bert_only.append(bert_info)
    if graph_flag:
        graph_only.append(graph_info)

print(len(both))
print(len(bert_only))
print(len(graph_only))




with open("../output/bert_error_list_class.csv", "w", newline='', encoding="utf-8-sig") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["name", "y_true", "y_predict"])  # 先写入columns_name
    writer.writerows(bert_only)  # 写入多行用writerows

with open("../output/graph_error_list_class.csv", "w", newline='', encoding="utf-8-sig") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["name", "y_true", "y_predict"])  # 先写入columns_name
    writer.writerows(graph_only)  # 写入多行用writerows

with open("../output/both_error_list_class.csv", "w", newline='', encoding="utf-8-sig") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["name", "y_true", "y_bert_predict", "y_graph_predict"])  # 先写入columns_name
    writer.writerows(both)  # 写入多行用writerows
