import csv
from sklearn.metrics import confusion_matrix

# 读取分类结果
f = open('../output/GraphSAGE_result_分类_df.csv', 'r', encoding='utf-8-sig')
class_results = f.readlines()[1:]
class_results = [r.strip().split(",") for r in class_results]
f.close()

# 读取分类结果
f = open('../output/GraphSAGE_result_分层_df.csv', 'r', encoding='utf-8-sig')
layer_results = f.readlines()[1:]
layer_results = [r.strip().split(",") for r in layer_results]
f.close()

tf = []
ft = []
ff = []

# for i, _ in enumerate(class_results):
#     res = [class_results[i][1], class_results[i][2], class_results[i][3], layer_results[i][3]]
#     if res[1] == res[2]:
#         if res[1] != res[3]:
#             tf.append(res)
#     else:
#         if res[1] == res[3]:
#             ft.append(res)
#         else:
#             ff.append(res)

# with open("../output/analyze.csv", "w", newline='', encoding='utf-8-sig') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["idx", "name", "y_true", "Bow_predict", "df_predict"])
#     writer.writerow(["Bow 分类正确, df 分类错误"])
#     writer.writerows(tf)
#     writer.writerow(["Bow 分类错误, df 分类正确"])
#     writer.writerows(ft)
#     writer.writerow(["Bow 分类错误, df 分类错误"])
#     writer.writerows(ff)




with open("../output/confusion_matrixs.csv", "w", newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    modes = ["class", "layer"]
    for mode in modes:
        results = class_results if mode == "class" else layer_results
        labels = ["工具", "服务", "库", "其它"] if mode == "class" else ["核心", "系统", "应用", "其它"]
        c = confusion_matrix([r[2] for r in results], [r[3] for r in results], labels=labels)
        writer.writerow(["类别"] + labels)
        for i, label in enumerate(labels):
            writer.writerow([label] + list(c[i]))