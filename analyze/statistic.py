# 读取分类结果
f = open('../result/GCN/GCN_result_layer.csv', 'r', encoding='utf-8-sig')
layers = f.readlines()[1:]
layers = [r.strip().split(",") for r in layers]
f.close()

f = open('../result/GCN/GCN_result_classification.csv', 'r', encoding='utf-8-sig')
classes = f.readlines()[1:]
classes = [r.strip().split(",") for r in classes]
f.close()

count = 0
for i, _ in enumerate(classes):
    if layers[i][2] == layers[i][3] and classes[i][2] == classes[i][3]:
        count += 1

print(count / len(classes))