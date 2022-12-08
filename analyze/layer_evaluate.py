import csv

# 读取分类结果
f = open('../output/GraphSAGE_result.csv', 'r', encoding='utf-8-sig')
results = f.readlines()[1:]
results = [r.strip().split(",") for r in results]
f.close()

f = open('../output/edges.csv', 'r', encoding='utf-8-sig')
edges = f.readlines()[1:]
edges = [r.strip().split(",") for r in edges]
f.close()

layer_map = {"核心": 1, "系统": 2, "应用": 3, "其它": 4}

for i, res in enumerate(results):
    results[i] = res + [0, 0, 0, 0, 0]
    if layer_map[res[2]] - layer_map[res[3]] > 0:
        results[i][6] = -1
    elif layer_map[res[2]] - layer_map[res[3]] < 0:
        results[i][6] = 1

for edge in edges:
    from_idx = int(edge[0])
    to_idx = int(edge[1])
    if layer_map[results[from_idx][2]] - layer_map[results[to_idx][2]] > 1:
        results[from_idx][4] += 1
        results[to_idx][5] += 1
    if layer_map[results[from_idx][2]] - layer_map[results[to_idx][2]] < 0:
        results[from_idx][7] += 1
        results[to_idx][8] += 1


from_error = 0
to_error = 0
only_error = 0
only_from = 0
only_to = 0
only_correct = 0
re_from_error = 0
re_to_error = 0
re_only_from = 0
re_only_to = 0


from_high = 0
from_low = 0
to_high = 0
to_low = 0

for result in results:
    if result[4] > 0:
        if result[6] != 0:
            from_error += 1
            if result[6] == 1:
                from_high += 1
            else:
                from_low += 1
        else:
            only_from += 1
    elif result[5] > 0:
        if result[6] != 0:
            to_error += 1
            if result[6] == 1:
                to_high += 1
            else:
                to_low += 1
        else:
            only_to += 1
    if result[7] > 0:
        if result[6] != 0:
            re_from_error += 1
        else:
            re_only_from += 1
    elif result[8] > 0:
        if result[6] != 0:
            re_to_error += 1
        else:
            re_only_to += 1
    if result[4] == 0 and result[5] == 0 and result[7] == 0 and result[8] == 0:
        if result[6] != 0:
            only_error += 1
        else:
            only_correct += 1

print("跨层依赖且分类错误：{}\n被跨层依赖且分类错误：{}\n跨层依赖且分类正确：{}\n被跨层依赖且分类正确：{}"
.format(from_error, to_error, only_from, only_to))
print("反向依赖且分类错误：{}\n被反向依赖且分类错误：{}\n反向依赖且分类正确：{}\n被反向依赖且分类正确：{}"
.format(re_from_error, re_to_error, re_only_from, re_only_to))
print("仅分类错误：{}\n仅分类正确：{}\n"
.format(only_error, only_correct))
print("跨层依赖分类正确率：{}\n被跨层依赖分类正确率：{}"
.format(only_from / (from_error + only_from), only_to / (to_error + only_to)))
print("反向依赖分类正确率：{}\n被反向依赖分类正确率：{}"
.format(re_only_from / (re_from_error + re_only_from), re_only_to / (re_to_error + re_only_to)))
print("正确依赖分类正确率：{}\n"
.format( only_correct / (only_correct + only_error)))
print("跨层依赖且分类错误的包中，被分高：{},被分低：{}\n被跨层依赖且分类错误的包中，被分高：{},被分低：{}\n"
.format(from_high, from_low, to_high, to_low))


with open("../output/evaluate_layer.csv", "w", newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["idx", "name", "y_true", "y_predict", "跨层依赖数", "被跨层依赖数", "分类错误"])
    writer.writerows(results)
