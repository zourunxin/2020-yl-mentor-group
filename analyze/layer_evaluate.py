# 读取分类结果
f = open('../output/GraphSAGE_result.csv', 'r', encoding='utf-8-sig')
results = f.readlines()[1:]
results = [r.strip().split(",") for r in results]
f.close()


def cal_layer_rate(layer_name, results):
    layer_map = {"核心": 1, "系统": 2, "应用": 3, "其它": 4}
    higher_count = 0
    lower_count = 0
    all_count = 0
    for result in results:
        if result[2] == layer_name:
            all_count += 1
            if layer_map[result[3]] > layer_map[result[2]]:
                higher_count += 1
            if layer_map[result[3]] < layer_map[result[2]]:
                lower_count += 1
    return higher_count / all_count, lower_count / all_count

layers = ["核心", "系统", "应用", "其它"]
result_map = {}

for layer in layers:
    result_map[layer] = cal_layer_rate(layer, results)

print(result_map)
