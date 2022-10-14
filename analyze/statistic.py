# 读取分类结果
f = open('../result/GraphSAGE_bow_binary_f5_sample100/result_analyze_all.csv', 'r', encoding='utf-8-sig')
results = f.readlines()[1:]
results = [r.strip().split(",") for r in results]
f.close()

