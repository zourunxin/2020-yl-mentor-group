import csv

# 读取词袋
f = open('../output/wordbag.csv', 'r', encoding='utf-8-sig')
bag = f.readline()
bag = bag.strip().split(",")
f.close()

# 读取分类结果
f = open('../output/GraphSAGE_result.csv', 'r', encoding='utf-8-sig')
results = f.readlines()[1:]
results = [r.strip().split(",") for r in results]
f.close()

# 读取 BoW 特征
f = open('../output/feature.csv', 'r', encoding='utf-8-sig')
features = f.readlines()
features = [l.strip().split(",") for l in features]
f.close()

for result in results:
    feature = features[int(result[0])]
    feature_words = []
    for i, feat in enumerate(feature):
        if int(feat) > 0:
            feature_words.append(bag[i])
    words = " ".join(feature_words)
    result.append(words)

with open("../output/result_analyze_all.csv", "w", newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["idx", "name", "y_true", "y_predict", "feature"])
    writer.writerows(results)

wrong_results = [x for x in results if x[2] != x[3]]

with open("../output/result_analyze_wrong.csv", "w", newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["idx", "name", "y_true", "y_predict", "feature"])
    writer.writerows(wrong_results)