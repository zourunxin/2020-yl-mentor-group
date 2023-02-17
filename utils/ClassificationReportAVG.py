import pdb
from operator import add


def cr_avg(reports):
    """
    reports: 列表，存储每个分类器的 classification_report
    """
    result = {}
    for report in reports:
        lines = report.strip().split('\n')
        # 计算 acc、precision、recall、support
        for line in lines[2:-4]:
            row = line.strip().split()
            class_label = row[0]
            result[class_label] = result.get(class_label, [0, 0, 0, 0])
            for i, value in enumerate(row[1:]):
                result[class_label][i] += float(value)

        # 计算 accuracy
        row = lines[7].strip().split()
        result['accuracy'] = result.get('accuracy', [0, 0])
        result['accuracy'][0] += float(row[1])
        result['accuracy'][1] += float(row[2])

        # 计算 macro avg、weighted avg
        for line in lines[8:]:
            row = line.strip().split()
            class_label = row[0] + ' ' + row[1]
            result[class_label] = result.get(class_label, [0, 0, 0, 0])
            for i, value in enumerate(row[2:]):
                result[class_label][i] += float(value)

    for class_label in result:
        result[class_label] = [format(value / len(reports), '.3f') for value in result[class_label]]

    print(' ', 'Precision', 'Recall', 'F1-Score', 'Support')
    for class_label, values in result.items():
        print(class_label, *values)



def save_cr(report, file):
    with open(file, 'a+', encoding='utf-8') as f:
        print("result:", file=f)
        print(report, file=f)






    #
    #
    # # 数组，存储所有分类器的 precision，recall，f1-score，accuracy 和 support
    # precision = []
    # recall = []
    # f1 = []
    # accuracy = []
    # support = []
    #
    # # 循环每个分类器的 classification_report
    # for i, report in enumerate(reports):
    #     lines = report.split('\n')
    #     # 找到 accuracy 行，并从中提取数值
    #     for line in lines:
    #         if 'accuracy' in line:
    #             accuracy.append(float(line.split()[-2]))
    #     # 找到 precision，recall 和 f1-score 行，并从中提取数值
    #     for line in lines[2:-3]:
    #         row = [x for x in line.strip().split(' ') if x]
    #         precision.append(float(row[0]))
    #         recall.append(float(row[1]))
    #         f1.append(float(row[2]))
    #         support.append(int(row[3]))
    #
    # # 将列表转换为 NumPy 数组
    # precision = np.array(precision)
    # recall = np.array(recall)
    # f1 = np.array(f1)
    # accuracy = np.array(accuracy)
    # support = np.array(support)
    #
    # # 计算平均值
    # precision_mean = np.mean(precision)
    # recall_mean = np.mean(recall)
    # f1_mean = np.mean(f1)
    # accuracy_mean = np.mean(accuracy)
    # support_mean = np.mean(support)
    #
    # # 计算 weighted avg
    # weighted_precision = np.average(precision, weights=support)
    # weighted_recall = np.average(recall, weights=support)
    # weighted_f1 = np.average(f1, weights=support)
    #

    #
    #
    # # 计算 macro_avg
    # macro_avg = [0.0] * 3
    # for line in result.values():
    #     macro_avg = list(map(add, macro_avg, line[:-1]))
    # macro_avg = [format(value / len(result), '.3f') for value in macro_avg[:-1]]
    # macro_avg.append(acc[-1])
    #
    # # 计算 weighted_avg
    # weighted_avg = [0.0] * 3
    # for line in result.values():
    #     weighted_avg = list(map(add, weighted_avg, [x * line[-1] for x in line[:-1]]))
    # weighted_avg = [format(value / acc[-1], '.3f') for value in weighted_avg]
    # weighted_avg.append(acc[-1])
    #
    # # 打印
    # for class_label in result:
    #     support = result[class_label][-1]
    #     result[class_label] = [format(value / len(reports), '.3f') for value in result[class_label][:-1]]
    #     result[class_label].append(str(int(support / len(reports))))
    #

