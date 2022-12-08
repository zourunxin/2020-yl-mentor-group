def class_cnt(class_list: list):
    """
    统计 class_list 中各类元素的个数
    :param class_list: <label1, label2, ...,>
    :return: <class1 : cnt1, class2 : cnt2, ...,>
    """
    res = dict()
    for i in class_list:
        cnt = res.get(i, 0) + 1
        res[i] = cnt
    return res


def get_label_list():
    f = open('../output/label_list.csv', 'r', encoding='utf-8-sig')
    result = f.readlines()[0]
    result = result.strip().split(",")
    f.close()
    return result


def get_label_layer_map():
    return {
        '内核': 0,
        '基础环境': 1,
        '核心库': 1,
        '核心工具': 1,
        '系统服务': 2,
        '系统库': 2,
        '系统工具': 2,
        '应用服务': 3,
        '应用库': 3,
        '应用工具': 3,
        '编程语言': 3,
    }


def get_digit_layer_map():
    label_list = ['内核', '基础环境、核心库、核心工具', '系统服务、系统库、系统工具', '应用服务、应用库、应用工具、编程语言', '其它']
    label_dict = dict()
    for i, enum in enumerate(label_list):
        label_dict[str(i)] = enum
    return label_dict


def get_label(label: list):
    """
    将 label 清洗为合法 label 并返回
    :param label:
    :return:
    """
    # labels = {'内核', '基础环境', '核心库', '核心工具', '系统服务', '系统库', '系统工具', '应用服务', '应用库',
    #           '应用工具', '编程语言', '其它'}
    labels = {'基础环境', '核心库', '核心工具', '系统服务', '系统库', '系统工具', '应用服务', '应用库',
              '应用工具', '编程语言', '其它'}   # 无内核，用于喂入 digcn。因为 digcn 在抽样数据时内核只有一条数据，抽样报错。
    for x in label:
        if x in labels:
            return x
    return '其它'
