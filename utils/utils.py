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
