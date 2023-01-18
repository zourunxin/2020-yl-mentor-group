from matplotlib import pyplot as plt

from utils.FileUtil import csv_reader, xlrd_reader


def draw_pic(file, titles, nums, labels):
    """
    绘图工具类
    """
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            # 同时显示数值和占比的饼图
            return '{p:.2f}%  ({v:d})'.format(p=pct, v=val)

        return my_autopct

    plt.rcParams['figure.figsize'] = (12.0, 12.0)  # 设置figure_size尺寸
    i = 1
    for num, label, title in zip(nums, labels, titles):
        plt.subplot(len(titles), 1, i)
        plt.pie(num, labels=label, autopct=make_autopct(num))
        plt.title(title)
        i += 1
    plt.savefig(file)
    return


def multi_overlap_pkg():
    nums = []
    for mode in ['layer', 'class']:
        reader = csv_reader('/Users/zourunxin/Mine/Seminar/20Data/1228/GraphSAGE/GraphSAGE_result_{}.csv'.format(mode))
        wrong_pkg = []
        for line in reader:
            wrong_pkg.append(line[0])

        cnt_list = []
        for sheet in ['仅有同层边的包数量', '同层-不同层边均有的包数量', '仅有不同层边的包数量', '没有边的包数量']:
            reader = xlrd_reader('/Users/zourunxin/Mine/Seminar/20Data/1228/analy_src_data/pkgs.xlsx', sheet)
            cnt = 0
            for i in range(1, reader.nrows):
                pkg = reader.row_values(i)[0]
                if pkg in wrong_pkg:
                    cnt += 1
            cnt_list.append(cnt)
        nums.append(cnt_list)
    draw_pic('/Users/zourunxin/Mine/Seminar/20Data/1228/GraphSAGE/wrong_pkg_statistic.jpg',
             ['layer_num', 'class_num'],
             nums,
             [['仅有同层边的包数量', '同层-不同层边均有的包数量', '仅有不同层边的包数量', '没有边的包数量']] * 2)
    return


def overlap_pkg():
    reader = csv_reader('../output/GraphSAGE_result.csv')
    wrong_pkg = []
    for line in reader:
        wrong_pkg.append(line[0])
    cnt_list = []
    for sheet in ['仅有同层边的包数量', '同层-不同层边均有的包数量', '仅有不同层边的包数量', '没有边的包数量']:
        reader = xlrd_reader('/Users/zourunxin/Mine/Seminar/20Data/1228/analy_src_data/pkgs.xlsx', sheet)
        cnt = 0
        for i in range(1, reader.nrows):
            pkg = reader.row_values(i)[0]
            if pkg in wrong_pkg:
                cnt += 1
        cnt_list.append(cnt)
    draw_pic('/Users/zourunxin/Mine/Seminar/20Data/1228/GraphSAGE/wrong_pkg_statistic_layer.jpg',
             ['layer_num'], [cnt_list], [['仅有同层边的包数量', '同层-不同层边均有的包数量', '仅有不同层边的包数量', '没有边的包数量']])


if __name__ == '__main__':
    overlap_pkg()