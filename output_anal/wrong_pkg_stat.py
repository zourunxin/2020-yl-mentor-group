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
            return '{p:.2f}%  ({v:d}) ({x:.2f)'.format(p=pct, v=val)

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


def overlap_pkg(file, sheets):
    """
    计算分错的包与特定类型包的重叠情况
    """
    reader = csv_reader(file)
    wrong_pkg = []
    for line in reader:
        wrong_pkg.append(line[0])
    ol_pkg = []     # 分错包与各特定类型包的重叠包
    ol_cnt = []     # 分错包与各特定类型包的重叠包数量
    overlaps = []     # 分错包与各特定类型包的重叠率
    for sheet in sheets:
        reader = xlrd_reader('/Users/zourunxin/Mine/Seminar/20Data/1228/analy_src_data/specificEdge_pkg.xlsx', sheet)
        cnt = 0
        pkgs = []
        for i in range(1, reader.nrows):
            pkg = reader.row_values(i)[0]
            if pkg in wrong_pkg:
                cnt += 1
                pkgs.append(pkg)
        ol_pkg.append(pkgs)
        ol_cnt.append(cnt)
        overlaps.append(cnt/(reader.nrows - 1) if cnt > 0 else 0)
        print('分错的包归属"' + sheet + '"的个数: {}, 重叠率: {:.2f}, 该类包总数: {}'.format(cnt, cnt/(reader.nrows - 1) if cnt > 0 else 0, reader.nrows - 1))
    return ol_pkg, ol_cnt, overlaps


def multi_file_overlap_pkg():
    """
    计算多个分错包与特定类型包的重叠情况，输出平均值
    """
    sheets = ['只有不同层边的包', '只有向上一层边的包', '只有向上两层边的包', '只有向上三层边的包', '含有向上一层边的包',
              '含有向上两层边的包', '含有向上三层边的包', '含有向上层边的包', '只有向上层边的包', '含有向下层边的包', '只有向下层边的包',
              '含有向下一层边的包', '含有向下两层边的包', '含有向下三层边的包', '只有向下一层边的包', '只有向下两层边的包', '只有向下三层边的包']
    overlap_cnt = [0] * 17
    overlap = [0] * 17
    for i in range(3):
        ol_pkg, ol_cnt, ol = overlap_pkg('../output/zrx/GraphSAGE_result{}.csv'.format(i), sheets)
        overlap_cnt = list(map(lambda x: x[0] + x[1], zip(overlap_cnt, ol_cnt)))
        overlap = list(map(lambda x: x[0] + x[1], zip(overlap, ol)))
    print('')
    for sheet, ol_cnt, ol in zip(sheets, overlap_cnt, overlap):
        print('分错的包归属"' + sheet + '"的个数: {}, 重叠率: {:.2f}'.format(ol_cnt/10, ol/10))


def overlap_pkg1():
    # 六个小连通分量的包
    wrong_pkg = ['gnome-getting-started-docs-hu', 'gnome-getting-started-docs-ru', 'gnome-getting-started-docs-pt_BR', 'gnome-getting-started-docs-de', 'gnome-getting-started-docs-it', 'gnome-getting-started-docs-fr', 'gnome-getting-started-docs-es', 'gnome-getting-started-docs-pl', 'gnome-getting-started-docs-gl', 'gnome-getting-started-docs-cs', 'gnome-user-docs', 'gnome-getting-started-docs',
                 'f32-backgrounds-extras-gnome', 'f32-backgrounds-extras-kde', 'f32-backgrounds-extras-base', 'f32-backgrounds-extras-xfce', 'f32-backgrounds-extras-mate',
                 'ipxe-roms', 'ipxe-roms-qemu',
                 'python3-langtable', 'langtable',
                 'seabios', 'seavgabios-bin', 'seabios-bin',
                 'elfutils-devel-static', 'elfutils-libelf-devel-static']
    reader = csv_reader('../output/GraphSAGE_result.csv')
    cnt = 0
    for line in reader:
        if line[0] in wrong_pkg:
            cnt += 1
    print('分错包与小连通分量包的重叠率在 {}'.format(cnt / len(wrong_pkg)))
    return



if __name__ == '__main__':
    # 获取分错包与各特定类型包的重叠情况
    # overlap_pkg('../output/GraphSAGE_result.csv',
    #             ['只有不同层边的包', '只有向上一层边的包', '只有向上两层边的包', '只有向上三层边的包', '含有向上一层边的包',
    #               '含有向上两层边的包', '含有向上三层边的包', '含有向上层边的包', '只有向上层边的包', '含有向下层边的包', '只有向下层边的包',
    #               '含有向下一层边的包', '含有向下两层边的包', '含有向下三层边的包', '只有向下一层边的包', '只有向下两层边的包', '只有向下三层边的包'])

    # 获取未去上层边、去除上层边的分错包的在特定类型包的重叠情况
    # ol_pkg1 = overlap_pkg('../output/GraphSAGE_result.csv', ['只有向上层边的包'])
    # ol_pkg2 = overlap_pkg('../output/GraphSAGE_legal_result.csv', ['只有向上层边的包'])
    # print('两者重叠包的个数: {}'.format(len(set(ol_pkg1[0]) & set(ol_pkg2[0]))))
    # print(set(ol_pkg1[0]) & set(ol_pkg2[0]))

    # 计算多次训练后分错的包与各特定类型包的平均重叠率
    multi_file_overlap_pkg()