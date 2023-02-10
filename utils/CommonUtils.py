import numpy as np
import matplotlib.pyplot as plt

def convert_label(label, mode="both"):
    layer_map = {
        "内核": "核心",
        "基础环境": "核心",
        "核心库": "核心",
        "核心工具": "核心",
        "核心服务": '核心',
        "系统服务": "系统",
        "系统库": "系统",
        "系统工具": "系统",
        "应用服务": "应用",
        "应用库": "应用",
        "应用工具": "应用",
        "虚拟化": "应用",
    }

    class_map = {
        "核心库": "库",
        "核心工具": "工具",
        "核心服务": '服务',
        "系统服务": "服务",
        "系统库": "库",
        "系统工具": "工具",
        "应用服务": "服务",
        "应用库": "库",
        "应用工具": "工具",
    }

    _label_map = {
        "基础环境": "基础环境",
        "核心库": "核心库",
        "核心工具": "核心工具",
        "核心服务": "核心服务",
        "系统服务": "系统服务",
        "系统库": "系统库",
        "系统工具": "系统工具",
        "应用服务": "应用服务",
        "应用库": "应用库",
        "应用工具": "应用工具",
    }

    if mode == "class":
        _label_map = class_map
    elif mode == "layer":
        _label_map = layer_map

    return _label_map[label] if label in _label_map else "其它"


def get_idx_name_map(names):
    """
    输入包名列表，输出 name_idx, idx_name Map
    """
    name_list = list(names)
    name_idx_map = {}
    idx_name_map = {}
    for i, name in enumerate(name_list):
        name_idx_map[name] = i
        idx_name_map[i] = name

    return idx_name_map, name_idx_map

def get_num_label_map(labels):
    """
    输入标签列表，输出 name_idx, idx_name Map
    """
    label_list = []
    # 为了保持每次的顺序（有无更好的方法）
    for label in list(labels):
        if not label in label_list:
            label_list.append(label)
    # label_list = list(set(list(labels)))
    label_num_map = {}
    num_label_map = {}
    label_list = []
    # for i, label in enumerate(label_list):
    for i, label in enumerate(['核心', '系统', '应用', '其它']):
        label_num_map[label] = i
        num_label_map[i] = label

    return num_label_map, label_num_map

def get_sample_splits(x, y, sample_size=100):
    '''
    分割数据集 (图算法可能需要使用 mask 形式， 参考 GraphSAGE 代码)
    y: one_hot 形式的标签
    '''
    idx_set = []
    for i in range(y.shape[1]):
        idx_set.append([])

    for i, label in enumerate(y):
        label = np.argmax(label)
        idx_set[label].append(i)

    idx_train = []
    idx_val = []
    idx_test = []

    for s in idx_set:
        np.random.seed(666)
        np.random.shuffle(s)
        idx_train = idx_train + s[0:int(len(s) * 0.7)]
        idx_val = idx_val + s[int(len(s) * 0.7):]
        idx_test = idx_test + s[int(len(s) * 0.7):]

    print("样本数量 train: {}, val: {}, test: {}".format(len(idx_train), len(idx_val), len(idx_test)))

    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    nd_y = np.array(y, dtype=np.int32)
    nd_x = np.array(x)
    y_train = nd_y[idx_train]
    y_val = nd_y[idx_val]
    y_test = nd_y[idx_test]
    x_train = nd_x[idx_train]
    x_val = nd_x[idx_val]
    x_test = nd_x[idx_test]

    print(x_train.shape)
    return x_train, x_val, x_test, y_train, y_val, y_test, idx_train, idx_test

def show_learning_curves(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()