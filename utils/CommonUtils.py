import numpy as np

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
    label_list = list(set(list(labels)))
    label_num_map = {}
    num_label_map = {}
    for i, label in enumerate(label_list):
        label_num_map[label] = i
        num_label_map[i] = label
    
    return num_label_map, label_num_map