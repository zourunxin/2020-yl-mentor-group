import pandas as pd

src_data = pd.read_csv("../result/12.17/人工特征分析.csv")
src_data = src_data.values.tolist()

key_class = {
    "tool": ["utility", "utilities", "tool", "tools"],
    "lib": ["library", "libraries"],
    "service": ["service"]
}

for data in src_data:
    keys = data[3].replace('"', "").replace("'", "").replace("[", "").replace("]", "").split(",")
    if len(keys) >= 1:
        if all(key in key_class["tool"] for key in keys):
            if data[1] != "工具":
                print(data)
        if all(key in key_class["lib"] for key in keys):
            if data[1] != "库":
                print(data)
        if all(key in key_class["service"] for key in keys):
            if data[1] != "服务":
                print(data)
