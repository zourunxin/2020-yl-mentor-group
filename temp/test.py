import sys
sys.path.append("../")
import pandas as pd
import utils.NLPUtils as NLPUtils

df_data = pd.read_csv("../result/1.17/标签核对_feedback.csv")

# def is_valid_label(label):
#     return label in ["库", "工具", "服务", "其它"]

# data_list = df_data.values.tolist()

# # for data in data_list:
# #     if not is_valid_label(data[8]):
# #         print(data[0])

print(len(df_data))
df_unknown = df_data[df_data["人工初判"] == "unknown"]
print(len(df_unknown))
df_known = df_data[df_data["人工初判"] != "unknown"]
print(len(df_known))
df_correct = df_known[df_known["人工初判"] == df_known["res"]]
print(len(df_correct))
df_wrong = df_known[df_known["人工初判"] != df_known["res"]]
print(len(df_wrong))
df_unique = df_unknown[df_unknown["label"] != df_unknown["res"]]
print(len(df_unique))
df_same = df_unknown[df_unknown["label"] == df_unknown["res"]]
print(len(df_same))
df_yc = df_wrong[df_wrong["label"] == df_wrong["res"]]
print(len(df_yc))
df_yw = df_wrong[df_wrong["label"] != df_wrong["res"]]
print(len(df_yw))

# 总计提交复议的包 201 个，其中：
# 我主动修改了标签的包：160 个(80%)
# 我仅提出异议但无法判断标签(标记为 unknown)的包：41 个(20%)
# 在我主动修改了标签的包中，修改正确 113 个(70.6%)，修改错误 47 个(29.4%)
# 修改错误 47 个中：社区认为原标签正确 23 个，社区重新修改了源标签 24 个
# 在我仅提出异议但无法判断标签(标记为 unknown)的包中，标签确实有误 23 个(56%)，标签没有误 18 个(44%)
# 总体接受率 68% (113 + 23 / 201), 