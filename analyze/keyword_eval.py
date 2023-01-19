import sys
sys.path.append("../")
import pandas as pd
import utils.NLPUtils as NLPUtils

keyword_map = {
    "库":["library", "interface", "api", "static", "runtime", "dynamic", "java", "python", "encode",],
    "服务":["daemon", "task", "network", "request", "monitor", "container", "virtualize", "authentic", "service",],
    "工具":["tool", "util", "utility", "command", "graphical", "processor", "gnu", "compile", "debug",],
    "其它":["font"],
}

df_data = pd.read_csv("../output/datasource_1228_new.csv")

#,name,label,text,summary,description
data_list = df_data.values.tolist()

lines = []
for key in keyword_map:
    keyword_list = keyword_map[key]
    for keyword in keyword_list:
        # keyword class num_lib num_service num_tool num_other
        line = [keyword, key, 0, 0, 0, 0]
        for data in data_list:
            text = NLPUtils.preprocess_text(data[3])
            if text.find(keyword) != -1:
                if data[2] == "库":
                    line[2] = line[2] + 1
                elif data[2] == "工具":
                    line[3] = line[3] + 1
                elif data[2] == "服务":
                    line[4] = line[4] + 1
                elif data[2] == "其它":
                    line[5] = line[5] + 1
        lines.append(line)
        print("finish")


df_res = pd.DataFrame(lines, columns=["keyword", "class", "num_lib", "num_service", "num_tool", "num_other"])
df_res.to_csv("keyword_sta.csv", index=False)