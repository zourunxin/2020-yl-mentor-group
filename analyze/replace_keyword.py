import sys
sys.path.append("../")
import pandas as pd
import utils.NLPUtils as NLPUtils

keyword_map = {"工具": ['package', 'file', 'provide', 'contain', 'tool', 'hunspell', 'support', 'language', 'dictionary', 'font', 'library', 'version', 'data', 'program', 'command', 'utility', 'tesseract', 'source', 'latex', 'engine', 'include', 'integer', 'script', 'model', 'fast', 'maven', 'train', 'module', 'user', 'open'],
"库": ['library', 'package', 'provide', 'python', 'module', 'file', 'support', 'application', 'perl', 'contain', 'interface', 'libreoffice', 'data', 'language', 'version', 'access', 'tool', 'java', 'function', 'server', 'implementation', 'include', 'allow', 'write', 'binding', 'class', 'runtime', 'code', 'documentation', 'format'],
"其它": ['font', 'noto', 'script', 'sans', 'family', 'fonts', 'unicode', 'support', 'harmonization', 'tofu', 'package', 'provide', 'contain', 'design', 'goal', 'available', 'achieve', 'multiple', 'remove', 'visual', 'georgian', 'serif', 'file', 'culmus', 'display', 'firmware', 'project', 'free', 'kernel', 'arabic'],
"服务": ['package', 'server', 'langpacks', 'provide', 'contain', 'agent', 'performance', 'metric', 'file', 'daemon', 'pmda', 'fence', 'gnome', 'plugin', 'support', 'client', 'library', 'access', 'network', 'metrics', 'module', 'device', 'tool', 'domain', 'protocol', 'application', 'driver', 'service', 'user', 'deepin'],}

df_data = pd.read_csv("../result/1.12/GraphSAGE_result.csv")

df_data["text"] = df_data.apply(lambda x: NLPUtils.preprocess_text(x["summary"] + " " + x["description"]), axis=1)

replace_map = {}

# 单词出现在多个类中，它属于出现靠前（权值更大）的那个类，相同则都算
for i in range(len(keyword_map[list(keyword_map.keys())[0]])):
    for key in keyword_map:
        word_list = keyword_map[key]
        if not word_list[i] in replace_map:
            replace_map[word_list[i]] = key

def mark_text(text):
    for word in replace_map:
        text = text.replace(word, "{}({})".format(word, replace_map[word]))
    
    return text


df_data["marked_text"] = df_data["text"].apply(lambda x: mark_text(x))

print(df_data["marked_text"])

df_output = df_data[["name", "label", "predict", "marked_text"]]

df_output.to_csv("GraphSAGE_result_analyze.csv", index=False)
