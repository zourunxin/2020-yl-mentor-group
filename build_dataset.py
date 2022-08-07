import json
import re
import numpy as np
import pandas as pd
import csv

from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize


invalid_word = stopwords.words('english')

# 获取单词的词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def add_to_dict(word_list, windows=5):
    valid_word_list = []  # 先进行过滤

    for word in word_list:
        word = str(word).lower()
        if is_valid(word):
            valid_word_list.append(word)

    # 根据窗口进行关系建立
    if len(valid_word_list) < windows:
        win = valid_word_list
        build_words_from_windows(win)
    else:
        index = 0
        while index + windows <= len(valid_word_list):
            win = valid_word_list[index:index + windows]
            index += 1
            build_words_from_windows(win)


# 根据小窗口，将关系建立到words中
def build_words_from_windows(win):
    for word in win:
        if word not in words.keys():
            words[word] = []
        for other in win:
            if other == word or other in words[word]:
                continue
            else:
                words[word].append(other)


# 预处理,如果是False就丢掉
def is_valid(word):
    if re.match("[()\-:;,.0-9]+", word) or word in invalid_word:
        return False
    elif len(word) < 4:
        return False
    else:
        return True


def text_rank(d=0.85, max_iter=100):
    min_diff = 0.05
    words_weight = {}  # {str,float)
    for word in words.keys():
        words_weight[word] = 1 / len(words.keys())
    for i in range(max_iter):
        n_words_weight = {}  # {str,float)
        max_diff = 0
        for word in words.keys():
            n_words_weight[word] = 1 - d
            for other in words[word]:
                if other == word or len(words[other]) == 0:
                    continue
                n_words_weight[word] += d * words_weight[other] / len(words[other])
            max_diff = max(n_words_weight[word] - words_weight[word], max_diff)
        words_weight = n_words_weight
        # print('iter', i, 'max diff is', max_diff)
        if max_diff < min_diff:
            # print('break with iter', i)
            break
    return words_weight


def read(description):
    str = description
    sens = sent_tokenize(str)
    for sentence in sens:
        # print(sentence)
        tokens = word_tokenize(sentence)  # 分词
        tagged_sent = pos_tag(tokens)  # 获取单词词性

        wnl = WordNetLemmatizer()
        lemmas_sent = []
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原
        # print(lemmas_sent)
        add_to_dict(lemmas_sent, 5)


def load_description():
    f1 = open('./resources/raw_features.txt', encoding='gbk')
    raw_lines = []
    for line in f1:
        raw_lines.append(line.strip())

    raw_datas = []
    raw_data = []

    description = ""

    n = 0
    while (n < len(raw_lines)):
        if (raw_lines[n].split(':')[0] == 'depname'):
            raw_data = {}
            n += 1
        while (raw_lines[n].split(':')[0].strip() != 'Description'):
            raw_data[raw_lines[n].split(':', 1)[0].strip()] = raw_lines[n].split(':', 1)[1].strip()
            n += 1
        if (raw_lines[n].split(':')[0].strip() == 'Description'):
            description = ""
            n += 1
            while (n < len(raw_lines) and raw_lines[n].split(':')[0] != 'depname'):
                description += raw_lines[n].strip() + ' '
                n += 1
            raw_data['Description'] = description
            raw_datas.append(raw_data)
    descriptions = {}

    for data in raw_datas:
        descriptions[data['Name']] = data['Description']

    return descriptions

'''
    extract the keywords from description of a pkg
    reference code: https://github.com/Cpaulyz/BigDataAnalysis/tree/master/Assignment2
    Returns: {pkg_name:{textrank: (keyword1: value1), ...}, ...}
'''
def extract_desc_keyword():
    descriptions = load_description()

    result = {}

    for pkg_name in descriptions:
        read(descriptions[pkg_name])
        words_weight = text_rank()
        # sort words_weight
        sorted_words = sorted(words_weight.items(), key=lambda x: x[1], reverse=True)
        # take top 5 words
        result[pkg_name] = {'desc_keyword': sorted_words[:5]}
    
    return result


# compute node degree from adjacency matrix
def compute_degree(M):
    # [[入度, 出度]...]
    degree = np.zeros((M.shape[0], 2), dtype=int)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if (M[i][j] == 1):
                degree[i][1] += 1
                degree[j][0] += 1
    return degree


'''
    Function load graph info and compute indegree, outdegree and pagerank
    Returns: {pkg_name:{indegree, outdegree, pagerank}, ...}
'''
def extract_graph_info():
    node_data = pd.read_csv("./resources/dep_node.csv")
    edge_data = pd.read_csv("./resources/dep_edge.csv")

    # edge_data.drop_duplicates(keep='first', subset=['idx1', 'idx2'], inplace=True)
    edge_list = edge_data.values.tolist()
    node_list = node_data.values.tolist()

    n = len(node_data)
    m = len(edge_data)

    M = np.zeros((n, n), dtype=int)

    for edge in edge_list:
        M[edge[0]][edge[1]] = 1

    degree = compute_degree(M)

    for i in range(n):
        sum_i = np.sum(M[:, i])
        if sum_i != 0:
            for j in range(n):
                M[j][i] = M[j][i] / sum_i

    R = np.ones((n, 1), dtype=int) * (1 / n)
    alpha = 0.85
    R_1 = np.zeros((n, n), dtype=int)
    e = 100000
    count = 0

    while e > 0.0000000000001 or count < 1000:
        R_1 = np.dot(M, R) * alpha + (1 - alpha) / n
        e = R - R_1
        e = np.max(np.abs(e))
        R = R_1
        count += 1
        # print(f'iteration {count}: {e}')

    graph_info = {}
    for node in node_list:
        graph_info[node[1]] = {"pagerank": R[node[0]][0], 
        "in_degree": degree[node[0]][0], 
        "out_degree": degree[node[0]][1]}

    return graph_info


'''
    extract the label from annotated data, if pkg can not be found in the annotated data
    annotated it with 'unknown'
    Returns: {pkg_name:{label}, ...}
'''
def extract_label():
    label_map = dict(pd.read_csv("./resources/label.csv").values.tolist())
    node_list = pd.read_csv("./resources/dep_node.csv").values.tolist()

    label_info = {}
    for node in node_list:
        label_info[node[1]] = {'label': label_map[node[1]]} if node[1] in label_map else {'label': 'unknown'}

    return label_info




words = {}

# extract info from raw data
description = extract_desc_keyword()
graph_info = extract_graph_info()
label_info = extract_label()

package_info = {}

# merge dict
for pkg_name in description:
    package_info[pkg_name] = {}
    package_info[pkg_name].update(description[pkg_name])
    package_info[pkg_name].update(graph_info[pkg_name])
    package_info[pkg_name].update(label_info[pkg_name])

# final dataset: dict
print(package_info)

# write dict to csv
# TODO: ','delimiter
f = open('pkg_info.csv', 'w')
f.write("pkg_name,desc_keyword,pagerank,in_degree,out_degree,label\n")
for key in package_info.keys():
    f.write("{}".format(key))
    for k in package_info[key].keys():
        f.write(",{}".format(package_info[key][k]))
    f.write("\n")
f.close()