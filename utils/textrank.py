import re
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
    elif not re.match("^[A-Za-z]+$", word):
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


def read(desc):
    sens = sent_tokenize(desc)
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


def extract_keyword(text, num):
    read(text)
    words_weight = text_rank()
    # sort words_weight
    sorted_words = sorted(words_weight.items(), key=lambda x: x[1], reverse=True)
    words.clear()
    # take top 5 words
    return sorted_words[:num]

def preprocess_text(text):
    '''
    文本清洗去除停用词、词形还原
    '''
    sens = sent_tokenize(text)
    filtered_words = []
    for sentence in sens:
        # print(sentence)
        tokens = word_tokenize(sentence)  # 分词
        tagged_sent = pos_tag(tokens)  # 获取单词词性

        wnl = WordNetLemmatizer()
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            word = wnl.lemmatize(tag[0], pos=wordnet_pos)
            if (is_valid(word)):
                filtered_words.append(word)
    return " ".join(list(set(filtered_words)))

words = {}
