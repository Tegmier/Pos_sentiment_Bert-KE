import numpy as np
import re
import pickle
from collections import Counter
import os
from transformers import BertTokenizer, BertModel
from typing import List, Tuple

import logging
import re

# 设置transformers库的日志级别为ERROR，仅显示Error及以上级别的消息
logging.getLogger("transformers").setLevel(logging.ERROR)

# 定义BIOES字典
labels2idx = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}

# 定义URLpattern正则表达式
url_pattern = re.compile(r'https?://\S+')

def get_data_list_from_path(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        datalist, taglist = [], []
        count = 0
        tweet_list = []
        tag_list = []
        for line in f:
            count += 1
            line = line.strip()
            tweet_list.append(line.split('\t')[0])
            tag_list.append(line.split('\t')[1])   
    return tweet_list, tag_list 

def get_dict(data_path):
    trn_data_path, test_data_path = data_path
    tweet_list, tag_list = get_data_list_from_path(trn_data_path)
    return tweet_list, tag_list

def bert_tokenizer_qualification(tweet_list, tag_list):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    count_total = 0
    count_same = 0
    qualified_tweet_list, qualified_tag_list = [], []
    for tweet, tag in zip(tweet_list, tag_list):
        count_total+=1
        tokenized_tag = tokenizer.tokenize(tag)
        if len(tag.split()) == len(tokenized_tag):
            count_same+=1
            qualified_tweet_list.append(tweet)
            qualified_tag_list.append(tag)
    # print("Total number of Tags is: " + str(count_total))
    # print("Total number of qualified Tags is: " + str(count_same))
    return qualified_tweet_list, qualified_tag_list

def get_train_test_dicts(data_path):
    """[summary]
    Args:
        data_path: [train_tweet_data_path, test_tweet_data_path]

    Returns:
        data_set: [train_set, test_set, dicts]
        train_set = [train_lex, train_y, train_z]
        test_set = [test_lex, test_y, test_z]
        discts = {'word2idx': word2idx, 'label2idx': label2idx}
    """
    train_tweet_data_path, test_tweet_data_path = data_path
    tweet_list, tag_list = get_dict([train_tweet_data_path, test_tweet_data_path])
    tweet_list, tag_list = bert_tokenizer_qualification(tweet_list, tag_list)
    return tweet_list, tag_list

def bert_tokenizing(tweet_list, tag_list):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_tweet_list, tokenized_tag_list = [], []
    for tweet, tag in zip(tweet_list, tag_list):
        tokenized_tweet_list.append(tokenizer.tokenize(tweet))
        tokenized_tag_list.append(tokenizer.tokenize(tag))
    return tokenized_tweet_list, tokenized_tag_list

def remove_url(qualified_tweet_list, qualified_tag_list):
    url_removed_tweet, url_removed_tag = [], []
    for tweet, tag in zip(qualified_tweet_list, qualified_tag_list):
        # print(tweet)
        # print(tag)
        cleaned_sentence = re.sub(url_pattern, '', tweet)
        if tag in cleaned_sentence:
            url_removed_tweet.append(cleaned_sentence)
            url_removed_tag.append(tag)
        # print(cleaned_sentence)
        # print(tag in cleaned_sentence)
        # print("---------------------------------")
    return url_removed_tweet, url_removed_tag


def get_label(tokenized_tweet_list, tokenized_tag_list):
    y , z = [], []
    bad_cnt = 0
    begin = -1
    # 获取关键词的开始位置
    for tweet, tag in zip(tokenized_tweet_list, tokenized_tag_list):
        begin = -1
        for i in range(len(tweet)):
            the_begining_of_keyphrase_flag = True
            # 从0开始，头到尾把tag长度遍历一遍，如果只是碰巧和前边单词部分重复或者完全不一致,则置flag为false
            for j in range(len(tag)):
                if tweet[i+j] != tag[j]:
                    the_begining_of_keyphrase_flag = False
                    break
            if the_begining_of_keyphrase_flag:
                begin = i
                break
        # 实装label_y
        labels_y = [0]*len(tweet)
        for i in range(len(tag)):
            labels_y[begin+i] = 1
        y.append(labels_y)
        # 实装label_z 
        # labels2idx = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}
        labels_z = [0]*len(tweet)
        if len(tag) == 1:
            labels_z[begin] = labels2idx['S']
        elif len(tag) > 1:
            labels_z[begin] = labels2idx['B']
            for i in range(len(tag) - 2):
                labels_z[begin+i+1] = labels2idx['I']
            labels_z[begin+len(tag)-1] = labels2idx['E']
        z.append(labels_z)
    return y, z



if __name__ == '__main__':

    data_preparation_flag = True
    
    if data_preparation_flag:
        train_tweet_data = os.path.join("data", "original_data", "trnTweet")
        test_tweet_data = os.path.join("data", "original_data", "testTweet")

        data_path = [train_tweet_data, test_tweet_data]
        qualified_tweet_list, qualified_tag_list = get_train_test_dicts(data_path)

        # 去除网址
        qualified_tweet_list, qualified_tag_list = remove_url(qualified_tweet_list, qualified_tag_list)
        tokenized_tweet_list, tokenized_tag_list = bert_tokenizing(qualified_tweet_list, qualified_tag_list)
        # print(len(tokenized_tweet_list))
        labels_y, labels_z = get_label(tokenized_tweet_list, tokenized_tag_list)
        labels = [labels_y, labels_z]
        tokenzied_data = [tokenized_tweet_list, tokenized_tag_list]
        qualified_data = [qualified_tweet_list, qualified_tag_list]

        with open('labels.pkl', 'wb') as file:
            pickle.dump(labels, file)
        with open('tokenized.pkl', 'wb') as file:
            pickle.dump(tokenzied_data, file)
        with open('qualified_data.pkl', 'wb') as file:
            pickle.dump(qualified_data, file)
        print("Finish preparaing data")
    else:
        print()
        with open('labels.pkl', 'rb') as file:
            labels = pickle.load(file)
        with open('tokenized.pkl', 'rb') as file:
            tokenzied_data = pickle.load(file)
        with open('qualified_data.pkl', 'rb') as file:
            qualified_data = pickle.load(file)
        labels_y, labels_z = labels
        tokenized_tweet_list, tokenized_tag_list = tokenzied_data
        qualified_tweet_list, qualified_tag_list = qualified_data
    




