import numpy as np
import re
import pickle
from collections import Counter
import os
from transformers import BertTokenizer, BertModel
from typing import List, Tuple

import logging

# 设置transformers库的日志级别为ERROR，仅显示Error及以上级别的消息
logging.getLogger("transformers").setLevel(logging.ERROR)


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
    print("Total number of Tags is: " + str(count_total))
    print("Total number of qualified Tags is: " + str(count_same))
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

if __name__ == '__main__':

    train_tweet_data = os.path.join("data", "original_data", "trnTweet")
    test_tweet_data = os.path.join("data", "original_data", "testTweet")

    data_path = [train_tweet_data, test_tweet_data]
    tweet_list, tag_list = get_train_test_dicts(data_path)
    tokenized_tweet_list, tokenized_tag_list = bert_tokenizing(tweet_list, tag_list)





