import numpy as np
import re
import pickle
from collections import Counter
import os
from transformers import BertTokenizer, BertModel
from typing import List, Tuple


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
            if count == 9:
                break
        return tweet_list, tag_list 
            


def get_dict(data_path):
    trn_data_path, test_data_path = data_path
    tweet_list, tag_list = get_data_list_from_path(trn_data_path)
    return tweet_list, tag_list


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
    return tweet_list, tag_list
    # dicts, trn_data, testdata = get_dict([train_tweet_data_path, test_tweet_data_path])

def tokenize_and_preserve_labels(sentences: List[List[str]], labels: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
    tokenized_texts = []
    tokenized_labels = []

    for sentence, label_seq in zip(sentences, labels):
        tokenized_sentence = []
        label_sequence = []

        for word, label in zip(sentence, label_seq):
            # 对每个单词进行分词
            tokenized_word = tokenizer.tokenize(word)
            tokenized_sentence.extend(tokenized_word)
            
            # 对分词后的每个子词复制标签
            label_sequence.extend([label] * len(tokenized_word))

        tokenized_texts.append(tokenized_sentence)
        tokenized_labels.append(label_sequence)

    return tokenized_texts, tokenized_labels


if __name__ == '__main__':

    train_tweet_data = os.path.join("data", "original_data", "trnTweet")
    test_tweet_data = os.path.join("data", "original_data", "testTweet")
    data_path = [train_tweet_data, test_tweet_data]
    # data_set = get_train_test_dicts(data_path)
    tweet_list, tag_list = get_train_test_dicts(data_path)
    print(tweet_list)
    print(tag_list)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # input_tweet = list(tweet_list[0].split())
    # input_tag = list(tag_list[0].split())
    input_tweet = tweet_list[0]
    input_tag = tag_list[0]
    tokenized = tokenizer.tokenize(input_tweet)
    print(tokenized)


