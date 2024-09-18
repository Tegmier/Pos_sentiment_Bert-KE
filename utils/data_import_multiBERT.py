import pickle
import re
import pickle
import os
import logging
import spacy
from tqdm import tqdm

# 设置transformers库的日志级别为ERROR，仅显示Error及以上级别的消息
logging.getLogger("transformers").setLevel(logging.ERROR)

# 定义BIOES字典 O:out, B:begin  I:inside, E:end, S:start
labels2idx = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}

spacy_model = spacy.load('en_core_web_sm')

# 定义URLpattern正则表达式
url_pattern = re.compile(r'https?://\S+')
at_pattern = re.compile(r'@\w+')

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
    tweet_list_trn, tag_list_trn = get_data_list_from_path(trn_data_path)
    tweet_list_test, tag_list_test = get_data_list_from_path(test_data_path)
    tweet = tweet_list_trn+tweet_list_test
    tag = tag_list_trn+tag_list_test
    return tweet, tag

def bert_tokenizer_qualification(tweet_list, tag_list, tokenizer):
    count_total = 0
    count_same = 0
    qualified_tweet_list, qualified_tag_list = [], []
    for tweet, tag in zip(tweet_list, tag_list):
        count_total+=1
        tokenized_tag = tokenizer.tokenize(tag)
        ## 不被分词的关键词被拿出来了
        if len(tag.split()) == len(tokenized_tag):
            count_same+=1
            qualified_tweet_list.append(tweet)
            qualified_tag_list.append(tag)
    return qualified_tweet_list, qualified_tag_list

def get_train_test_dicts(data_path, tokenizer):
    train_tweet_data_path, test_tweet_data_path = data_path
    tweet_list, tag_list = get_dict([train_tweet_data_path, test_tweet_data_path])
    tweet_list, tag_list = bert_tokenizer_qualification(tweet_list, tag_list, tokenizer)
    return tweet_list, tag_list

def bert_tokenizing(tweet_list, tag_list, tokenizer):
    print("About to proceed BERT Tokenizing")
    tokenized_tweet_list, tokenized_tag_list = [], []
    for tweet, tag in tqdm(zip(tweet_list, tag_list), total=len(tweet_list)):
        tokenized_tweet_list.append(tokenizer.tokenize(tweet))
        tokenized_tag_list.append(tokenizer.tokenize(tag))
    print("BERT Tokenizing finished")
    return tokenized_tweet_list, tokenized_tag_list

def remove_url(qualified_tweet_list, qualified_tag_list):
    print(f"About to proceed url removal")
    print(f"number of tweets before removal: {len(qualified_tweet_list)}")
    url_removed_tweet, url_removed_tag = [], []
    for tweet, tag in tqdm(zip(qualified_tweet_list, qualified_tag_list), total=len(qualified_tweet_list)):
        cleaned_sentence = re.sub(url_pattern, '', tweet)
        if tag in cleaned_sentence:
            url_removed_tweet.append(cleaned_sentence)
            url_removed_tag.append(tag)
    print(f"number of tweets after removal: {len(url_removed_tweet)}")
    print("url removal finished")
    return url_removed_tweet, url_removed_tag

def remove_stop_words_markers(qualified_tweet_list, qualified_tag_list):
    print(f"About to proceed stop words and markers removal")
    print(f"number of tweets before removal: {len(qualified_tweet_list)}")
    tweets = []
    tags = []
    for tweet, tag in tqdm(zip(qualified_tweet_list, qualified_tag_list), total = len(qualified_tweet_list)):
        processed_tweet = re.sub(at_pattern, '', tweet)
        processed_tweet = spcay_process(processed_tweet)
        if tag in processed_tweet:
            tweets.append(spcay_process(processed_tweet))    
            tags.append(tag)
    print(f"number of tweets after removal: {len(tweets)}")
    print(f"stop words and markers removal is finished")
    return tweets, tags

def spcay_process(sentence):
    doc = spacy_model(sentence)
    processed_sen = ' '.join([token.text for token in doc if not token.is_punct])
    return processed_sen

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

def multi_bert_data_process(tokenizer, model_name):
    data_preparation_flag = False
    print("--------------------------------------------------------------------")
    print("Start Data Process:")
    
    if data_preparation_flag:
        print("Processing Data from Scretch")
        train_tweet_data = os.path.join("data", "original_data", "trnTweet")
        test_tweet_data = os.path.join("data", "original_data", "testTweet")

        data_path = [train_tweet_data, test_tweet_data]

        # 把分词前后一样的推特和tag筛选出来
        qualified_tweet_list, qualified_tag_list = get_train_test_dicts(data_path, tokenizer)
        print(f"The number of qualified but unprocessed tweets is: {len(qualified_tweet_list)}")

        # 去除网址
        qualified_tweet_list, qualified_tag_list = remove_url(qualified_tweet_list, qualified_tag_list)
        qualified_tweet_list, qualified_tag_list = remove_stop_words_markers(qualified_tweet_list, qualified_tag_list)
        tokenized_tweet_list, tokenized_tag_list = bert_tokenizing(qualified_tweet_list, qualified_tag_list, tokenizer)

        # labels_y, labels_z = get_label(tokenized_tweet_list, tokenized_tag_list)
        # labels = [labels_y, labels_z]
        tokenzied_data = [tokenized_tweet_list, tokenized_tag_list]
        qualified_data = [qualified_tweet_list, qualified_tag_list]
        with open(f'data/qualified_data_{model_name}.pkl', 'wb') as file:
            pickle.dump(qualified_data, file)
        print("Finish preparaing data")
    else:
        print("Loading Processed Data")
        # with open('labels.pkl', 'rb') as file:
        #     labels = pickle.load(file)
        with open(f'data/qualified_data_{model_name}.pkl', 'rb') as file:
            qualified_data = pickle.load(file)
        qualified_tweet_list, qualified_tag_list = qualified_data

def data_import(numberofdata, train_test_rate, model_name):

    with open(f'data/qualified_data_{model_name}.pkl', 'rb') as file:
        qualified_data = pickle.load(file)

    qualified_tweet_list, qualified_tag_list = qualified_data
    data = []
    for i in range(len(qualified_tweet_list)):
        data.append([qualified_tweet_list[i], qualified_tag_list[i]])
    data = data[:numberofdata]
    train_test_split = int(len(data)*train_test_rate)
    training_data = data[:train_test_split]
    test_data = data[train_test_split:]
    return training_data, test_data