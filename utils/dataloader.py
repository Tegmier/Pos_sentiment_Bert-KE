# 实现通过Dataloader读取数据的方法
from torch.utils.data import Dataset
import torch

labels2idx = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}

class KE_Dataloader(Dataset):
    def __init__(self, data) -> None:
        self.tweets = [data[i][0] for i in range(len(data))]
        self.tags = [data[i][1] for i in range(len(data))]

    def __len__(self):
        return len(self.tweets)
    
    def __getitem__(self, index):
        return self.tweets[index], self.tags[index]


def batch_padding_tokenizing_collate_function(batch, tokenizer):
    tweets, tags = zip(*batch)
    tweets = list(tweets)
    tags = list(tags)
    encoded_tweet = tokenizer.batch_encode_plus(
            tweets,
            is_split_into_words=False,  # 指示输入已经是分好词的
            add_special_tokens=True,  # 添加特殊token，如[CLS]和[SEP]
            padding=True,  # 填充到最大长度
            truncation=True,  # 截断超过最大长度的句子
            return_tensors='pt'  # 返回PyTorch张量
        )
    tag_tensor = []
    for t in tags:
        encoded_tag = tokenizer.encode_plus(
            t,
            is_split_into_words=False,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors='pt'
            )
        tag_tensor.append(encoded_tag["input_ids"][0])
    
    y, z = get_label(encoded_tweet, tag_tensor)
    inputs = {}
    inputs["input_ids"] = encoded_tweet["input_ids"]
    inputs["attention_mask"] = encoded_tweet["attention_mask"]
    inputs["label_y"] = torch.tensor(y, dtype=torch.int64)
    inputs["mask_y"] = torch.tensor(y, dtype=torch.int64)!=0
    inputs["label_z"] = torch.tensor(z, dtype=torch.int64)
    inputs["mask_z"] = torch.tensor(z, dtype=torch.int64)!=0
    return inputs

def get_label(encoded_tweet, encoded_tag):
    y , z = [], []
    begin = -1
    # 获取关键词的开始位置
    encoded_tweet_input = encoded_tweet["input_ids"]
    encoded_tag_input = encoded_tag
    for k in range(encoded_tweet_input.size(0)):
        begin = -1
        for i in range(len(encoded_tweet_input[k])):
            the_begining_of_keyphrase_flag = True
            # 从0开始，头到尾把tag长度遍历一遍，如果只是碰巧和前边单词部分重复或者完全不一致,则置flag为false
            for j in range(len(encoded_tag_input[k])):
                if encoded_tweet_input[k][i+j] != encoded_tag_input[k][j]:
                    the_begining_of_keyphrase_flag = False
                    break
            if the_begining_of_keyphrase_flag:
                begin = i
                break
        # 实装label_y
        labels_y = [0]*encoded_tweet_input.size(1)
        for i in range(len(encoded_tag_input[k])):
            labels_y[begin+i] = 1
        y.append(labels_y)
        # 实装label_z 
        # labels2idx = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}
        labels_z = [0]*len(encoded_tweet_input[k])
        if len(encoded_tag_input[k]) == 1:
            labels_z[begin] = labels2idx['S']
        elif len(encoded_tag_input[k]) > 1:
            labels_z[begin] = labels2idx['B']
            for i in range(len(encoded_tag_input[k]) - 2):
                labels_z[begin+i+1] = labels2idx['I']
            labels_z[begin+len(encoded_tag_input[k])-1] = labels2idx['E']
        z.append(labels_z)
    return y, z
