import pickle
from collections import Counter
def voc_create():
    with open('data/qualified_data.pkl', 'rb') as file:
        qualified_data = pickle.load(file)

    qualified_tweet_list, qualified_tag_list = qualified_data
    voc = []
    for snetence in qualified_tweet_list:
        word_list = snetence.lower().split()
        voc.extend(word_list)
    word_counts = Counter(voc)
    word2idx = {word[0]: i+1 for i,word in enumerate(word_counts.most_common())}

    lex, y, z = [], [], []
    for tweet, tag in zip(qualified_tweet_list, qualified_tag_list):
        word_list = tweet.lower().split()
        t_list = tag.lower().split()

        emb = [word2idx[x] for x in word_list]

        bad_cnt = 0
        begin = -1
        for i in range(len(word_list)):
            ok = True
            for j in range(len(t_list)):
                if word_list[i+j] != t_list[j]:
                    ok = False
                    break
            if ok:
                begin = i
                break
        if begin == -1:
            bad_cnt += 1
            continue

        lex.append(emb)
        labels2idx = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}

        labels_y = [0]*len(word_list)
        for i in range(len(t_list)):
            labels_y[begin+i] = 1
        y.append(labels_y)

        labels_z = [0]*len(word_list)
        if len(t_list) == 1:
            labels_z[begin] = labels2idx['S']
        elif len(t_list) > 1:
            labels_z[begin] = labels2idx['B']
            for i in range(len(t_list)-2):
                labels_z[begin+i+1] = labels2idx['I']
            labels_z[begin+len(t_list)-1] = labels2idx['E']
        z.append(labels_z)

    return lex,y,z,word2idx