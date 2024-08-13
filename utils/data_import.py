import pickle

def data_import(numberofdata, train_test_rate):
    with open('data/tokenized.pkl', 'rb') as file:
        tokenzied_data = pickle.load(file)
        tokenized_tweet_list, tokenized_tag_list = tokenzied_data

    with open('data/qualified_data.pkl', 'rb') as file:
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