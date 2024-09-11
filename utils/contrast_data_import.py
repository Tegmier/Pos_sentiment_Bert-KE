def contrast_data_import(numberofdata, train_test_rate, lex, y ,z):
    data = []
    for i in range(len(lex)):
        data.append([lex[i], y[i], z[i]])
    data = data[:numberofdata]
    train_test_split = int(len(data)*train_test_rate)
    training_data = data[:train_test_split]
    test_data = data[train_test_split:]
    return training_data, test_data