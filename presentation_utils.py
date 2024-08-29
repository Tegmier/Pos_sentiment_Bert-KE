from transformers import BertModel, BertTokenizer
sentence = "Unpredictability in hyper-localized microclimates challenges meteorologists"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenized_result = tokenizer.tokenize(sentence)
print(tokenized_result)