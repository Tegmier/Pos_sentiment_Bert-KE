from transformers import BertModel, BertTokenizer
sentence = "Unpredictability in hyper-localized microclimates challenges meteorologists"
sentence2 = "Republicans push Senate Dems toward hard votes on  iran"
sentence3 = "A few things for the ankle-biters...ShopSmall and  shop main street (@ Tiny Toes) [pic]:"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenized_result = tokenizer.tokenize(sentence)
print(tokenized_result)
print(tokenizer.tokenize(sentence3))