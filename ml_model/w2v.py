from gensim.models import word2vec
import pandas as pd

print("Loading data...")

train = pd.read_csv("data/train_set.csv")
new_train = train.rename(columns={'class': 'article_class'}, inplace=False)
y_train = pd.get_dummies(new_train['article_class'])
y = y_train.values
x_train = list(new_train.word_seg)

test = pd.read_csv("data/test_set.csv")
x_test = list(test.word_seg)

x_text = x_train+x_test

sentence_list = []

for i in range(len(x_text)):
    sentence_list.append(x_text[i].split())
print(sentence_list[:5])

model = word2vec.Word2Vec(sentence_list, size=300, window=1, min_count=10, workers=10)

model.save('w2v/w2v_w1.model')