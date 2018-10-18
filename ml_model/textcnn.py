# coding: utf-8

# 导入使用到的库
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape,BatchNormalization
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.utils.np_utils import to_categorical
from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from data_helper import preprocess
from tensorflow.contrib import learn
import numpy as np
from scipy.sparse import csr_matrix
from gensim.models import Word2Vec

class MLPConfig(object):
    """CNN配置参数"""

    embedding_dim =256  # 词向量维度
    seq_length = 1000  # 序列长度
    num_classes = 19  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 7  # 卷积核尺寸
    vocab_size = 20000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 2  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

config = MLPConfig()


model = Word2Vec.load('w2v/w2v.model')
word2idx = {"_PAD": 0} # 初始化 `[word : token]` 字典，后期 tokenize 语料库就是用该词典。
vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
# 存储所有 word2vec 中所有向量的数组，留意其中多一位，词向量全为 0， 用于 padding
embeddings_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
for i in range(len(vocab_list)):
    word = vocab_list[i][0]
    word2idx[word] = i + 1
    embeddings_matrix[i + 1] = vocab_list[i][1]


# Load data
print("Loading data...")

train = pd.read_csv("data/train_set.csv")
new_train = train.rename(columns={'class': 'article_class'}, inplace=False)
y_train = pd.get_dummies(new_train['article_class'])
y = y_train.values


x_train = list(new_train.word_seg)

test = pd.read_csv("data/test_set.csv")
x_test = list(test.word_seg)

#x_text = x_train+x_test

train_idx = []
for line in x_train:
    l = []
    for word in line:
        if word in word2idx:
            l.append(word2idx[word])
    train_idx.append(l)

test_idx = []
for line in x_test:
    l = []
    for word in line:
        if word in word2idx:
            l.append(word2idx[word])
    test_idx.append(l)


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


#x = load_sparse_csr("tfidf/train_tfidf_mindf001.npz")
#x_test = load_sparse_csr("tfidf/test_tfidf_mindf001.npz")


#vocab_dir = "vocab_dir/vocab"


#vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_dir)
x = pad_sequences(train_idx, maxlen=1000)
x_test = pad_sequences(test_idx,maxlen = 1000)
#x_test = pad_sequences(x_test, maxlen=1000)
#x = pad_sequences(x, maxlen=1000)

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(0.02 * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

del x, y, x_shuffled, y_shuffled

print("Vocabulary Size: {:d}".format(len(word2idx)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))





# x_train, y_train, vocab_processor, x_val, y_val,x_test = preprocess()
# print(x_train.shape)


# 模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接
main_input = Input(shape=(config.seq_length,), dtype='float64')
# 词嵌入（使用预训练的词向量）
#embedder = Embedding(config.vocab_size + 1, 300, input_length = 20, weights = [embedding_matrix], trainable = False)
#embed = embedder(main_input)
embed = Embedding(len(embeddings_matrix),300,weights=[embeddings_matrix],trainable=True)(main_input)
# 词窗大小分别为3,4,5

cnn1 = Convolution1D(256, 3, padding='same', strides = 1, activation='relu')(embed)
cnn1 = MaxPool1D(pool_size=4)(cnn1)
cnn2 = Convolution1D(256, 4, padding='same', strides = 1, activation='relu')(embed)
cnn2 = MaxPool1D(pool_size=4)(cnn2)
cnn3 = Convolution1D(256, 5, padding='same', strides = 1, activation='relu')(embed)
cnn3 = MaxPool1D(pool_size=4)(cnn3)
cnn4 = Convolution1D(256, 8, padding='same', strides = 1, activation='relu')(embed)
cnn4 = MaxPool1D(pool_size=4)(cnn4)
# 合并三个模型的输出向量
cnn = concatenate([cnn1,cnn2,cnn3,cnn4], axis=-1)
flat = Flatten()(cnn)
drop = Dropout(0.5)(flat)
main_output = Dense(config.num_classes, activation='softmax')(drop)
model = Model(inputs = main_input, outputs = main_output)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=64,
          epochs=2,
          validation_data=(x_dev, y_dev))

model.save('test.h5')

all_predictions = []

all_predictions = model.predict(x_train,batch_size=64)


print(all_predictions.shape)
print(type(all_predictions),all_predictions[:10])
print(y_train[:10])
score=model.evaluate(x_train,y_train,64,verbose=0)
print(score)



'''
print("Loading data...")

test = pd.read_csv("data/test_set.csv")

x_test = list(test.word_seg)

vocab_dir = "vocab_dir/vocab"

vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_dir)
x = np.array(list(vocab_processor.fit_transform(x_test)))
x_test = pad_sequences(x, maxlen=1000)
'''

all_predictions = []

all_predictions = model.predict(x_test,batch_size=64)


print(all_predictions.shape)
print(type(all_predictions),all_predictions[:5])
# x_train, y_train,
#           batch_size=64,
#           epochs=15,
#           validation_data=(x_val, y_val)


test_id = pd.read_csv("data/back_test.csv")


# config = TRNNConfig()
# model = TextRNN(config)

test_prob=pd.DataFrame(all_predictions)
test_prob.columns=["class_prob_%s"%i for i in range(1,all_predictions.shape[1]+1)]
test_prob["id"]=list(test_id["id"])
test_prob.to_csv('prob_textcnn_test.csv',index=None)


all_predictions=np.argmax(all_predictions,axis=1)


test_pred=pd.DataFrame(all_predictions)
test_pred.columns=["class"]
test_pred["class"]=(test_pred["class"]+1).astype(int)
print(test_pred.shape)
print(test_id.shape)
test_pred["id"]=list(test_id["id"])
test_pred[["id","class"]].to_csv('textcnn_test.csv',index=None)