# coding: utf-8

# 导入使用到的库
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
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
from data_helper import preprocess,preprocess_keras

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm

class TFMLPConfig(object):
    """CNN配置参数"""

    embedding_dim = 64  # 词向量维度
    seq_length = 2820641  # 序列长度
    num_classes = 19  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 7  # 卷积核尺寸
    vocab_size = 11021  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 1  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

config = TFMLPConfig()



column = "word_seg"
train = pd.read_csv('data/short_train.csv')
test = pd.read_csv('data/test_set.csv')
test_id = test["id"].copy()
vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(train[column])
test_term_doc = vec.transform(test[column])
new_train = train.rename(columns={'class': 'article_class'}, inplace=False)
y_train = pd.get_dummies(new_train['article_class'])
y = y_train.values

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = trn_term_doc[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(0.02 * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

del x_shuffled, y_shuffled

#print("Vocabulary Size: {:d}".format(len(x_train)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))



model = Sequential()

# 全连接层
model.add(Dense(1024, input_shape=(config.seq_length,), activation='relu'))
# DropOut层
#model.add(Dropout(0.5))

# 全连接层
model.add(Dense(512, input_shape=(1024,), activation='relu'))
# DropOut层
#model.add(Dropout(0.5))


# 全连接层+分类器
model.add(Dense(config.num_classes,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=config.batch_size,
          epochs=15,
          validation_data=(x_dev, y_dev))