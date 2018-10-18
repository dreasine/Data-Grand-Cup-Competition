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


class MLPConfig(object):
    """CNN配置参数"""

    embedding_dim = 64  # 词向量维度
    seq_length = 1000  # 序列长度
    num_classes = 19  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 7  # 卷积核尺寸
    vocab_size = 11021  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

config = MLPConfig()

x_train, y_train, vocab_processor, x_val, y_val = preprocess()


model = Sequential()


# 模型结构：嵌入-卷积池化*2-dropout-BN-全连接-dropout-全连接
model.add(Embedding(11021, 300, input_length=1000))
model.add(Convolution1D(256, 3, padding='same'))
model.add(MaxPool1D(3,3,padding='same'))
model.add(Convolution1D(128, 3, padding='same'))
model.add(MaxPool1D(3,3,padding='same'))
model.add(Convolution1D(64, 3, padding='same'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(BatchNormalization()) # (批)规范化层
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(19,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=64,
          epochs=15,
          validation_data=(x_val, y_val))

model.save('lenet5_model.h5')