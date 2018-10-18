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

x_train, y_train, vocab_processor, x_val, y_val,x_test = preprocess()
print(x_train.shape)


# 模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接
main_input = Input(shape=(config.seq_length,), dtype='float64')
# 词嵌入（使用预训练的词向量）
#embedder = Embedding(config.vocab_size + 1, 300, input_length = 20, weights = [embedding_matrix], trainable = False)
#embed = embedder(main_input)
embed = Embedding(config.vocab_size + 1, 300, input_length=config.seq_length)(main_input)
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
#lstm = Bidirectional(LSTM(128, return_sequences = True))(cnn)
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
          validation_data=(x_val, y_val))

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


test_id = pd.read_csv("data/test_set.csv")


# config = TRNNConfig()
# model = TextRNN(config)

test_prob=pd.DataFrame(all_predictions)
test_prob.columns=["class_prob_%s"%i for i in range(1,all_predictions.shape[1]+1)]
test_prob["id"]=list(test_id["id"])
test_prob.to_csv('result/prob_long_textcnn_test.csv',index=None)


all_predictions=np.argmax(all_predictions,axis=1)


test_pred=pd.DataFrame(all_predictions)
test_pred.columns=["class"]
test_pred["class"]=(test_pred["class"]+1).astype(int)
print(test_pred.shape)
print(test_id.shape)
test_pred["id"]=list(test_id["id"])
test_pred[["id","class"]].to_csv('result/long_textcnn_test.csv',index=None)