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
from scipy.sparse import csr_matrix
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm

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

#x_train, y_train,vocab, x_val, y_val,x_test= preprocess()


column = "word_seg"
train = pd.read_csv('data/train_set.csv')
train = train.rename(columns={'class': 'article_class'}, inplace=False)
test = pd.read_csv('data/test_set.csv')
test_id = test["id"].copy()
# vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
# trn_term_doc = vec.fit_transform(train[column])
# test_term_doc = vec.transform(test[column])


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


trn_term_doc = load_sparse_csr("train_tfidf_normal50.npz")
test_term_doc = load_sparse_csr("test_tfidf_normal50.npz")


y_train = pd.get_dummies(train['article_class'])
y = y_train.values



model = Sequential()

# 全连接层
model.add(Dense(1024, input_shape=(2820641,), activation='relu'))
# DropOut层
#model.add(Dropout(0.5))

# 全连接层
model.add(Dense(512, input_shape=(1024,), activation='relu'))
# DropOut层
#model.add(Dropout(0.5))


# 全连接层+分类器
model.add(Dense(19,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(trn_term_doc, y,
          batch_size=config.batch_size,
          epochs=15)

model.save('mlp.h5')


all_predictions = []

all_predictions = model.predict(test_term_doc,batch_size=64)


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
test_prob.to_csv('prob_textcnn_test.csv',index=None)


all_predictions=np.argmax(all_predictions,axis=1)


test_pred=pd.DataFrame(all_predictions)
test_pred.columns=["class"]
test_pred["class"]=(test_pred["class"]+1).astype(int)
print(test_pred.shape)
print(test_id.shape)
test_pred["id"]=list(test_id["id"])
test_pred[["id","class"]].to_csv('textcnn_test.csv',index=None)