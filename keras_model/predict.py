import numpy as np
import pandas as pd
import tensorflow as tf
#from rnn_model import TRNNConfig, TextRNN
from tensorflow.contrib import learn

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

from keras.models import load_model
from data_helper import preprocess

import os




print("Loading data...")

test = pd.read_csv("data/test_set.csv")

x_test = list(test.word_seg)

vocab_dir = "vocab_dir/vocab"

vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_dir)
x = np.array(list(vocab_processor.fit_transform(x_test)))
x_test = pad_sequences(x, maxlen=1000)


model = load_model('test.h5')

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
test_prob.to_csv('prob_enhance_test.csv',index=None)

all_predictions=np.argmax(all_predictions,axis=1)



test_pred=pd.DataFrame(all_predictions)
test_pred.columns=["class"]
test_pred["class"]=(test_pred["class"]+1).astype(int)
print(test_pred.shape)
print(test_id.shape)
test_pred["id"]=list(test_id["id"])
test_pred[["id","class"]].to_csv('enhance_test.csv',index=None)


x_train, y_train, vocab_processor, x_val, y_val = preprocess()
score=model.evaluate(x_train,y_train,64,verbose=0)
print(score)

    #print("Model restored.")
    # Check the values of the variables
    #print("embedding : %s" % embedding.eval())
    # print("v1 : %s" % v1.eval())
    # print("v2 : %s" % v2.eval())

#生成提交结果
# preds=np.argmax(preds,axis=1)
# test_pred=pd.DataFrame(preds)
# test_pred.columns=["class"]
# test_pred["class"]=(test_pred["class"]+1).astype(int)
# print(test_pred.shape)
# print(test_id.shape)
# test_pred["id"]=list(test_id["id"])
# test_pred[["id","class"]].to_csv('sub_lr_baseline.csv',index=None)
# t2=time.time()
# print("time use:",t2-t1)

