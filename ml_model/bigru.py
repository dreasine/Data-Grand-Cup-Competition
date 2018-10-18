import numpy as np
np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv1D, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from data_helper import preprocess,preprocess2

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=0)


class MLPConfig(object):
    """CNN配置参数"""

    embedding_dim =256  # 词向量维度
    seq_length = 2000  # 序列长度
    num_classes = 19  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 7  # 卷积核尺寸
    vocab_size = 22924  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 2  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

config = MLPConfig()

x_train, y_train, vocab_processor, x_val, y_val,x_test = preprocess2()

def get_model():
    inp = Input(shape=(config.seq_length,))
    x = Embedding(config.vocab_size + 1, 256, input_length=config.seq_length)(inp)
    x = SpatialDropout1D(0.2)(x)
    x1 = Bidirectional(GRU(128, return_sequences=True))(x)
    x2 = Bidirectional(GRU(64, return_sequences=True))(x)
    conc = concatenate([x1, x2])
    avg_pool = GlobalAveragePooling1D()(conc)
    max_pool = GlobalMaxPooling1D()(conc)
    conc = concatenate([avg_pool, max_pool])
    x = Dense(64, activation='relu')(conc)
    x = Dropout(0.2)(x)
    outp = Dense(config.num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

model = get_model()


batch_size = 32
epochs = 3


hist = model.fit(x_train, y_train, batch_size=64, epochs=epochs, validation_data=(x_val, y_val),
                 callbacks=[early_stopping])


model.save('model/bigru.h5')

all_predictions = []

all_predictions = model.predict(x_train,batch_size=64)


print(all_predictions.shape)
print(type(all_predictions),all_predictions[:10])
print(y_train[:10])
score=model.evaluate(x_train,y_train,64,verbose=0)
print(score)


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
test_prob.to_csv('result/prob/prob_bigru_split.csv',index=None)


all_predictions=np.argmax(all_predictions,axis=1)


test_pred=pd.DataFrame(all_predictions)
test_pred.columns=["class"]
test_pred["class"]=(test_pred["class"]+1).astype(int)
print(test_pred.shape)
print(test_id.shape)
test_pred["id"]=list(test_id["id"])
test_pred[["id","class"]].to_csv('result/bigru_split.csv',index=None)