

import pandas as pd
import numpy as np
import tensorflow as tf
#from rnn_model import TRNNConfig, TextRNN
from tensorflow.contrib import learn
import os
from sklearn.model_selection import train_test_split


# 导入使用到的库
from keras import preprocessing
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
from sklearn import preprocessing as pr
import pandas as pd
import numpy as np




# Load data
print("Loading data...")

train = pd.read_csv("data/train_set.csv")
new_train = train.rename(columns={'class': 'article_class'}, inplace=False)
y_train = pd.get_dummies(new_train['article_class'])
y = y_train.values
x_train = list(new_train.word_seg)

test = pd.read_csv("data/test_set.csv")
x_test = list(test.word_seg)

x_text = x_train+x_test



# Build vocabulary
#max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(1000)
#x = np.array(list(vocab_processor.fit_transform(x_text)))




#vocab_dir = "vocab_dir/vocab"

#vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_dir)
vocab_processor.fit(x_text)

print(vocab_processor.vocabulary_)

#x = np.array(list(vocab_processor.transform(x_train)))
#x_test = np.array(list(vocab_processor.transform(x_test)))


# Write vocabulary
vocab_dir = "vocab_dir"
if not os.path.exists(vocab_dir):
    os.makedirs(vocab_dir)

vocab_processor.save(os.path.join(vocab_dir, "vocab"))