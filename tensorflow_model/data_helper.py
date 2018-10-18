


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


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .02, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 64, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

def preprocess_keras():
    # 划分训练/测试集
    train = pd.read_csv("data/long_train.csv")
    new_train = train.rename(columns={'class': 'article_class'}, inplace=False)
    #y_train = pd.get_dummies(new_train['article_class'])
    y = new_train.article_class.values
    x_text = new_train.word_seg.values
    X_train, X_test, y_train, y_test = train_test_split(x_text, y, test_size=0.1, random_state=42)

    # 对类别变量进行编码，共10类
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    y_labels = list(y_train.value_counts().index)
    le = pr.LabelEncoder()
    le.fit(y_labels)
    num_labels = len(y_labels)
    y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), num_labels)
    y_test = to_categorical(y_test.map(lambda x: le.transform([x])[0]), num_labels)

    # 分词，构建单词-id词典
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
    tokenizer.fit_on_texts(x_text)
    vocab = tokenizer.word_index

    # 将每个词用词典中的数值代替
    X_train_word_ids = tokenizer.texts_to_sequences(X_train)
    X_test_word_ids = tokenizer.texts_to_sequences(X_test)

    # One-hot
    x_train_o = tokenizer.sequences_to_matrix(X_train_word_ids, mode='binary')
    x_test_o = tokenizer.sequences_to_matrix(X_test_word_ids, mode='binary')

    # 序列模式
    x_train_p = pad_sequences(X_train_word_ids, maxlen=20)
    x_test_p = pad_sequences(X_test_word_ids, maxlen=20)

    return x_train_o,y_train,vocab,x_test_o,y_test


def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")

    train = pd.read_csv("data/train_set.csv")
    #back_train = pd.read_csv('data/back_train.csv')
    new_train = train.rename(columns={'class': 'article_class'}, inplace=False)
    #y_train = pd.get_dummies(new_train['article_class'])
    y_train = new_train['article_class']
    #back_y = pd.get_dummies(back_train['article_class']).values
    y = y_train.values
    #y = np.concatenate((y, back_y), axis=0)


    #back_train = list(back_train.word_seg)
    #print(type(back_train),len(back_train))
    #print(type(back_train[0]),back_train[0])
    new_train = list(new_train.word_seg)
    #print(type(new_train),len(new_train))
    #print(type(new_train[0]),new_train[0])
    #x_train = new_train+back_train
    x_train = new_train

    #print(len(x_train))

    test = pd.read_csv("data/test_set.csv")
    x_test = list(test.word_seg)

    x_text = x_train+x_test
    #print(len(x_text))



    # Build vocabulary
    #max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(1000,500)
    #x = np.array(list(vocab_processor.fit_transform(x_text)))




    vocab_dir = "vocab_dir/vocab"

    #vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_dir)
    vocab_processor.fit(x_text)
    x = np.array(list(vocab_processor.transform(x_train)))
    #print(len(x))
    x_test = np.array(list(vocab_processor.transform(x_test)))
    #x_test = pad_sequences(x_test, maxlen=1000)
    #x = pad_sequences(x, maxlen=1000)

    # Write vocabulary
    vocab_dir = "vocab_dir"
    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)

    vocab_processor.save(os.path.join(vocab_dir, "vocab"))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev,x_test

