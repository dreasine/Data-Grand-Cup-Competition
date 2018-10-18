
# coding: utf-8

# In[1]:


#coding=utf-8
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
from tensorflow.python.ops.rnn import static_rnn
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
import pandas as pd



# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .01, "Percentage of the training data to use for validation")
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



# In[2]:


# 数据



positive_texts = [
    "我 今天 很 高兴",
    "我 很 开心",
    "他 很 高兴",
    "他 很 开心"
]
negative_texts = [
    "我 不 高兴",
    "我 不 开心",
    "他 今天 不 高兴",
    "他 不 开心"
]

label_name_dict = {
    0: "正面情感",
    1: "负面情感"
}


train = pd.read_csv("data/train_set.csv")
new_train = train.rename(columns={'class': 'article_class'}, inplace=False)
y_train = pd.get_dummies(new_train['article_class'])
y = y_train.values
x_text = list(new_train.word_seg)


# In[3]:


embedding_size = 50
num_classes = 19

all_texts = positive_texts + negative_texts
labels = [0] * len(positive_texts) + [1] * len(negative_texts)
max_document_length = 1000
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

datas = np.array(list(vocab_processor.fit_transform(x_text)))
vocab_size = len(vocab_processor.vocabulary_)


# In[4]:


datas_placeholder = tf.placeholder(tf.int32,[None,max_document_length])
labels_placeholder = tf.placeholder(tf.int32,[None])


# In[5]:


embeddings = tf.get_variable("embeddings",[vocab_size,embedding_size],initializer = tf.truncated_normal_initializer)

embedded = tf.nn.embedding_lookup(embeddings,datas_placeholder)


# In[6]:


rnn_input = tf.unstack(embedded,max_document_length,axis=1)


# In[7]:


lstm_cell = BasicLSTMCell(20,forget_bias=1.0)
rnn_outputs,rnn_states = static_rnn(lstm_cell,rnn_input,dtype=tf.float32)

logits = tf.layers.dense(rnn_outputs[-1],num_classes)

predicted_labels = tf.argmax(logits,axis = 1)


# In[8]:


losses = tf.nn.softmax_cross_entropy_with_logits(
    labels = tf.one_hot(labels_placeholder,num_classes),
    logits = logits
)

mean_loss = tf.reduce_mean(losses)
optimizer = tf.train.AdamOptimizer(learning_rate = 1e-2).minimize(mean_loss)




def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


x_train = datas
y_train = labels


batches = batch_iter(
    list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)


# In[12]:

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
#with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # feed_dict = {
    # datas_placeholder:datas,
    # labels_placeholder:labels
    # }

    print("Start train...")
    for step in range(100):
        i = 0
        batches = batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        for batch in batches:
            i += 1
            x_batch,y_batch = zip(*batch)
            feed_dict = {
                datas_placeholder: x_batch,
                labels_placeholder: y_batch
            }
            _,mean_loss_val = sess.run([optimizer,mean_loss],feed_dict)
            print("batch = {}\tmean loss = {}".format(i, mean_loss_val))

        if step % 10 == 0:
            print("step = {}\tmean loss = {}".format(step,mean_loss_val))

    print('train end... start predict...')

    predicted_labels_val = sess.run(predicted_labels,feed_dict=feed_dict)
    for i,text in enumerate(all_texts):
        label = predicted_labels_val[i]
        label_name = label_name_dict[label]
        print("{} => {}".format(text,label_name))

