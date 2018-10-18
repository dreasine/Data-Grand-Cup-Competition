import tensorflow as tf
from tensorflow.contrib import rnn
from data_helper import preprocess


class MLPConfig(object):
    """CNN配置参数"""

    embedding_dim =256  # 词向量维度
    seq_length = 1000  # 序列长度
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

x_train, y_train, vocab_processor, x_val, y_val,x_test = preprocess()


class WordRNN(object):
    def __init__(self, vocabulary_size, document_max_len, num_class):
        self.embedding_size = 256
        self.num_hidden = 256
        self.num_layers = 2
        self.learning_rate = 1e-3

        self.x = tf.placeholder(tf.int32, [None, document_max_len], name="x")
        self.x_len = tf.reduce_sum(tf.sign(self.x), 1)
        self.y = tf.placeholder(tf.int32, [None], name="y")
        self.is_training = tf.placeholder(tf.bool, [], name="is_training")
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_prob = tf.where(self.is_training, 0.5, 1.0)

        with tf.name_scope("embedding"):
            init_embeddings = tf.random_uniform([vocabulary_size, self.embedding_size])
            self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            self.x_emb = tf.nn.embedding_lookup(self.embeddings, self.x)

        with tf.name_scope("birnn"):
            fw_cells = [rnn.BasicLSTMCell(self.num_hidden) for _ in range(self.num_layers)]
            bw_cells = [rnn.BasicLSTMCell(self.num_hidden) for _ in range(self.num_layers)]
            fw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob) for cell in fw_cells]
            bw_cells = [rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob) for cell in bw_cells]

            rnn_outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                fw_cells, bw_cells, self.x_emb, sequence_length=self.x_len, dtype=tf.float32)
            rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, document_max_len * self.num_hidden * 2])

        with tf.name_scope("output"):
            self.logits = tf.layers.dense(rnn_outputs_flat, num_class, activation=None)
            self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")