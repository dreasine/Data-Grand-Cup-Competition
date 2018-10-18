import argparse
import tensorflow as tf
from data_utils import *
from sklearn.model_selection import train_test_split
from cnn_models.word_cnn import WordCNN
from cnn_models.char_cnn import CharCNN
from cnn_models.vd_cnn import VDCNN
from rnn_models.word_rnn import WordRNN
from rnn_models.attention_rnn import AttentionRNN
from rnn_models.rcnn import RCNN
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

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="word_cnn",
                    help="word_cnn | char_cnn | vd_cnn | word_rnn | att_rnn | rcnn")
args = parser.parse_args()

# if not os.path.exists("dbpedia_csv"):
#     print("Downloading dbpedia dataset...")
#     download_dbpedia()

NUM_CLASS = 19
BATCH_SIZE = 64
NUM_EPOCHS = 10
WORD_MAX_LEN = 100
CHAR_MAX_LEN = 1014


'''
print("Building dataset...")
if args.model == "char_cnn":
    x, y, alphabet_size = build_char_dataset("train", "char_cnn", CHAR_MAX_LEN)
elif args.model == "vd_cnn":
    x, y, alphabet_size = build_char_dataset("train", "vdcnn", CHAR_MAX_LEN)
else:
    word_dict = build_word_dict()
    vocabulary_size = len(word_dict)
    x, y = build_word_dataset("train", word_dict, WORD_MAX_LEN)

train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.15)
'''
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’'\"/|_#$%ˆ&*˜‘+=<>()[]{} "
alphabet_size = len(alphabet)

with tf.Session() as sess:
    if args.model == "word_cnn":
        model = WordCNN(config.vocab_size, config.seq_length, NUM_CLASS)
    elif args.model == "char_cnn":
        model = CharCNN(alphabet_size, CHAR_MAX_LEN, NUM_CLASS)
    elif args.model == "vd_cnn":
        model = VDCNN(alphabet_size, CHAR_MAX_LEN, NUM_CLASS)
    elif args.model == "word_rnn":
        model = WordRNN(config.vocab_size, config.seq_length, NUM_CLASS)
    elif args.model == "att_rnn":
        model = AttentionRNN(config.vocab_size, config.seq_length, NUM_CLASS)
    elif args.model == "rcnn":
        model = RCNN(config.vocab_size, config.seq_length, NUM_CLASS)
    else:
        raise NotImplementedError()

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())

    train_batches = batch_iter(x_train, y_train, BATCH_SIZE, NUM_EPOCHS)
    num_batches_per_epoch = (len(x_train) - 1) // BATCH_SIZE + 1
    max_accuracy = 0

    for x_batch, y_batch in train_batches:
        train_feed_dict = {
            model.x: x_batch,
            model.y: y_batch,
            model.is_training: True
        }

        _, step, loss = sess.run([model.optimizer, model.global_step, model.loss], feed_dict=train_feed_dict)

        if step % 100 == 0:
            print("step {0}: loss = {1}".format(step, loss))

        if step % 2000 == 0:
            # Test accuracy with validation data for each epoch.
            valid_batches = batch_iter(x_val, y_val, BATCH_SIZE, 1)
            sum_accuracy, cnt = 0, 0

            for valid_x_batch, valid_y_batch in valid_batches:
                valid_feed_dict = {
                    model.x: valid_x_batch,
                    model.y: valid_y_batch,
                    model.is_training: False
                }

                accuracy = sess.run(model.accuracy, feed_dict=valid_feed_dict)
                sum_accuracy += accuracy
                cnt += 1
            valid_accuracy = sum_accuracy / cnt

            print("\nValidation Accuracy = {1}\n".format(step // num_batches_per_epoch, sum_accuracy / cnt))

            # Save model
            if valid_accuracy > max_accuracy:
                max_accuracy = valid_accuracy
                saver.save(sess, "{0}/{1}.ckpt".format(args.model, args.model), global_step=step)
                print("Model is saved.\n")
