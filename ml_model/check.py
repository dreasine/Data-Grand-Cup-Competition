import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
import tqdm

train = pd.read_csv('data/train_set.csv')
test = pd.read_csv("data/test_set.csv")
word_list=[]

print("strat train...")
for line in train.word_seg:
    line = line.split()
    for code in line:
        if int(code) not in word_list:
            word_list.append(int(code))

print("strat test...")
for line in test.word_seg:
    line = line.split()
    for code in line:
        if int(code) not in word_list:
            word_list.append(int(code))

word_list.sort()
print(len(word_list))
print(word_list[:100])
print(word_list[-100:])

'''

def build_vocab(vocab_path, data):
    files = open(vocab_path, 'w', encoding='utf-8')
    files.write(
        "{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<SOS>", "<EOS>"))
    for word, count in data:
        files.write("{}\t{}\n".format(word, count))


def mini_batch(vocab_path, data, padding):
    token_seqs, sentences = load_datasets(vocab_path, data, padding)
    num_batch = int(len(sentences) / seq_batch_size)
    token_seqs = token_seqs[:num_batch * seq_batch_size]
    sentences = sentences[:num_batch * seq_batch_size]
    token_batch = np.split(np.array(token_seqs), num_batch, 0)
    sentence_batch = np.split(np.array(sentences), num_batch, 0)
    return token_batch, sentence_batch, num_batch


def load_datasets(vocab_path, data, padding=False):
    sentences = [line for line in data if line]
    word2idx, idx2word = load_vocab(vocab_path)

    token_list, sources = [], []
    for source in sentences:
        temp_seg = jieba.cut(source, cut_all=False)
        seg_list = [i for i in temp_seg]
        x = [word2idx.get(word, 1) for word in (" ".join(str(i) for i in seg_list) + ' <EOS>').split()]
        if padding:
            if len(x) < seq_length:
                x += [0 for _ in range(seq_length - len(x))]
            else:
                x = x[:100]
        token_list.append(x)
        sources.append(source)
    return token_list, sources


def load_vocab(vocab_path):
    vocab = [line.split()[0] for line in open(vocab_path, 'r', encoding='utf-8').read().splitlines()]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {word2idx[word]: word for word in word2idx}
    return word2idx, idx2word


def next_batch(token_batches, sentence_batches, pointer, num_batch):
    result = token_batches[pointer]
    sentence = sentence_batches[pointer]
    pointer = (pointer + 1) % num_batch
    return result, sentence

'''