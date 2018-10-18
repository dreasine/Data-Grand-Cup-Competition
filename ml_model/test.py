import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

train = pd.read_csv('data/train_set.csv')


all_article = []
for line in tqdm(train.word_seg):
    line = line.split()
    #article = list(set(all_top).intersection(set(line)))
    for word in line:
        if word in all_top:
            article.append(word)
    all_article.append(article)

print(len(all_article),len(all_article[0]),all_article[:5])


# data = np.arange(6).reshape(2,3)
# print(data)
#
#
# test = pd.read_csv('prob_lr_baseline.csv')
# print(test.head())
#
# sub = pd.read_csv('sub_lr_baseline.csv')
# print(sub.head())