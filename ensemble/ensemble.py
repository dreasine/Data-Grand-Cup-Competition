import numpy as np
import pandas as pd


gru = pd.read_csv('prob_enhance_lr_baseline.csv') # PL score 0.9829
lstm_nb_svm = pd.read_csv('../input/minimal-lstm-nb-svm-baseline-ensemble/submission.csv') # 0.9811
lr = pd.read_csv('../input/logistic-regression-with-words-and-char-n-grams/submission.csv') # 0.9788
lgb = pd.read_csv('../input/lightgbm-with-select-k-best-on-tfidf/lgb_submission.csv') # 0.9785
textCNN = pd.read_csv('../input/textcnn-2d-convolution/submission.csv') # 0.9821
GRUCNN = pd.read_csv('../input/bi-gru-cnn-poolings/submission.csv') # 0.9841
LGBM = pd.read_csv('../input/lgbm-with-words-and-chars-n-gram/lvl0_lgbm_clean_sub.csv') # 0.9792
Blend = pd.read_csv('../input/blend-it-all/blend_it_all.csv') # 0.9867
LightGBM = pd.read_csv('../input/lightgbm-with-select-k-best-on-tfidf/lgb_submission.csv') # 0.97..