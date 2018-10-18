import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from scipy.sparse import csr_matrix

import time


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


t1=time.time()
train = pd.read_csv('data/train_set.csv')
train = train.rename(columns = {'class': 'article_class'}, inplace=False)
test = pd.read_csv('data/test_set.csv')
test_id = pd.read_csv('data/test_set.csv')[["id"]].copy()

column="word_seg"
n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
#vec = TfidfVectorizer(min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(train[column])
test_term_doc = vec.transform(test[column])

save_sparse_csr('tfidf/train_tfidf_all',trn_term_doc)
save_sparse_csr('tfidf/test_tfidf_all',test_term_doc)

print(trn_term_doc.shape)

y=(train["article_class"]-1).astype(int)
#clf = LogisticRegression(C=4, dual=True)
#clf = MultinomialNB(alpha = 0.01)
clf = GaussianNB()

clf.fit(trn_term_doc, y)
preds=clf.predict_proba(test_term_doc)

#保存概率文件
test_prob=pd.DataFrame(preds)
test_prob.columns=["class_prob_%s"%i for i in range(1,preds.shape[1]+1)]
test_prob["id"]=list(test_id["id"])
test_prob.to_csv('result/prob_normal_tfidf_gnb.csv',index=None)

#生成提交结果
preds=np.argmax(preds,axis=1)
test_pred=pd.DataFrame(preds)
test_pred.columns=["class"]
test_pred["class"]=(test_pred["class"]+1).astype(int)
print(test_pred.shape)
print(test_id.shape)
test_pred["id"]=list(test_id["id"])
test_pred[["id","class"]].to_csv('result/sub_normal_tfidf_gnb.csv',index=None)
t2=time.time()
print("time use:",t2-t1)