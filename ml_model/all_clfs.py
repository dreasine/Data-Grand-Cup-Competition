import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import time

from sklearn import tree, svm, naive_bayes,neighbors
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from scipy.sparse import csr_matrix


t1=time.time()


train = pd.read_csv('data/train_set.csv')
train = train.rename(columns={'class': 'article_class'}, inplace=False)
'''
test = pd.read_csv('data/test_set.csv')
test_id = pd.read_csv('data/test_set.csv')[["id"]].copy()

'''

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

column="word_seg"
n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
#vec = TfidfVectorizer(min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)

#trn_term_doc = vec.fit_transform(train[column])
#test_term_doc = vec.transform(test[column])

trn_term_doc = load_sparse_csr('tfidf/train_tfidf_all.npz')
test_term_doc = load_sparse_csr('tfidf/test_tfidf_all.npz')
print(trn_term_doc.shape)

y=(train["article_class"]-1).astype(int)
#clf = LogisticRegression(C=4, dual=True)

'''
clf = RandomForestClassifier(oob_score=True, random_state=10)
clf = MultinomialNB(alpha = 0.01)
clf.fit(trn_term_doc, y)
preds=clf.predict_proba(test_term_doc)
'''

clfs = {#'svm': svm.LinearSVC(),\
        #'lr': LogisticRegression(C=4, dual=True),\
        #'decision_tree':tree.DecisionTreeClassifier(),
        #'naive_gaussian': naive_bayes.GaussianNB(), \
        #'naive_mul':naive_bayes.MultinomialNB(),\
        #'K_neighbor' : neighbors.KNeighborsClassifier(),\
        #'bagging_knn' : BaggingClassifier(neighbors.KNeighborsClassifier(), max_samples=0.5,max_features=0.5), \
        #'bagging_tree': BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.5,max_features=0.5),
        #'random_forest' : RandomForestClassifier(n_estimators=50),\
        #'adaboost':AdaBoostClassifier(n_estimators=50),\
        'gradient_boost' : GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,max_depth=1, random_state=0)
        }

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = trn_term_doc[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(0.02 * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]



def try_different_method(clf):
    clf.fit(x_train,y_train)
    print('fit finish')
    score = clf.score(x_dev,y_dev)
    print('the score is :', score)
    tscore = clf.score(x_train,y_train)
    print('training score is :',tscore)

for clf_key in clfs.keys():
    print('the classifier is :',clf_key)
    clf = clfs[clf_key]
    try_different_method(clf)

'''
#保存概率文件
test_prob=pd.DataFrame(preds)
test_prob.columns=["class_prob_%s"%i for i in range(1,preds.shape[1]+1)]
test_prob["id"]=list(test_id["id"])
test_prob.to_csv('result/prob/prob_rf_baseline.csv',index=None)

#生成提交结果
preds=np.argmax(preds,axis=1)
test_pred=pd.DataFrame(preds)
test_pred.columns=["class"]
test_pred["class"]=(test_pred["class"]+1).astype(int)
print(test_pred.shape)
print(test_id.shape)
test_pred["id"]=list(test_id["id"])
test_pred[["id","class"]].to_csv('result/rf_baseline.csv',index=None)
t2=time.time()
print("time use:",t2-t1)
'''