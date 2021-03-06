
# Data Grand Cup Competition

### 長文本數據和分類信息
經過脫敏處理，分別提供字級別(article)和詞級別(word_seg)的數據。
建立模型通过长文本数据正文(article)，预测文本对应的类别(class)。

![](https://i.imgur.com/zinSujQ.png)



### Data Information
總共十萬筆左右training data。
字平均長度1348，詞平均長度717。
以長度來看可能是中文。
![](https://i.imgur.com/8CSC1eU.png)
![](https://i.imgur.com/iHRbtwE.png)

### Visualize
* Length Frequency
training data和test data的長度分布幾乎是一樣的。
![](https://i.imgur.com/PZDjwFl.png)
* Class Frequency
分類分布有一些bias，嘗試將training data每個分類補到一樣的數量，但沒什麼效果，可能因爲test data本身和training data 分布差不多。
![](https://i.imgur.com/mZtUtjm.png)

### Pre-processing
* Reverse: 一般句子中越靠後的詞重要程度越高，對句子進行逆序輸入。
* Enhance: 樣本數較小的數據增強，打亂句子順序來構建新樣本。
* Sort_by_len: 對句子按照長短排序。
* TF-IDF: 挑選tfidf較高的10000個詞作爲vocabulary。
* 长度超过1000的sequence分段。



### RNN model
* two layer GRU
* accuary: 0.71
* time: 3h
![](https://i.imgur.com/8plPYzO.png)

### CNN model
* Cov+Pooling+FC
* accuracy: 0.72
* time: 2min

![](https://i.imgur.com/IGjtWSN.png)

### TextCNN filter size
window size 1,2,4,7 are chosen.
![](https://i.imgur.com/0NTXnfO.png)


### Classifier Result

#### DL

| model | train accuracy | validation accuracy |
| -------- | -------- | -------- |
| textcnn     | 0.8013    | 0.7306     |
| textrnn     | \    | 0.6880203001201153     |
| rnn with attention  | \    | 0.7238089144229889     |
| rcnn  | \    | 0.727667136117816     |


#### ML

SVM在小樣本長文本分類上效果反而比較nn好。

| model | train score | validation score |
| -------- | -------- | -------- |
| **svm**     | 0.9907414797669407    | 0.789242053789731     |
| **LogisticRegression**   | 0.9594839971266661    | 0.7907090464547677     |
| knn     | \    | 0.7070904645476772     |
| naive_bayes     | 0.6469191475776199    | 0.5965770171149144     |
| adaboost  | 0.48001636204006704    | 0.47334963325183377     |
| gradient boost  | 0.4600925852023306    | 0.44987775061124696     |
