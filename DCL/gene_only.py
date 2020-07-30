#基因+预测位置liver
import numpy as np
import pandas as pd
from sklearn import datasets
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
#读入文件
data = pd.read_csv(r'liver.csv')
data['liver'] = data['liver'].astype(str)
data.info()
#进行训练集测试集数据分离
array = data.values
X= array[:, 2:17]
Y = array[:, 1]
testsize = 0.3
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=testsize, random_state=seed)

clf = DecisionTreeClassifier(random_state=0)
rfc = RandomForestClassifier(random_state=0)
clf = clf.fit(X_train,Y_train)
rfc = rfc.fit(X_train,Y_train)
score_c = clf.score(X_test,Y_test)
score_r = rfc.score(X_test,Y_test)
print("Single Tree:{}".format(score_c),"Random Forest:{}".format(score_r))

#载入模型
models = {}
models['KNN'] = KNeighborsClassifier()
#进行k折交叉验证
num_folds = 9
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed,shuffle=True)
# 评估算法
results = []
for name in models:
    result = cross_val_score(models[name], X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(result)
    msg = '%s: %.3f (%.3f)' % (name, result.mean(), result.std())
    print(msg)
#输出结果的线箱图，进行均值和方差评估
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(models.keys())
pyplot.show()