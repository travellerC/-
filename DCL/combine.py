#临床参数+基因+预测位置liver
import pandas as pd
import numpy as np
from pandas import read_csv
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os
import csv

csv_1=pd.read_csv(r'gse62254_linchuangcanshu2.csv')
csv_2=pd.read_csv(r'liver.csv')
#删除重复列
csv_2.drop('GEO',axis = 1,inplace=True)
csv_2.drop('liver',axis = 1,inplace=True)
#进行合并
data=pd.concat([csv_1,csv_2],axis=1)
data.to_csv(r'combine_liver.csv',index=False)

data.drop('id',axis = 1,inplace=True)
#进行列名简化
colNameDict=({'geo_accession':'GEO',
              'MLH1.IHC':'MLH',
              'WHO.1.w.d.adeno.2.m.d.adeno.3.p.d.adeno.4.signet.ring.5..mucinous.6.papillary.adeno.7.adenosquamous.8.undifferentiated.ca.9.hepatoid.adenoca.10.tubular.adenoca.11.others..text.':'WHO',
              'Perineural.Invasion':'PI',
              'VENOUS.INVASION.':'VI',
              'lymphatic..lymphovascular.inv.':'LI',
              'documented.recurrence.':'recurr',
              'peritoneal.seeding':'fm',
              'intraabdominal_LN':'lbj',
              'distant.lymph.node':'lbj2',
              'FU.status0.无复发存活1.复发但存活..2.未复发死亡.3.复发死亡..4.死因未明的死亡..5..删失':'FU',
              'H..pylori.0.No.1.Yes.blank...H.pylori.not.checked':'ym',
              'DFS..months.':'DFS',
              'OS..months.':'OS',
              'Mol..Subtype..0.MSS.TP53...1.MSS.TP53...2...MSI..3..EMT':'Mol'})
data.rename(columns=colNameDict,inplace=True)
#删除NA数量大于40%的列
data.drop('VI',axis = 1,inplace=True)
data.drop('ym',axis = 1,inplace=True)
#删除与预测结果无关的列
data.drop('description',axis = 1,inplace=True)

data['MLH'] = data['MLH'].map({'negative':1,'positive':0,'Negative':1})
data['sex'] = data['sex'].map({'M':1,'F':0})
data['Lauren'] = data['Lauren'].map({'intestinal':0,'diffuse':1,'mixed':2,'indeterminate':3})

#删除复发结果未知的样本
data = data[~data['recurr'].isin([2])]

#用众数填补EBV,PI,LI的NA
most_common1=data['EBV'].value_counts().index[0]
data['EBV'].fillna(most_common1,inplace=True)
most_common2=data['PI'].value_counts().index[0]
data['PI'].fillna(most_common2,inplace=True)
most_common3=data['LI'].value_counts().index[0]
data['LI'].fillna(most_common3,inplace=True)

#特征工程
#连续数据离散化:按照分位数对样本进行划分的，这样划分的结果是的每个区间的大小基本相同
data['age']=pd.qcut(data['age'],q=4,labels=[1,2,3,4])
#data['DFS'] = pd.qcut(data['OS'], q=4, labels=[1, 2, 3, 4])
#data['OS'] = pd.qcut(data['OS'], q=4, labels=[1, 2, 3, 4])
#改正数据的瑕疵
data['MLH']= data['MLH'].map({'positive': 1, 'Positive': 1,'positive; MSH2 mutation (+)': 1, 'negative': 0, 'Negative': 0,'partial loss':0})
#异常数据处理
data['WHO'] = data['WHO'].map({'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                               '8': 8, '9': 9, '10': 10, '11': 11,
                               '①2/ ②1/ ③1': 3, '①2/ ②1': 3, '2,3': 3, '2,6(20%)': 3,
                               '3, 11(neuroendocrine differentiation)': 3, '6,2': 3,
                               '11(lymphoepithelioma-like carcinoma)': 11,
                               '11 Adenocarcinoma with neuroendocrine differentiation': 11,
                               '11(composite adenoca and neuroendocrine ca)': 11})
data['T'] = data['T'].map({'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                               '8': 8, '9': 9, '10': 10, '11': 11,
                               '①2/ ②1/ ③1': 2, '①2/ ②1': 2, '2,3': 3, '2,6(20%)': 3,
                               '3, 11(neuroendocrine differentiation)': 3, '6,2': 3,
                               '11(lymphoepithelioma-like carcinoma)': 11,
                               '11 Adenocarcinoma with neuroendocrine differentiation': 11,
                               '11(composite adenoca and neuroendocrine ca)': 11})

#标签和特征变量转换为字符串
data['EBV'] = data['EBV'].astype(str)
data['WHO'] = data['WHO'].astype(str)
data['PI'] = data['PI'].astype(str)
data['LI'] = data['LI'].astype(str)
data['T'] = data['T'].astype(str)
data['N'] = data['N'].astype(str)
data['M'] = data['M'].astype(str)
data['MLH'] = data['MLH'].astype(str)
data['liver'] = data['liver'].astype(str)
data['fm'] = data['fm'].astype(str)
data['ascites'] = data['ascites'].astype(str)
data['lbj'] = data['lbj'].astype(str)
data['lbj2'] = data['lbj2'].astype(str)
data['bone'] = data['bone'].astype(str)
data['other'] = data['other'].astype(str)

#使用了LabelEncoder可以将英文或特殊字符的categorical labels 转换为不同的数字
data['sex'] = data['sex'].astype(str)
data['sex'] =LabelEncoder().fit_transform(data['sex'])
data['pStage'] = data['pStage'].astype(str)
data['pStage'] =LabelEncoder().fit_transform(data['pStage'])
data['Lauren'] = data['Lauren'].astype(str)
data['Lauren'] =LabelEncoder().fit_transform(data['Lauren'])
data['Code_site'] = data['Code_site'].astype(str)
data['Code_site'] =LabelEncoder().fit_transform(data['Code_site'])


data.to_csv(r'combine_liver2.csv',index=False)
#统计各转移位置的数据有多少个
print(Counter(data['liver']))
print(Counter(data['fm']))
print(Counter(data['ascites']))
print(Counter(data['lbj']))
print(Counter(data['lbj2']))
print(Counter(data['bone']))
print(Counter(data['other']))

print(data.apply(lambda col:sum(col.isnull())/col.size))
data.info()

#进行训练集测试集数据分离
array = data.values
X1 = array[:, 1:14] # C_D为编号，与Y无相关性，过滤掉
X2 = array[:, 22:41]

#采用Z-Score标准化，保证每个特征维度的数据均值为0，方差为1
ss = StandardScaler()               
X2 = ss.fit_transform(X2)

X= np.hstack((X1,X2))

Y = array[:, 15]
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
models['LR'] = LogisticRegression()

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
