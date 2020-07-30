#工具包/wxr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''
data = pd.read_csv(r'D:/2020_summer/data/20200712-DCL/DCL/gse62254_linchuangcanshu.csv')
data['DFS'] = pd.qcut(data['OS'], q=4, labels=[1, 2, 3, 4])
data['OS'] = pd.qcut(data['OS'], q=4, labels=[1, 2, 3, 4])
data.to_csv(r'D:/2020_summer/data/20200712-DCL/DCL/gse62254_linchuangcanshu_wxr.csv', index=False)
'''

'''
# csv文件删除列
data = pd.read_csv(r'D:/2020_summer/gse62254_biaoda_test.csv')
for i in range(2, 16762):
    data.drop(data.columns[i], axis=1, inplace=True)
data.to_csv('D:/2020_summer/gse62254_biaoda_2.csv')
'''

'''
# csv文件转置
file = 'D:/2020_summer/gse62254_linchuangcanshu_new.csv'
df = pd.read_csv(file, 'gbk', header=None)
df.values
# data = df.as_matrix()
data = df.iloc[:, :].values
data = list(map(list, zip(*data)))
data = pd.DataFrame(data)
data.to_csv('D:/2020_summer/gse62254_linchuangcanshu_paixu_zhuanzhi.csv', header=0, index=0)
'''

'''# 没啥用
data1 = pd.read_csv(r'D:/2020_summer/gse62254_biaoda(1).csv', usecols=[0]) # 读取第一列
data2 = pd.read_csv(r'D:/2020_summer/gse62254_biaoda.csv', 'gbk')
data3 = pd.merge(data1, data2) #合并csv文件，没调好
print(data3)
'''


# 相关度rank
data = pd.read_csv(r'D:/2020_summer/data/20200712-DCL/DCL/gse62254_linchuangcanshu_wxr.csv')
# data.drop('', axis=1, inplace=True)
data.drop('GEO', axis=1, inplace=True)  # 删列 axis=0删行
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
# data['sex'] = data['sex'].map({'M': 1, 'F': 0})  # 值替换
res = data.corr().abs()
# f = plt.figure(figsize=(25, 25))
# cmap = sns.cm.rocket_r
# sns.heatmap(res, annot=True, cmap=cmap)     # annot = True 显示每个方格的数据
# f.savefig('D:/2020_summer/data/20200712-DCL/DCL/abs.png')
res.to_csv("D:/2020_summer/data/20200712-DCL/DCL/gse62254_linchuangcanshu_relation_abs.csv")


'''
# 热力图
data.drop('id', axis=1, inplace=True)
data.drop('X', axis=1, inplace=True)
featurs_mean = list(data.columns[1:30])
corr = data[featurs_mean].corr()
# f = plt.figure(figsize=(20, 20))
# sns.heatmap(corr, annot=True)     # annot = True 显示每个方格的数据
#f.savefig('D:/2020_summer/2.png')
print(corr)
'''
