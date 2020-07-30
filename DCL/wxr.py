#工具包/wxr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# csv文件删除列
'''
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
data = pd.read_csv(r'D:/2020_summer/data/gse62254_linchuangcanshu_new.csv', encoding='ANSI')
# data.drop('', axis=1, inplace=True)
data.drop('geo_accession', axis=1, inplace=True)  # 删列 axis=0删行
data['MLH1.IHC'] = data['MLH1.IHC'].map({'positive': 1, 'Positive': 1, 'negative': 0, 'Negative': 0})
data['sex'] = data['sex'].map({'M': 1, 'F': 0})  # 值替换
res = data.corr()
res.to_csv("D:/2020_summer/data/linchuangcanshu_relation.csv")

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
