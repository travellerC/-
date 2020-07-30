import pandas as pd
from pandas import read_csv
import pandas as pd
from sklearn import datasets
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
#读入文件
data = pd.read_csv(r'gse62254_linchuangcanshu2.csv')
#删除与数据分析无关的列
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
data['DFS'] = pd.qcut(data['OS'], q=4, labels=[1, 2, 3, 4])
data.to_csv(r'DFS_lisan.csv',index=False)