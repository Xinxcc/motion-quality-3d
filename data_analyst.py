import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#导入数据集'titanic'，命名为'titanic'
titanic=sns.load_dataset('titanic')

# ### distplot

# #查看数据集的随机10行数据，用sample方法
# titanic.sample(10)

# #去除'age'中的缺失值，distplot不能处理缺失数据
# age1=titanic['age'].dropna()

# # 通过'bins'参数设定数据片段的数量,去掉拟合的密度估计曲线，kde参数设为False
# #可以分别控制直方图、密度图的关键参数
# #rug: 边际毛毯
# #histfalse: 无直方图
# fig,axes=plt.subplots(1,2) 
# sns.distplot(age1,rug=True,bins=30,kde=False,ax=axes[0])
# sns.distplot(age1,rug=True,bins=30,kde=True,
#                      hist_kws={'color':'green','label':'hist'},
#                      kde_kws={'color':'red','label':'KDE'},
#                      ax=axes[1])
# plt.show()


### barplot

#将'class'设为x轴，'survived'为y轴，传入'titanic'数据
sns.barplot(x='class',y='survived',data=titanic,errorbar=('ci', 95))
plt.show()