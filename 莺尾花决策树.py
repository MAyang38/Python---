import pandas as pd
from sklearn import tree
import matplotlib as mpl
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

#加载数据集
data = pd.read_csv('./iris_data.csv')
data.head()
X = data.drop(['target','label'],axis=1)
y = data.loc[:,'label']
print(X.shape,y.shape)

#形成决策树
dc_tree = tree.DecisionTreeClassifier(criterion='entropy',min_samples_leaf=5)
dc_tree.fit(X,y)
y_predict = dc_tree.predict(X)

accuracy = accuracy_score(y,y_predict)
print(accuracy)

font2 = {'family' : 'SimHei',
'weight' : 'normal',
'size'   : 20,
}
mpl.rcParams['font.family'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

#决策树可视化
fig = plt.figure(figsize=(20,20))
tree.plot_tree(dc_tree,filled='True',
               feature_names=['花萼长', '花萼宽', '花瓣长', '花瓣宽'],
               class_names=['山鸢尾', '变色鸢尾', '维吉尼亚鸢尾'])
#可视化保存为图片
plt.savefig('./1.png', bbox_inches='tight', pad_inches=0.0)
