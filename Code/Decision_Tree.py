from sklearn.feature_extraction import DictVectorizer
# sklearn.feature_extraction模块，可以用于从包含文本和图片的数据集中提取特征
# 类DictVectorizer可以用于将各列使用标准的python dict对象表示的特征数组，转换成sklearn中的estimators可用的NumPy/SciPy表示的对象
import csv
import codecs
from sklearn import preprocessing
#提供了几个常用的实用程序函数和变换器类，用于将原始特征向量更改为更适合下游估计器的表示
from sklearn import tree
from sklearn.externals.six import StringIO

allElectronicsData = open(r'E:\Python\Decision Tree\allElectronics.csv','rb')
Reader = csv.reader(codecs.iterdecode(allElectronicsData, 'utf-8')) #python3
headers = next(Reader)  #python3 用 next(name) 而不是 name.next()
print(headers)

featureList = []  #用来存储特征值的dict
labelList = []  #用来存储类标签值

for row in Reader:
    labelList.append(row[len(row) - 1])
    rowDict = {}
    for i in range(1, len(row) - 1):
       # print(row[i])
        rowDict[headers[i]] = row[i]
       # print("rowDict=", rowDict)
    featureList.append(rowDict)
    # print('rowDict=',rowDict)

print(featureList)

#Vectorize feature
vec = DictVectorizer() # 实例化一个Vectorizer
dummyX = vec.fit_transform(featureList) .toarray() # 转化成1 0 0这种

print("dummyX:" + str(dummyX))
print(vec.get_feature_names())

#Vectorize class labels
lb = preprocessing.LabelBinarizer() # 类标签特有的转换方法
dummyY = lb.fit_transform(labelList)
print("dummyY=" + str(dummyY))

#Using decision tree for classification
clf = tree.DecisionTreeClassifier(criterion='entropy') # 以 ID3算法标准(信息熵) 来生成决策树分类器
clf = clf.fit(dummyX, dummyY) # 建模
print("clf: " + str(clf))

with open("allElectronicInformationGainOri.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names = vec.get_feature_names(), out_file = f)

oneRowX = dummyX[0] # 取第一行
print("oneRow = " + str(oneRowX))

# 创建一行新数据
newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX: " + str(newRowX))

# 进行预测
predictedY = clf.predict([newRowX]) # newRowX要加[] ,python3不同于2
print("predictedY: " + str(predictedY))