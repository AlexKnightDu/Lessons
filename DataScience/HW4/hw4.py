
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
print(iris.data.shape)

ss = StandardScaler()

data = iris.data[:,[0,2]]

# 划分为训练集,测试集
X_train, X_test, y_train, y_test = train_test_split(data, iris.target, test_size = 0.25, random_state = 33)
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# SVM 模型
C = 0.1
linear_svm = svm.LinearSVC(C=C).fit(X_train, y_train)
rbf_svm = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train)
poly_svm = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, y_train)

y_pred = linear_svm.predict(X_train)
print(classification_report(y_train,y_pred, target_names = iris.target_names))
y_pred = rbf_svm.predict(X_train)
print(classification_report(y_train,y_pred, target_names = iris.target_names))
y_pred = poly_svm.predict(X_train)
print(classification_report(y_train,y_pred, target_names = iris.target_names))


# bagging集成
linear_svm_bagging = BaggingClassifier(
    svm.LinearSVC(C=C),
    n_estimators=20,         # 构造20个SVM模型
    bootstrap=True,
    max_samples=1.0,          # Bootstrap样本大小用所有实例
    bootstrap_features=True,
    max_features=0.7,         # Bootstrap特征使用70%,各模型不一样
    random_state=9)
linear_svm_bagging.fit(X_train,y_train)

rbf_svm_bagging = BaggingClassifier(
    svm.SVC(kernel='rbf', gamma=0.7, C=C),
    n_estimators=20,         # 构造20个SVM模型
    bootstrap=True,
    max_samples=1.0,          # Bootstrap样本大小用所有实例
    bootstrap_features=True,
    max_features=0.7,         # Bootstrap特征使用70%,各模型不一样
    random_state=9)
rbf_svm_bagging.fit(X_train,y_train)

poly_svm_bagging = BaggingClassifier(
    svm.SVC(kernel='poly', degree=3, C=C),
    n_estimators=20,         # 构造20个SVM模型
    bootstrap=True,
    max_samples=1.0,          # Bootstrap样本大小用所有实例
    bootstrap_features=True,
    max_features=0.7,         # Bootstrap特征使用70%,各模型不一样
    random_state=9)
poly_svm_bagging.fit(X_train,y_train)



# 在测试集上预测
y_pred = linear_svm.predict(X_test)
print(classification_report(y_test,y_pred, target_names = iris.target_names))

y_pred = rbf_svm.predict(X_test)
print(classification_report(y_test,y_pred, target_names = iris.target_names))

y_pred = poly_svm.predict(X_test)
print(classification_report(y_test,y_pred, target_names = iris.target_names))


y_pred = linear_svm_bagging.predict(X_test)
print(classification_report(y_test,y_pred, target_names = iris.target_names))

y_pred = rbf_svm_bagging.predict(X_test)
print(classification_report(y_test,y_pred, target_names = iris.target_names))

y_pred = poly_svm_bagging.predict(X_test)
print(classification_report(y_test,y_pred, target_names = iris.target_names))

#
# 创建网格，以绘制图像
h = .02  # 网格中的步长
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 图的标题
titles = ['SVM with linear kernel',
          'SVM with RBF kernel',
          'SVM with polynomial (degree 3) kernel',
          'Bagging linear SVM',
          'Bagging RBF SVM',
          'Bagging polynomial SVM']


for i, clf in enumerate((linear_svm, rbf_svm, poly_svm, linear_svm_bagging, rbf_svm_bagging, poly_svm_bagging)):
    # 绘出决策边界，不同的区域分配不同的颜色
    plt.subplot(2, 3, i + 1) # 创建一个2行2列的图，并以第i个图为当前图
    plt.subplots_adjust(wspace=0.4, hspace=0.4) # 设置子图间隔

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) #将xx和yy中的元素组成一对对坐标，作为支持向量机的输入，返回一个array

    # 把分类结果绘制出来
    Z = Z.reshape(xx.shape) #(220, 280)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8) #使用等高线的函数将不同的区域绘制出来

    # 将训练数据以离散点的形式绘制出来
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()
