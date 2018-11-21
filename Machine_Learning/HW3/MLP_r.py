#import pudb
#pu.db
from sklearn.naive_bayes import GaussianNB


from sklearn.neural_network import MLPClassifier as mlpc
from sklearn import datasets as ds
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import time
import scipy.sparse as sps

train_data_file = './a9a.txt'
test_data_file = './a9a.t'



def load_data():
    x,y=ds.load_svmlight_file(train_data_file)
    xt,yt=ds.load_svmlight_file(test_data_file)
    x.todense()
    xt.todense()
    x = sps.csr_matrix((x.data, x.indices, x.indptr), shape=(x.get_shape()[0], 123))
    xt = sps.csr_matrix((xt.data, xt.indices, xt.indptr), shape=(xt.get_shape()[0], 123))
    return x,y,xt,yt

def main():
    x,y,xt,yt = load_data()
    # fout = open('./' + 'knn' + '_' + '.out', 'w+')
    # parameter_values = list(range(1, 21))
    # 对每个k值的准确率进行计算
    # for k in parameter_values:
    #     # 创建KNN分类器
    #     clf = KNeighborsClassifier(n_neighbors=k)
    #     clf.fit(x.toarray(), y)
    #     predictions = clf.predict(xt.toarray())
    #     print(classification_report(yt, predictions))
    #
    #     fout.write(str(classification_report(yt, predictions)))
    #     fout.write('\n')
    #     fout.flush()
    # fout.close()

        # 创建KNN分类器
    clf = DecisionTreeClassifier(max_depth=8)
    clf.fit(x.toarray(), y)
    predictions = clf.predict(xt.toarray())
    print(classification_report(yt, predictions))

    # fout.write(str(classification_report(yt, predictions)))
    # fout.write('\n')
    # fout.flush()
    # fout.close()



main()
