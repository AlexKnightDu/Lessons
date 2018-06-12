#import pudb
#pu.db
from sklearn.naive_bayes import GaussianNB


from sklearn.neural_network import MLPClassifier as mlpc
from sklearn import datasets as ds
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import multiprocessing as mp
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

def train(k,fout):
    x,y,xt,yt = load_data()

    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(x.toarray(), y)
    predictions = clf.predict(xt.toarray())
    print(classification_report(yt, predictions))

    fout.write(str(k) + '\n' + str(classification_report(yt, predictions)))
    fout.write('\n')
    fout.flush()



def main():
    fout = open('./' + 'knn' + '_d' + '.out', 'w+')
    pool = mp.Pool()
    processes = []
    result = []

    k_num = 14

    for i in range(2,k_num):
        processes += [pool.apply_async(train, args=(i,fout))]

    pool.close()
    pool.join()
    fout.close()

main()
