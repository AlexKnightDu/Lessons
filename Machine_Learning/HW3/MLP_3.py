#import pudb
#pu.db


from sklearn.neural_network import MLPClassifier as mlpc
from sklearn import datasets as ds
from sklearn.metrics import classification_report,confusion_matrix

import time
import scipy.sparse as sps

train_data_file = './dna.scale.txt'
test_data_file = './dna.scale.t'



def load_data():
    x,y=ds.load_svmlight_file(train_data_file)
    xt,yt=ds.load_svmlight_file(test_data_file)
    x.todense()
    xt.todense()
    x = sps.csr_matrix((x.data, x.indices, x.indptr), shape=(x.get_shape()[0], xt.get_shape()[1]))
    return x,y,xt,yt

def main():

    time_stamp = time.strftime("%H-%M-%S",time.localtime())
    fout = open('./MLP_d/s' + 'MLP' + '_' + time_stamp + '.out', 'w+')

    networks = [(100,20), (200,50),
                (100,200,30), (100,300,40)]
    activations = ['logistic', 'tanh', 'relu']
    solvers = ['lbfgs','sgd', 'adam']
    learning_rates = [0.001, 0.01]
    learning_rate_setings = ['constant',  'adaptive', 'invscaling']
    for lr in learning_rates:
        for lrs in learning_rate_setings:
            for activation in activations:
                for solver in solvers:
                    for network in networks:
                        clf = mlpc(activation=activation, batch_size='auto',
                           beta_1=0.9, beta_2=0.999, early_stopping=False,
                           epsilon=1e-08, hidden_layer_sizes=network, learning_rate=lrs,
                           learning_rate_init=lr, max_iter=200, momentum=0.9,
                           nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                           solver=solver, tol=0.0001, validation_fraction=0.1, verbose=False,
                           warm_start=False)

                        x,y,xt,yt = load_data()
                        clf.fit(x,y)
                        predictions = clf.predict(xt)
                        param = str(lr) + ' ' + lrs + ' ' + str(network) + ' ' + activation + ' ' + solver
                        print(param)
                        print(classification_report(yt, predictions))
                        fout.write(param + '\n')
                        fout.write(str(classification_report(yt, predictions)))
                        fout.write('\n')
                        fout.flush()

main()