#import pudb
#pu.db


from sklearn.neural_network import MLPClassifier as mlpc
from sklearn import datasets as ds
from sklearn.metrics import classification_report,confusion_matrix

import time


train_data_file = './a9a.txt'
test_data_file = './a9a.txt'



def load_data():
    x,y=ds.load_svmlight_file(train_data_file)
    xt,yt=ds.load_svmlight_file(test_data_file)
    x.todense()
    xt.todense()
    return x,y,xt,yt

def main():

    time_stamp = time.strftime("%H-%M-%S",time.localtime())
    fout = open('./result/' + 'MLP' + '_' + time_stamp + '.out', 'w+')

    networks = [(100,20), (200,50), (200,20),
                (200,300,100), (200,400,100),
                (400,100,40), (400,200,100),
                (400,200,100,30),(200,400,100,30)]
    for network in networks:
        clf = mlpc(activation='relu', alpha=1e-05, batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping=False,
           epsilon=1e-08, hidden_layer_sizes=network, learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
           solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
           warm_start=False)

        x,y,xt,yt = load_data()
        clf.fit(x,y)
        predictions = clf.predict(xt)
        # print(len(xt),len(yt))
        # score = clf.score(xt, yt)
        print(classification_report(yt, predictions))
        fout.write(str(network) + '\n')
        fout.write(str(classification_report(yt, predictions)))
        fout.write('\n')
        fout.flush()
        # print(score)

main()
