import numpy as np
from sklearn import datasets

iris = datasets.load_iris()

sta = np.zeros([4,5])

for i in range(0,4):
    print(iris.feature_names[i])
    datum = iris.data.T[i]
    print("Min:\t" + str(min(datum)))
    print("Max:\t" + str(max(datum)))
    print("Mean:\t" + str(np.mean(datum)))
    print("Median:\t" + str(np.median(datum)))
    print("Std:\t" + str(np.std(datum)))
    for j in range(0,4):
        if (i != j):
            compare = iris.data.T[j]
            print("Correlation with " + iris.feature_names[j] + ":\n" + str(np.corrcoef(datum,compare)))
            sta[i][j] = (np.corrcoef(datum,compare))[0][1]
    print("Correlation with target: \n" + str(np.corrcoef(datum,iris.target)))
    sta[i][4] = (np.corrcoef(datum,iris.target))[0][1]
    print('-----------------------')


for i in range(0,4):
    s = ""
    for j in range(0,5):
        s += (str(round(sta[i][j],3)) + '\t\t\t')
    print(s)

print(iris.DESCR)
print(iris.data)
print(iris.data.shape)
print(iris.feature_names)
print(iris.target)
print(iris.target.shape)
print(iris.target_names)





