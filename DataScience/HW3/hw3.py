#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler as ss



def get_data():
    file_name = './Boston数据集.txt'
    data_file = open(file_name, 'r')
    data = data_file.readlines()
    data = data[22:]
    outfile_name = './data.txt'
    out_file = open(outfile_name, 'w')
    for datum in data:
        out_file.write(datum)
    data_file.close()
    out_file.close()
    data = np.loadtxt(outfile_name)
    return data

# centralize the data
def center(data):
    m,n = data.shape
    mean = np.mean(data,axis=0)
    avgs = np.tile(mean, (m, 1))
    centered_data = data - avgs
    return centered_data

# smooth the exception of the data
def preprocess(data, flag):
    cont_v = [n for n in range(np.shape(data)[1]) if n != 3]
    normalized_data = ss().fit_transform(data[:, cont_v])
    outlier_row, outlier_col = np.where(np.abs(normalized_data) > 3)
    if (flag):
        for i in range(0,len(outlier_col)):
            normalized_data[outlier_row[i]][outlier_col[i]] = np.sign(normalized_data[outlier_row[i]][outlier_col[i]]) * 3
    return normalized_data


def pca(data, ratio):
    cov = np.cov(data, rowvar=0)
    eig_vals, eig_vects = np.linalg.eig(np.mat(cov))

    all = sum(eig_vals)
    choosen = 0
    n = 0
    for i in range(0,len(eig_vals)):
        choosen += eig_vals[i] * 1.0 / all
        n += 1
        if (choosen >= ratio):
            break
    print(choosen,n)

    print(eig_vals)
    principal_vect = eig_vects[:n]
    principal_comp = data * principal_vect.T
    return principal_comp

def main():
    # Get the data
    data = get_data()

    # Zero-mean normalization
    data = center(data)

    # Just scale the data without removing the Exception data
    # preprocess(data, False)
    # Scale data also remove the Exception data
    preprocess(data, True)

    pricipal_data = pca(data, 0.99)
    print(pricipal_data)


def test():
    boston_dataset = datasets.load_boston()
    X_full = boston_dataset.data
    Y = boston_dataset.target

    pca_X_full = pca(center(X_full),0.99)

    # 特征选择
    selector = SelectKBest(f_regression,k=1)
    selector.fit(X_full,Y)
    X = X_full[:,selector.get_support()]

    selector.fit(pca_X_full,Y)
    pca_X = np.array(pca_X_full[:,selector.get_support()])

    fig = plt.figure('数据集')
    ax = fig.add_subplot(121)
    ax.scatter(X,Y,color='black')
    ax.set_title('Without PCA')
    ax = fig.add_subplot(122)
    ax.scatter(pca_X,Y,color='black')
    ax.set_title('With PCA')


    # 线性回归
    regressor = LinearRegression(normalize=True)

    fig1 = plt.figure('线性回归')

    regressor.fit(X,Y)
    ax = fig1.add_subplot(121)
    ax.scatter(X,Y,color='black')
    ax.plot(X,regressor.predict(X),color='red',linewidth=3)
    ax.set_title('Without PCA')

    regressor.fit(pca_X,Y)
    ax = fig1.add_subplot(122)
    ax.scatter(pca_X,Y,color='black')
    ax.plot(pca_X,regressor.predict(pca_X),color='red',linewidth=3)
    ax.set_title('With PCA')


    # SVM
    regressor = SVR()

    fig2 = plt.figure('SVM')

    regressor.fit(X, Y)
    ax = fig2.add_subplot(121)
    ax.scatter(X,Y,color='black')
    ax.scatter(X, regressor.predict(X), color='red', linewidth=3)
    ax.set_title('Without PCA')

    regressor.fit(pca_X, Y)
    ax = fig2.add_subplot(122)
    ax.scatter(pca_X,Y,color='black')
    ax.scatter(pca_X, regressor.predict(pca_X), color='red', linewidth=3)
    ax.set_title('With PCA')

    # Random Forest回归
    regressor = RandomForestRegressor()

    fig3 = plt.figure('Random Forest回归')

    regressor.fit(X, Y)
    ax = fig3.add_subplot(121)
    ax.scatter(X,Y,color='black')
    ax.scatter(X,regressor.predict(X),color='red',linewidth=3)
    ax.set_title('Without PCA')

    regressor.fit(pca_X, Y)
    ax = fig3.add_subplot(122)
    ax.scatter(pca_X,Y,color='black')
    ax.scatter(pca_X,regressor.predict(pca_X),color='red',linewidth=3)
    ax.set_title('With PCA')

    plt.show()

main()
test()



