import tensorflow as tf
import numpy as np
import scipy.io as scio

def main():
    data_file = './data.mat'
    data = scio.loadmat(train_data_file)

    train_data = data['train_de']
    train_label = data['train_label_eeg']
    test_data = data['test_de']
    test_label = data['test_label_eeg']

    


main()







