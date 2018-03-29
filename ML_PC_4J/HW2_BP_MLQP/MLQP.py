#coding:utf-8
import numpy as np
import time
import matplotlib.pyplot as plt


def sigmod(z):
    return 1.0 / (1.0 + np.exp(-z))

class mlqp(object):
    def __init__(self, learning_rate=0.1, momentum=0.5, threshold_error =1e-5, max_epoch=100, structure=None):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.threshold_error = threshold_error
        self.max_epoch = max_epoch
        self.structure = structure

        self.u = []
        self.v = []
        self.b = []
        self.last_u = []
        self.last_v = []
        self.last_b = []
        self.init()

    def init(self):
        for i in xrange(len(self.structure) - 1):  # 初始化权重和偏执
            self.u.append(np.mat(np.random.uniform(-0.5, 0.5, size=(self.structure[i + 1], self.structure[i]))))
            self.v.append(np.mat(np.random.uniform(-0.5, 0.5, size=(self.structure[i + 1], self.structure[i]))))
            self.b.append(np.mat(np.random.uniform(-0.5, 0.5, size=(self.structure[i + 1], 1))))

    def forwardPropagation(self, mat_x=None):
        x = [mat_x]
        for i in xrange(len(self.u)):
            x.append(sigmod(self.u[i] * np.multiply(x[-1], x[-1]) + self.v[i] * x[-1] + self.b[i]))
        return x

    def backPropagation(self, label=None, x=None):
        delta = []
        e_j = x[-1] - label
        # f'(x) = f(x)(1 - f(x))
        # 输出层 delta
        delta.append(np.multiply(e_j, np.multiply(x[-1], (1.0 - x[-1]))))

        # 中间层 delta
        # 从后往前计算
        for i in xrange(len(self.u) - 1):
            f_yj = np.multiply(x[-2 - i], 1 - x[-2 - i])
            # print((np.multiply(self.u[-1-i] * 2, x[-2 - i])))
            # print((np.multiply(self.u[-1-i] * 2, x[-2 - i].T)))
            middle_layer_delta = np.multiply((np.multiply(self.u[-1-i] * 2, x[-2 - i].T) + self.v[-1 - i]).T * delta[-1], f_yj)
            delta.append(middle_layer_delta)

        if not len(self.last_u):
            # 第一次反向传播
            for i in xrange(len(self.structure) - 1):
                self.last_u.append(np.mat(np.zeros_like(self.u[i])))
                self.last_v.append(np.mat(np.zeros_like(self.v[i])))
                self.last_b.append(np.mat(np.zeros_like(self.b[i])))

            for j in xrange(len(delta)):
                self.last_u[-1 - j] = -self.learning_rate * (delta[j] * (np.multiply(x[-2 - j], x[-2 - j])).T)
                self.last_v[-1 - j] = -self.learning_rate * (delta[j] * x[-2 - j].T)
                self.last_b[-1 - j] = -self.learning_rate * delta[j]
                self.u[-1 - j] = self.u[-1 - j] + self.last_u[-1 - j]
                self.v[-1 - j] = self.v[-1 - j] + self.last_v[-1 - j]
                self.b[-1 - j] = self.b[-1 - j] + self.last_b[-1 - j]
        else:
            for j in xrange(len(delta)):
                self.last_u[-1 - j] = -self.learning_rate * (delta[j] * np.multiply(x[-2 - j], x[-2 - j]).T + self.momentum * self.last_u[-1 - j])
                self.last_v[-1 - j] = -self.learning_rate * (delta[j] * x[-2 - j].T + self.momentum * self.last_v[-1 - j])
                self.last_b[-1 - j] = -self.learning_rate * (delta[j] + self.momentum * self.last_b[-1 - j])
                self.u[-1 - j] = self.u[-1 - j] + self.last_u[-1 - j]
                self.v[-1 - j] = self.v[-1 - j] + self.last_v[-1 - j]
                self.b[-1 - j] = self.b[-1 - j] + self.last_b[-1 - j]
        error = sum(0.5 * np.multiply(x[-1] - label, x[-1] - label))
        return error

    def train(self, input_=None, target=None, show=10, fout=None):
        start = time.clock()
        for ep in xrange(self.max_epoch):
            error = []
            for item in xrange(input_.shape[1]):
                x = self.forwardPropagation(input_[:, item])
                e = self.backPropagation(target[:, item], x)
                error.append(e[0, 0])
            epoch_error = sum(error) / len(error)
            if epoch_error < self.threshold_error:
                elapsed = (time.clock() - start)
                print "Finish {0}: ".format(ep), epoch_error
                print("Time used:", elapsed)
                fout.write(str(epoch_error) + '\n')
                return
            elif ep % show == 0:
                print "epoch {0}: ".format(ep), epoch_error
                fout.write(str(epoch_error) + '\n')

    def sim(self, inp=None):
        return self.forwardPropagation(mat_x=inp)[-1]


def main():
    learning_rate = 0.5
    fout = open('error_' + str(learning_rate) + '.txt', 'w')

    # 网络结构，输入层2个节点，中间层10个节点，输出层1个节点
    network_structure = [2,10,1]

    # 训练数据
    train_data_file = 'two_spiral_train.txt'
    train_data = np.loadtxt(train_data_file)
    train_data /= train_data.max(axis=0)  # 正则化
    train_samples = []
    train_labels = []
    for i in range(len(train_data)):
        train_samples.append(train_data[i][:-network_structure[-1]])
        train_labels.append(train_data[i][-network_structure[-1]:])
    train_samples = np.mat(train_samples).transpose()
    train_labels = np.mat(train_labels).transpose()

    # 测试数据
    test_data_file = 'two_spiral_test.txt'
    test_data = np.loadtxt(test_data_file)
    test_data /= test_data.max(axis=0)
    test_samples = []
    test_labels = []
    for i in range(len(test_data)):
        test_samples.append(test_data[i][:-network_structure[-1]])
        test_labels.append(test_data[i][-network_structure[-1]:])
    test_samples = np.mat(test_samples).transpose()
    test_labels = np.mat(test_labels).transpose()


    # 这里我们通过设置训练误差上限和最大迭代次数来判断训练过程是否应该终止
    # 通过上面的network_structure来确定网络结构
    # 这里按照slide内容添加了momentum机制来加速训练过程
    model = mlqp(learning_rate=learning_rate, momentum=0.5, threshold_error=5e-5, max_epoch=10000, structure=network_structure)

    # 训练网络
    model.train(input_=train_samples, target=train_labels, show=10, fout=fout)

    # 训练数据的预测
    sims = []
    [sims.append(model.sim(train_samples[:, idx])) for idx in xrange(train_samples.shape[1])]
    print "training error: ", sum(np.array(sum(0.5 * np.multiply(train_labels - np.mat(np.array(sims).transpose()),
                                                                 train_labels - np.mat(np.array(sims).transpose())) /
                                               train_labels.shape[1]).tolist()[0]))

    sims_test = []
    [sims_test.append(model.sim(test_samples[:, idx])) for idx in xrange(test_samples.shape[1])]
    print "test error: ", sum(np.array(sum(0.5 * np.multiply(test_labels - np.mat(np.array(sims_test).transpose()),
                                                             test_labels - np.mat(np.array(sims_test).transpose())) /
                                           test_labels.shape[1]).tolist()[0]))

    print("Finished")
    fout.close()

    # 绘制决策面
    p_x = np.array(range(-100,100,1))
    p_x = p_x / 100.0
    p_y = p_x
    px = []
    py = []
    colors = []

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Learning rate = ' + str(learning_rate))
    plt.xlabel('X')
    plt.ylabel('Y')

    for x in p_x:
        for y in p_y:
            if (model.sim(np.mat([x,y]).transpose())) > 0.5:
                px.append(x)
                py.append(y)
                colors.append('b')
            else:
                px.append(x)
                py.append(y)
                colors.append('r')
    ax1.scatter(px, py, c=colors, s=1, marker=',')
    plt.show()

main()







