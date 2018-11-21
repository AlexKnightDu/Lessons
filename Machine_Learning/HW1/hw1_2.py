import numpy as np
import matplotlib.pyplot as plt


def distance2(p1,p2):
    return np.sum((p1-p2)**2)

def min_distance(p,l):
    min_dist = distance2(p,l[0])
    index = 0
    for i in range(1,len(l)):
        distance = distance2(p,l[i])
        if (distance < min_dist):
            min_dist = distance
            index = i
    return index


def initialization(k, low, high):
    return (np.random.randn(k,2)) * (high - low) / 2 + (low + high) * 1.0 / 2

def E_step(data, centers):
    clusters = []
    k = len(centers)
    for i in range(0,k):
        clusters += [[]]
    for point in data:
        index = min_distance(point, centers)
        clusters[index] += [point]
    return clusters

def M_step(data, centers, clusters):
    new_centers = []
    k = len(clusters)
    for i in range(0,k):
        points = np.array(clusters[i])
        if (len(points) == 0):
            center = centers[i]
        else:
            num = len(points)
            sum_x = np.sum(points[:,0])
            sum_y = np.sum(points[:,1])
            center = np.array([sum_x,sum_y]) * 1.0 / num
        new_centers += [center]
    return np.array(new_centers)

def penalization(old_centers, new_centers, clusters):
    centers = []
    k = len(clusters)
    for i in range(0,k):
        if ((len(clusters[i]) != 0) and (distance2(new_centers[i], old_centers[i]) > 1e-1)):
            pull = (new_centers[i] - old_centers[i]) / distance2(new_centers[i], old_centers[i])
            push = np.array([0.0,0.0])
            for j in range(0,k):
                if (j != i):
                    push += (new_centers[j] - old_centers[i]) / distance2(new_centers[j], old_centers[i])
            move = (pull - push) * distance2(new_centers[i], old_centers[i])
            centers += [old_centers[i] + move]
        else:
            centers += [old_centers[i]]
    return np.array(centers)


def main():
    cluster_num = 3
    samples_num = 200

    samples_points = [np.array([10, 10]), np.array([20, 10]), np.array([10, 20])]

    np.random.seed(0)
    cluster_data = []
    data = []
    for i in range(cluster_num):
        cluster_data += [np.random.randn(samples_num, 2) + samples_points[i]]
        data.extend(cluster_data[i])

    k = cluster_num + 1
    centers = initialization(k,10,20)

    iteration_num = 10

    colors = []
    for i in range(k):
        colors += [[1-i*1.0/k, 0.8, i*1.0/k]]

    Figs = []
    fig_init = plt.figure('K-mean with CL')
    Figs += [fig_init]
    for i in range(cluster_num):
        plt.scatter(cluster_data[i][:, 0], cluster_data[i][:, 1],s=2)
    plt.scatter(centers[:, 0], centers[:, 1], marker='x',s=100)
    plt.title('Initialization')
    plt.xlim(0,50)
    plt.ylim(0,50)
    plt.show()


    for i in range(0, iteration_num):
        clusters = E_step(data, centers)

        # Normal
        centers = M_step(data,centers,clusters)

        # Penalization
        # new_centers = M_step(data,centers,clusters)
        # centers = penalization(centers, new_centers, clusters)

        fig = plt.figure('Iteration round ' + str(i + 1))
        plt.title('Iteration round ' + str(i + 1))
        Figs += [fig]
        for i in range(len(clusters)):
            if (len(clusters[i]) != 0):
                points = np.array(clusters[i])
                plt.scatter(points[:,0], points[:,1],c=colors[i],s=1)
        plt.scatter(centers[:,0],centers[:,1],marker='+',c='r',s=100)

        # plt.scatter(new_centers[:,0],new_centers[:,1],marker='x',c='b', s=100)

        plt.xlim(0,50)
        plt.ylim(0,50)
        plt.show()

main()