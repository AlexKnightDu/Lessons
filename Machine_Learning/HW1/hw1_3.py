import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture


def main():
    cluster_nums = [2,4,8,12]
    samples_nums = [100,500,900]
    dimensions = [2,3,4]
    k_range = list(range(1,16))

    result_aic = []
    result_bic = []
    result_vbem = []
    cond_cluster = []
    cond_dimension = []
    cond_sample = []

    np.random.seed(0)
    cluster_data = []
    for dimension in dimensions:
        for samples_num in samples_nums:
            for cluster_num in cluster_nums:
                data = []
                for i in range(cluster_num):
                    cluster_data += [np.random.randn(np.random.randint(100,100+samples_num), dimension) + np.random.randint(10,50,size=(1,dimension))]
                    data.extend(cluster_data[i])
                data = np.array(data)

                lowest_aic = np.infty
                best_aic_k = 0
                aic = []

                lowest_bic = np.infty
                best_bic_k = 0
                bic = []

                lowest_bound = np.infty
                best_vbem_k = 0
                vbem = []

                for k in k_range:
                    gmm = GaussianMixture(n_components=k, covariance_type='full')
                    gmm.fit(data)
                    aic.append(gmm.aic(data))
                    bic.append(gmm.bic(data))
                    if aic[-1] < lowest_aic:
                        lowest_aic = aic[-1]
                        best_aic_k = k
                    if bic[-1] < lowest_bic:
                        lowest_bic = bic[-1]
                        best_bic_k = k


                    vbgmm = BayesianGaussianMixture( weight_concentration_prior_type="dirichlet_distribution", n_components=k,covariance_type='full')
                    vbgmm.fit(data)
                    vbem.append(vbgmm.lower_bound_ * (-1))
                    if vbem[-1] < lowest_bound:
                        lowest_bound = vbem[-1]
                        best_vbem_k = k

                cond_cluster += [cluster_num]
                cond_dimension += [dimension]
                cond_sample += ['<' + str(samples_num)]
                result_aic += [best_aic_k]
                result_bic += [best_bic_k]
                result_vbem += [best_vbem_k]

                print(str(cluster_num) + '\t\t' + str(dimension) + '\t\t<' + str(100 + samples_num) + '\t\t' + str(best_aic_k) + '\t\t' + str(best_bic_k) + '\t\t' + str(best_vbem_k))
                plt.figure(str(cluster_num) + '-cluster ' + str(dimension) + 'D ' + 'with random <' + str(100 + samples_num) + ' samples')
                plt.title(str(cluster_num) + ' clusters ' + str(dimension) + 'D ' + ' <' + str(100 + samples_num) + ' samples')
                plt.plot(k_range, aic, 'b-o', label='AIC Optimal k = ' + str(best_aic_k))
                plt.plot(k_range, bic, 'r-o', label='BIC Optimal k = ' + str(best_bic_k))
                plt.plot(k_range, vbem, 'g-o', label='VBEM\'s (minus) lower bound Optimal k = ' + str(best_vbem_k))
                plt.legend(loc='upper right')
                plt.savefig('./'+ str(cluster_num) + 'cluster_' + str(dimension) + 'D_' + 'with_random_' + str(100 + samples_num) + '_samples')
                # plt.show()
    cond_cluster = np.array(cond_cluster)
    result_aic = np.array(result_aic)
    result_bic = np.array(result_bic)
    result_vbem = np.array(result_vbem)
    error_aic = result_aic - cond_cluster
    error_bic = result_bic - cond_cluster
    error_vbem = result_vbem - cond_cluster
    error_ratio_aic = sum(list(map(lambda x: 1 if x != 0 else 0,(result_aic - cond_cluster)))) * 1.0 / len(cond_cluster)
    error_ratio_bic = sum(list(map(lambda x: 1 if x != 0 else 0,(result_bic - cond_cluster)))) * 1.0 / len(cond_cluster)
    error_ratio_vbem = sum(list(map(lambda x: 1 if x != 0 else 0,(result_vbem - cond_cluster)))) * 1.0 / len(cond_cluster)

    print(cond_cluster)
    print(cond_dimension)
    print(cond_sample)
    print(result_aic)
    print(result_bic)
    print(result_vbem)
    print(error_aic)
    print(error_bic)
    print(error_vbem)
    print(error_ratio_aic)
    print(error_ratio_bic)
    print(error_ratio_vbem)



main()
