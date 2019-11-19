"""
Parzen window with Naive Bayes, from the website
https://sebastianraschka.com/Articles/2014_kernel_density_est.html
"""

import operator

import numpy as np
from scipy.stats import kde

# Covariance matrices
cov_mats = {}
for i in range(1, 4):
    cov_mats[i] = i * np.eye(2)
    print(cov_mats[i])

print(' ')

# mean vectors
mu_vecs = {}
for i, j in zip(range(1, 4), [[0, 0], [3, 0], [4, 5]]):
    mu_vecs[i] = np.array(j).reshape(2, 1)
    print(mu_vecs[i])

print(' ')

all_samples = {}
for i in range(1, 4):
    # generating 40x2 dimensional arrays with random Gaussian-distributed samples
    class_samples = np.random.multivariate_normal(mu_vecs[i].ravel(), cov_mats[i], 40)
    # adding class label to 3rd column
    class_samples = np.append(class_samples, np.zeros((40, 1)) + i, axis=1)
    all_samples[i] = class_samples
    print(all_samples[i])

print(' ')

# Dividing the samples into training and test datasets
train_set = np.append(all_samples[1][0:20], all_samples[2][0:20], axis=0)
train_set = np.append(train_set, all_samples[3][0:20], axis=0)

test_set = np.append(all_samples[1][20:40], all_samples[2][20:40], axis=0)
test_set = np.append(test_set, all_samples[3][20:40], axis=0)

print(train_set)


def bayes_classifier(x_vec, kdes):
    """
    Classifies an input sample into class w_j determined by
    maximizing the class conditional probability for p(x|w_j).

    Keyword arguments:
        x_vec: A dx1 dimensional numpy array representing the sample.
        kdes: List of the gausssian_kde (kernel density) estimates

    Returns a tuple ( p(x|w_j)_value, class label ).

    """
    p_vals = []
    for kde in kdes:
        p_vals.append(kde.evaluate(x_vec))
    max_index, max_value = max(enumerate(p_vals), key=operator.itemgetter(1))
    return max_value, max_index + 1


class1_kde = kde.gaussian_kde(train_set[train_set[:, 2] == 1].T[0:2],
                              bw_method='scott')
class2_kde = kde.gaussian_kde(train_set[train_set[:, 2] == 2].T[0:2],
                              bw_method='scott')
class3_kde = kde.gaussian_kde(train_set[train_set[:, 2] == 3].T[0:2],
                              bw_method='scott')

print(' ')

print(class1_kde)

# classification_dict, error = empirical_error(test_set, [1, 2, 3], bayes_classifier,
#                                              [[class1_kde, class2_kde, class3_kde]])
