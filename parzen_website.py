"""
Website:
https://sebastianraschka.com/Articles/2014_kernel_density_est.html
"""
import numpy as np
import operator


def window_function(x_vec, unit_len=1):
    """
    Implementation of the window function. Returns 1 if 3x1-sample vector
    lies within a origin-centered hypercube, 0 otherwise.

    """
    for row in x_vec:
        if np.abs(row) > (unit_len / 2):
            return 0
    return 1


"""
X_all = np.vstack((X_inside,X_outside))
assert(X_all.shape == (10,3))

k_n = 0
for row in X_all:
    k_n += window_function(row.reshape(3,1))

print('Points inside the hypercube:', k_n)
print('Points outside the hybercube:', len(X_all) - k_n)
"""


def parzen_window_est(x_samples, h=1, center=[0, 0, 0]):
    """
    Implementation of the Parzen-window estimation for hypercubes.

    Keyword arguments:
        x_samples: A 'n x d'-dimensional numpy array, where each sample
            is stored in a separate row.
        h: The length of the hypercube.
        center: The coordinate center of the hypercube

    Returns the probability density for observing k samples inside the hypercube.

    """
    dimensions = x_samples.shape[1]

    assert (len(center) == dimensions), \
        'Number of center coordinates have to match sample dimensions'

    k = 0
    for x in x_samples:
        is_inside = 1
        for axis, center_point in zip(x, center):
            if np.abs(axis - center_point) > (h / 2):
                is_inside = 0
        k += is_inside
    return (k / len(x_samples)) / (h ** dimensions)


# print('p(x) =', parzen_window_est(X_all, h=1))


def pdf_multivariate_gauss(x, mu, cov):
    """
    Caculate the multivariate normal density (pdf)

    Keyword arguments:
        x = numpy array of a "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    """

    # mu_vec = np.array([0, 0])
    # cov_mat = np.array([[1, 0], [0, 1]])

    assert (mu.shape[0] > mu.shape[1]), \
        'mu must be a row vector'
    assert (x.shape[0] > x.shape[1]), \
        'x must be a row vector'
    assert (cov.shape[0] == cov.shape[1]), \
        'covariance matrix must be square'
    assert (mu.shape[0] == cov.shape[0]), \
        'cov_mat and mu_vec must have the same dimensions'
    assert (mu.shape[0] == x.shape[0]), \
        'mu and x must have the same dimensions'

    part1 = 1 / (((2 * np.pi) ** (len(mu) / 2)) * (np.linalg.det(cov) ** (1 / 2)))
    part2 = (-1 / 2) * ((x - mu).T.dot(np.linalg.inv(cov))).dot((x - mu))
    return float(part1 * np.exp(part2))


def hypercube_kernel(h, x, x_i):
    """
    Implementation of a hypercube kernel for Parzen-window estimation.

    Keyword arguments:
        h: window width
        x: point x for density estimation, 'd x 1'-dimensional numpy array
        x_i: point from training sample, 'd x 1'-dimensional numpy array

    Returns a 'd x 1'-dimensional numpy array as input for a window function.

    """
    assert (x.shape == x_i.shape), 'vectors x and x_i must have the same dimensions'
    return (x - x_i) / (h)


def parzen_window_func(x_vec, h=1):
    """
    Implementation of the window function. Returns 1 if 'd x 1'-sample vector
    lies within inside the window, 0 otherwise.

    """
    for row in x_vec:
        if np.abs(row) > (1 / 2):
            return 0
    return 1


def parzen_estimation(x_samples, point_x, h, d, window_func, kernel_func):
    """
    Implementation of a parzen-window estimation.

    Keyword arguments:
        x_samples: A 'n x d'-dimensional numpy array, where each sample
            is stored in a separate row. (= training sample)
        point_x: point x for density estimation, 'd x 1'-dimensional numpy array
        h: window width
        d: dimensions
        window_func: a Parzen window function (phi)
        kernel_func: A hypercube or Gaussian kernel functions

    Returns the density estimate p(x).

    """
    k_n = 0
    for row in x_samples:
        x_i = kernel_func(h=h, x=point_x, x_i=row[:, np.newaxis])
        k_n += window_func(x_i, h=h)
    return (k_n / len(x_samples)) / (h ** d)


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

# Parzen Windows using Gaussian Kernel
# from scipy.stats import kde
# gde = kde.gaussian_kde(x_2Dgauss.T, bw_method=0.3)
# silverman = kde.gaussian_kde(x_2Dgauss.T, bw_method='silverman')
