import numpy as np
import numpy.random as rn
from scipy import optimize, stats
import scipy.linalg as linalg


# ##############################################################################
# load_data generates a binary dataset for visualisation and testing using two
# parameters:
# * A **jitter** parameter that controls how noisy the data are; and
# * An **offset** parameter that controls the separation between the two classes.
#
# Do not change this function!
# ##############################################################################
def load_data(N=50, jitter=0.7, offset=1.2):
    # Generate the data
    x = np.vstack([rn.normal(0, jitter, (N // 2, 1)),
                   rn.normal(offset, jitter, (N // 2, 1))])
    y = np.vstack([np.zeros((N // 2, 1)), np.ones((N // 2, 1))])
    x_test = np.linspace(-2, offset + 2).reshape(-1, 1)

    # Make the augmented data matrix by adding a column of ones
    x_train = np.hstack([np.ones((N, 1)), x])
    x_test = np.hstack([np.ones((N, 1)), x_test])
    return x_train, y, x_test


# ##############################################################################
# predict takes a input matrix X and parameters of the logistic regression theta
# and predicts the output of the logistic regression.
# ##############################################################################
def predict(X, theta):
    # X: K x D matrix of test inputs
    # theta: D x 1 vector of parameters
    # returns: prediction of f(X); K x 1 vector
    prediction = np.zeros((X.shape[0], 1))
    z = X.dot(theta)
    prediction =1 / (1 + np.exp(-z))
    # Task 1:
    # TODO: Implement the prediction of a logistic regression here.

    return prediction


def predict_binary(X, theta):
    # X: K x D matrix of test inputs
    # theta: D x 1 vector of parameters
    # returns: binary prediction of f(X); K x 1 vector; should be 0 or 1

    prediction = 1. * (predict(X, theta) > 0.5)

    return prediction


# ##############################################################################
# log_likelihood takes data matrices x and y and parameters of the logistic
# regression theta and returns the log likelihood of the data given the logistic
# regression.
# ##############################################################################
def log_likelihood(X, y, theta):
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # theta: parameters (D x 1)
    # returns: log likelihood, scalar
    # returns: log likelihood, scalar
    y_pred = predict(X,theta)
    L = 0
    L  = np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


    # Task 2:
    # TODO: Calculate the log-likelihood of a dataset
    # given a value of theta.

    return L


# ##############################################################################
# max_lik_estimate takes data matrices x and y ands return the maximum
# likelihood parameters of a logistic regression.
# ##############################################################################
def max_lik_estimate(X, y):
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # returns: maximum likelihood parameters (D x 1)

    N, D = X.shape
    theta_init = rn.rand(D, 1)
    theta_ml = theta_init
    func = lambda x: (-log_likelihood(X, y, x.reshape(-1,1)), np.dot(X.T,(predict(X, x.reshape(-1,1)) - y)).reshape(-1))
    #3 Take the average cost derivative for each feature
    res = optimize.minimize(func, theta_ml.reshape(-1), method='BFGS', jac=True)
    theta_ml = res.x
    # Task 3:
    # TODO: Optimize the log-likelihood function you've
    # written above an obtain a maximum likelihood estimate

    return theta_ml


# ##############################################################################
# neg_log_posterior takes data matrices x and y and parameters of the logistic
# regression theta as well as a prior mean m and covariance S and returns the
# negative log posterior of the data given the logistic regression.
# ##############################################################################
def neg_log_posterior(theta, X, y, m, S):
    # theta: D x 1 matrix of parameters
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # m: D x 1 prior mean of parameters
    # S: D x D prior covariance of parameters
    # returns: scalar negative log posterior

    L = log_likelihood(X,y,theta)
    negative_log_posterior = 0
    negative_log_posterior = 0.5*(theta - m).T.dot(np.linalg.inv(S)).dot(theta-m) - L

    # TODO: Calculate the log-posterior

    return np.asscalar(negative_log_posterior)


# ##############################################################################
# map_estimate takes data matrices x and y as well as a prior mean m and
# covariance  and returns the maximum a posteriori parameters of a logistic
# regression.
# ##############################################################################
def map_estimate(X, y, m, S):
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # m: D x 1 prior mean of parameters
    # S: D x D prior covariance of parameters
    # returns: maximum a posteriori parameters (D x 1)

    N, D = X.shape
    theta_init = rn.rand(D, 1)
    theta_map = theta_init
    func = lambda x: neg_log_posterior(x.reshape(-1,1), X,y,m,S)
    gradient = lambda x: (linalg.inv(S).dot(x.reshape(-1,1)-m) + np.dot(X.T,predict(X, x.reshape(-1,1)) - y)).reshape(-1)

    res = optimize.minimize(func, theta_map.reshape(-1),method='BFGS', jac=gradient, options={'disp':True})
    theta_map = res.x
    # Task 5:
    # TODO: Optimize the log-posterior function you've
    # written above an obtain a maximum a posteriori estimate

    return theta_map


# ##############################################################################
# laplace_q takes an array of points z and returns an array with Laplace
# approximation q evaluated at all points in z.
# ##############################################################################
def laplace_q(z):
    # z: double array of size (T,)
    # returns: array with Laplace approximation q evaluated
    #          at all points in z

    q = np.zeros_like(z)
    q = stats.norm.pdf(z,2,2)
    # Task 6:
    # TODO: Evaluate the Laplace approximation $q(z)$.

    return q


# ##############################################################################
# get_posterior takes data matrices x and y as well as a prior mean m and
# covariance and returns the maximum a posteriori solution to parameters
# of a logistic regression as well as the covariance approximated with the
# Laplace approximation.
# ##############################################################################
def get_posterior(X, y, m, S):
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # m: D x 1 prior mean of parameters
    # S: D x D prior covariance of parameters
    # returns: maximum a posteriori parameters (D x 1)
    #          covariance of Laplace approximation (D x D)

    mu_post = np.zeros_like(m)
    S_post = np.zeros_like(S)
    theta_map = map_estimate(X,y,m,S)
    y_pred = predict(X,theta_map)
    H = np.zeros_like(S)
    for i in range(X.shape[0]):
        H += y_pred[i]*(1-y_pred[i])*np.outer(X[i,:],X[i,:])
    S_post = linalg.inv(linalg.inv(S) + H)
    mu_post = theta_map
    # Task 7:
    # TODO: Calculate the Laplace approximation of p(theta | X, y)

    return mu_post, S_post


# ##############################################################################
# metropolis_hastings_sample takes data matrices x and y as well as a prior mean
# m and covariance and the number of iterations of a sampling process.
# It returns the sampling chain of the parameters of the logistic regression
# using the Metropolis algorithm.
# ##############################################################################
def metropolis_hastings_sample(X, y, m, S, nb_iter):
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # m: D x 1 prior mean of parameters
    # S: D x D prior covariance of parameters
    # returns: nb_iter x D matrix of posterior samples

    D = X.shape[1]
    samples = np.zeros((nb_iter, D))
    mu_post,S_post = get_posterior(X,y,m,S)
    mu_post.reshape(-1)


    samples[0,:] = rn.rand(D,1).reshape(-1)
    for i in range(1,nb_iter):
        curr = samples[i-1,:]
        cand = stats.multivariate_normal.rvs(curr,np.identity(D))
        acc = min(1,stats.multivariate_normal.pdf(cand, mu_post,S_post) /stats.multivariate_normal.pdf(curr, mu_post,S_post)  )
        if rn.rand() < acc:
            samples[i]  = cand
        else:
            samples[i]  = curr
    # Task 8:
    # TODO: Write a function to sample from the posterior of the
    # parameters of the logistic regression p(theta | X, y) using the
    # Metropolis algorithm.


    return samples
