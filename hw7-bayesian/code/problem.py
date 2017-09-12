from _future_ import division
import matplotlib.pyplot as plt
import numpy.matlib as matlib
from scipy.stats import multivariate_normal
import numpy as np
import support_code

def likelihoodFunc(W, x, y_train, likelihood_var):
    '''
    Implement likelihoodFunc. This function returns the data likelihood
    given f(y_train | x; W) ~ Normal(w^Tx, likelihood_var).

    Args:
        W: Weights
        x: Training design matrix with first col all ones (np.matrix)
        y_train: Training response vector (np.matrix)
        likelihood_var: likelihood variance

    Returns:
        likelihood: Data likelihood (float)
    '''

    #TO DO
    likelihood = 1.0
    fixed_term = 1.0/(np.sqrt(2 * np.pi * likelihood_var))

    for i in range(len(x)):

        likelihood = likelihood * fixed_term * np.exp(-1 * ((np.square(y_train[i] - np.dot(x[i], W.T)))/(2 * likelihood_var))) 

    return likelihood

    # likelihood = 1
    # for idx,row in enumerate(x):
    #     const = 1.0/np.sqrt(2*np.pi*likelihood_var)
    #     #import pdb
    #     #pdb.set_trace()
    #     square_loss = np.square(y_train[idx] - np.dot(row,W.T))
    #     likelihood*= const*np.exp(-1*(square_loss/(2*likelihood_var)))
    # return likelihood

def getPosteriorParams(x, y_train, prior, likelihood_var = 0.2**2):
    '''
    Implement getPosterioParams. This function returns the posterior
    mean vector \mu_p and posterior covariance matrix \Sigma_p for
    Bayesian regression (normal likelihood and prior).

    Note support_code.make_plots takes this completed function as an argument.

    Args:
        x: Training design matrix with first col all ones (np.matrix)
        y_train: Training response vector (np.matrix)
        prior: Prior parameters; dict with 'mean' (prior mean np.matrix)
               and 'var' (prior covariance np.matrix)
        likelihood_var: likelihood variance- default (0.2**2) per the lecture slides

    Returns:
        postMean: Posterior mean (np.matrix)
        postVar: Posterior mean (np.matrix)
    '''
    # TO DO

    # This infact assume prior mean are all zero

    x_square = np.dot(x.T,x)
    first_term_mean = np.matrix.getI(x_square + (likelihood_var**2) * np.matrix.getI(prior['var']))
    second_term_mean = np.dot(x.T, y_train)
    postMean = np.dot(first_term_mean, second_term_mean)
    
    first_term_var = x_square/(likelihood_var**(2))
    second_term_var = np.matrix.getI(prior['var'])
    postVar = np.matrix.getI(first_term_var + second_term_var)


    # postMean = np.dot(np.dot(np.matrix.getI(np.dot(x.T,x) + likelihood_var*np.matrix.getI(prior['var'])),x.T),y_train)
    # postVar = np.matrix.getI(np.dot(x.T,x)/likelihood_var + np.matrix.getI(prior['var']))

    return postMean, postVar

def getPredictiveParams(x_new, postMean, postVar, likelihood_var = 0.2**2):
    '''
    Implement getPredictiveParams. This function returns the predictive
    distribution parameters (mean and variance) given the posterior mean
    and covariance matrix (returned from getPosteriorParams) and the
    likelihood variance (default value from lecture).

    Args:
        x: New observation (np.matrix object)
        postMean, postVar: Returned from getPosteriorParams
        likelihood_var: likelihood variance (0.2**2) per the lecture slides

    Returns:
        - predMean: Mean of predictive distribution
        - predVar: Variance of predictive distribution
    '''

    # TO DO

    x_new = x_new.reshape(-1,1)
    predMean = np.dot(postMean.T, x_new)
    prod = np.dot(x_new.T, postVar)
    predVar = np.dot(prod, x_new) + likelihood_var**2

    return predMean, predVar

if __name__ == '__main__':

    '''
    If your implementations are correct, running
        python problem.py
    inside the Bayesian Regression directory will, for each sigma in sigmas_to-test generates plots
    '''
    np.random.seed(46134)
    actual_weights = np.matrix([[0.3], [0.5]])
    dataSize = 800
    noise = {"mean":0, "var":0.2 ** 2}
    likelihood_var = noise["var"]
    xtrain, ytrain = support_code.generateData(dataSize, noise, actual_weights)

    #Question (b)
    sigmas_to_test = [1/2, 1/(2**5), 1/(2**10)]
    for sigma_squared in sigmas_to_test:
        prior = {"mean":np.matrix([[0], [0]]),
                 "var":matlib.eye(2) * sigma_squared}

        support_code.make_plots(actual_weights,
                                xtrain,
                                ytrain,
                                likelihood_var,
                                prior,
                                likelihoodFunc,
                                getPosteriorParams,
                                getPredictiveParams)
