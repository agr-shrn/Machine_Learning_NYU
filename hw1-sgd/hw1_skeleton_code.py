import math
import pandas as pd
import logging
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

### Assignment Owner: Tian Wang

#######################################
#### Normalization


def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test  - test set, a 2D numpy array of size (num_instances, num_features)
    Returns:
        train_normalized - training set after normalization
        test_normalized  - test set after normalization

    """
    # TODO
    max_train = np.zeros(train.shape[1])
    min_train = np.zeros(train.shape[1])
    
    train_normalized = np.zeros_like(train)
    test_normalized = np.zeros_like(test)

    # for i in range(len(train[0])):
        
    max_train = np.amax(train, 0)
    min_train = np.amin(train, 0)

    train_normalized = (train - min_train)/(max_train - min_train)
    test_normalized  = (test - min_train)/(max_train - min_train)
    

    return(train_normalized, test_normalized)


########################################
#### The square loss function

def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the square loss for predicting y with X*theta
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)
    
    Returns:
        loss - the square loss, scalar
    """
    loss = 0 #initialize the square_loss
    #TODO
    # lossMatrix = np.square(np.dot(X,theta)-y)
    # return np.sum(lossMatrix)/(2*len(X))
    loss = (np.dot(X, theta) - y)
    return (np.dot(loss.T, loss)) / (2 * X.shape[0])


########################################
### compute the gradient of square loss function
def compute_square_loss_gradient(X, y, theta):
    """
    Compute gradient of the square loss (as defined in compute_square_loss), at the point theta.
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
    
    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    #TODO
    grad = np.dot(X.T, np.dot(X, theta) - y)/X.shape[0]
    return grad 
    # gradientMatrix = (X.T * (np.dot(X, theta) - y)).T
    # return np.sum(gradientMatrix, axis=0)/len(X)
        
       
        
###########################################
### Gradient Checker
#Getting the gradient calculation correct is often the trickiest part
#of any gradient-based optimization algorithm.  Fortunately, it's very
#easy to check that the gradient calculation is correct using the
#definition of gradient.
#See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4): 
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions: 
    (e_1 = (1,0,0,...,0), e_2 = (0,1,0,...,0), ..., e_d = (0,...,0,1) 

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by: 
    ( J(theta + epsilon * e_i) - J(theta - epsilon * e_i) ) / (2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error
    
    Return:
        A boolean value indicate whether the gradient is correct or not

    """
    true_gradient = compute_square_loss_gradient(X, y, theta) #the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    #TODO
    e = np.zeros(num_features)

    for i in range(num_features):
        e[i] = 1
        approx_grad[i] = (compute_square_loss(X, y, theta + epsilon * e) - compute_square_loss(X, y, theta - epsilon * e))/(2 * epsilon)
        e[i] = 0

    # print(dist)
    
    # dist = np.linalg.norm(true_gradient - approx_grad)    
        
    # if dist > tolerance:
    #     return False
    # else:
    #     return True
    return np.allclose(approx_grad, true_gradient, atol = tolerance)
    
#################################################
### Generic Gradient Checker
def generic_gradient_checker(X, y, theta, objective_func, gradient_func, epsilon=0.01, tolerance=1e-4):
    """
    The functions takes objective_func and gradient_func as parameters. And check whether gradient_func(X, y, theta) returned
    the true gradient for objective_func(X, y, theta).
    Eg: In LSR, the objective_func = compute_square_loss, and gradient_func = compute_square_loss_gradient
    """
    true_gradient = gradient_func(X, y, theta) #the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    #TODO
    e = np.zeros(num_features)

    for i in range(num_features):
        e[i] = 1
        approx_grad[i] = (objective_func(X, y, theta + epsilon * e) - objective_func(X, y, theta - epsilon * e))/(2 * epsilon)
        e[i] = 0

    dist = np.linalg.norm(true_gradient - approx_grad)    
        
    # print(dist)
    
    if dist > tolerance:
        return False
    else:
        return True

####################################
#### Batch Gradient Descent
def batch_grad_descent(X, y, alpha=0.5, num_iter=1000, check_gradient=False):
    """
    In this question you will implement batch gradient descent to
    minimize the square loss objective
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        num_iter - number of iterations to run 
        check_gradient - a boolean value indicating whether checking the gradient when updating
        ;
    Returns:
        theta_hist - store the the history of parameter vector in iteration, 2D numpy array of size (num_iter+1, num_features) 
                    for instance, theta in iteration 0 should be theta_hist[0], theta in iteration (num_iter) is theta_hist[-1]
        loss_hist - the history of objective function vector, 1D numpy array of size (num_iter+1) 
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    theta = np.ones(num_features) #initialize theta
    #TODO   
    # batch_size=4
    # sample_size=100
    theta_hist[0] = theta
    loss_hist[0] = compute_square_loss(X, y, theta)
    
    for i in range(num_iter):
        
        grad = compute_square_loss_gradient(X, y, theta) 
        # print(grad)

        # if not (grad_checker(X, y, theta)):
        #     print("BUG IN THE CODE")

        theta = theta - (alpha * grad)  #alpha = 0.093
        
        loss_hist[i + 1] = compute_square_loss(X, y, theta)
        theta_hist[i + 1] = theta  
    
    # print(loss_hist[1000])
    plt.plot(loss_hist)
    plt.xlabel('Number of iterations')
    plt.ylabel('Objective function')
    plt.show()
    return(loss_hist, theta_hist)

        # for j in range(0,int(sample_size/batch_size)):

        #     x_batch=X[batch_size*j : batch_size*(j+1),:]
        #     y_batch=y[batch_size*j : batch_size*(j+1)]     
            
        # x_batch = X[np.random.randint(X.shape[0], size=8), :]
        # y_batch = np.random.choice(y, 8)
        
        # loss_hist[i] = compute_square_loss(x_batch, y_batch, theta)
        # grad = compute_square_loss_gradient(x_batch, y_batch, theta)
        
        #update parameters


####################################
###Q2.4b: Implement backtracking line search in batch_gradient_descent
###Check http://en.wikipedia.org/wiki/Backtracking_line_search for details
#TODO
    


###################################################
### Compute the gradient of Regularized Batch Gradient Descent
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg = 0.01):
    """
    Compute the gradient of L2-regularized square loss function given X, y and theta
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        lambda_reg - the regularization coefficient
    
    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    #TODO
    grad = (np.dot(X.T, np.dot(X, theta) - y))/X.shape[0] + 2 * lambda_reg * theta
    return grad

###################################################
### Batch Gradient Descent with regularization term
def regularized_grad_descent(X, y, alpha, lambda_reg = 0.01, num_iter=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        numIter - number of iterations to run 
        
    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_iter+1, num_features) 
        loss_hist - the history of regularized loss value, 1D numpy array
    """
    (num_instances, num_features) = X.shape
    theta = np.ones(num_features) #Initialize theta
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #Initialize loss_hist
    #TODO
    theta_hist[0] = theta
    loss_hist[0] = compute_square_loss(X, y, theta)
    
    for i in range(num_iter):
        
        grad = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg) 

        theta = theta - (alpha * grad)  #alpha = 0.093
        
        loss_hist[i + 1] = compute_square_loss(X, y, theta) + lambda_reg * np.dot(theta.T, theta)
        theta_hist[i + 1] = theta  
    
    return(loss_hist, theta_hist)



#############################################
## Visualization of Regularized Batch Gradient Descent
##X-axis: log(lambda_reg)
##Y-axis: square_loss

#############################################
### Stochastic Gradient Descent
def stochastic_grad_descent(X, y, alpha=0.01, lambda_reg=0.01, num_iter=1000):
    """
    In this question you will implement stochastic gradient descent with a regularization term
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - string or float. step size in gradient descent
                NOTE: In SGD, it's not always a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every iteration is alpha.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t)
                if alpha == "1/t", alpha = 1/t
        lambda_reg - the regularization coefficient
        num_iter - number of epochs (i.e number of times) to go through the whole training set
    
    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size (num_iter, num_instances, num_features) 
        loss hist - the history of regularized loss function vector, 2D numpy array of size(num_iter, num_instances)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features) #Initialize theta
    
    
    theta_hist = np.zeros((num_iter, num_instances, num_features))  #Initialize theta_hist
    loss_hist = np.zeros((num_iter, num_instances)) #Initialize loss_hist
    plot_loss_hist = np.zeros(num_iter)
    #TODO

    batch_size=1
    sample_size=100

    for i in range(num_iter):

        # alpha = 1/np.sqrt(i+1)

        for j in range(int(sample_size/batch_size)):

            x_batch=X[j]
            y_batch=y[j]

            grad = compute_regularized_square_loss_gradient(x_batch, y_batch, theta, 0.01)               
            
            theta = theta - (alpha * grad)  #alpha = 0.08
            
            theta_hist[i][j] = theta
            loss_hist[i][j] = compute_square_loss(x_batch, y_batch, theta) + lambda_reg * np.dot(theta.T, theta)

        plot_loss_hist[i] = compute_square_loss(X, y, theta) + lambda_reg * np.dot(theta.T, theta)  
    
    plt.plot(plot_loss_hist)
    plt.xlabel('Number of epochs')
    plt.ylabel('log(objective function)')
    plt.show()
    print(plot_loss_hist[900:999])
    return(loss_hist, theta_hist)

################################################
### Visualization that compares the convergence speed of batch
###and stochastic gradient descent for various approaches to step_size
##X-axis: Step number (for gradient descent) or Epoch (for SGD)
##Y-axis: log(objective_function_value) and/or objective_function_value
def main():
    #Loading the dataset
    print('loading the dataset')
    
    df = pd.read_csv('hw1-data.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)

    
    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1)))) # Add bias term
    x_coor = np.arange(1001)
    # TODO
    theta = np.ones(49)
    loss_hist = np.zeros(1001)
#     X = np.array([[ 5.0, 1.0 ,3.0],
# -                  [ 1.0, 1.0 ,1.0],
# -                  [ 1.0, 2.0 ,1.0]])
# -    theta = np.array([1.0, 2.0, 3.0])
# -    y = np.array([10.0, 4.0, 4.0])

    print(compute_square_loss(X_train, y_train, theta))
    loss_hist, theta_hist = stochastic_grad_descent(X_train, y_train)
    # print(loss_hist[:,-1])
    # plt.plot(loss_hist[:,-1])
    # plt.show()
    # print(compute_regularized_square_loss_gradient(X_train, y_train, theta, 0.01))
    
    # min_loss = np.zeros(7)
    # log_lambda = [-7, -5, -3, -2, -1, 0, 1]

    # for i in range(7):
    #     loss_hist = np.zeros(1001)
    #     loss_hist, theta_hist = regularized_grad_descent(X_train, y_train, 0.05, pow(10, log_lambda[i]))
    #     theta = theta_hist[1000]
    #     print(theta)
    #     min_loss[i] = compute_square_loss(X_train, y_train, theta)
    #     # print(min_loss[i])
    
    # plt.plot(log_lambda, min_loss)
    # plt.xlabel('log lambda')
    # plt.ylabel('Train loss')
    # plt.show()
    

    # print(compute_square_loss_gradient(X_train, y_train, theta))
    # print(generic_gradient_checker(X_train,y_train,theta, compute_square_loss, compute_square_loss_gradient))
    # loss_hist, theta_hist = batch_grad_descent(X_train, y_train)

    # loss_hist, theta_hist = batch_grad_descent(X_train, y_train, 0.05)
    # plt.plot(x_coor, loss_hist, label = 'aplha = 0.05')
    # print(loss_hist[900:1000])
    # loss_hist, theta_hist = batch_grad_descent(X_train, y_train, 0.5)
    # plt.plot(x_coor, loss_hist, label = 'aplha = 0.5')
    # print(loss_hist[900:1000])
    # loss_hist, theta_hist = batch_grad_descent(X_train, y_train, 0.1)
    # plt.plot(x_coor, loss_hist,label = 'aplha = 0.1')
    # print(loss_hist[900:1000])
    # loss_hist, theta_hist = batch_grad_descent(X_train, y_train, 0.01)
    # plt.plot(x_coor, loss_hist, label = 'aplha = 0.01')
    # print(loss_hist[900:1000])
    # plt.legend()
    # plt.show()
    # loss_hist, theta_hist = regularized_grad_descent(X_train, y_train, 0.1)
    # plt.plot(x_coor, loss_hist, label = 'aplha = 0.1')
    # plt.show()
    # loss_hist, theta_hist = stochastic_grad_descent(X_train, y_train)
    # print(loss_hist)
    # print(theta_hist)
    #     print(X_test)

if __name__ == "__main__":
    main()
