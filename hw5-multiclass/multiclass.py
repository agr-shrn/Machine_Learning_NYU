import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets.samples_generator import make_blobs
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn import metrics
from sklearn import svm
from sklearn import metrics



def zeroOne(y,a) :
    '''
    Computes the zero-one loss.
    @param y: output class
    @param a: predicted class
    @return 1 if different, 0 if same
    '''
    return int(y != a)

def featureMap(X,y,num_classes) :
    '''
    Computes the class-sensitive features.
    @param X: array-like, shape = [n_samples,n_inFeatures] or [n_inFeatures,], input features for input data
    @param y: a target class (in range 0,..,num_classes-1)
    @return array-like, shape = [n_samples,n_outFeatures], the class sensitive features for class y
    '''
    #The following line handles X being a 1d-array or a 2d-array
    num_samples, num_inFeatures = (1,X.shape[0]) if len(X.shape) == 1 else (X.shape[0],X.shape[1])
    num_outFeatures = num_classes * num_inFeatures  
    
    dim = (num_samples, num_outFeatures)
    phi = np.zeros(dim)

    if num_samples == 1:

        new_row = np.zeros(num_outFeatures)
        new_row[y*num_inFeatures:(y+1)*num_inFeatures] = X
        return new_row 
        
    for i in range(X.shape[0]):

        new_row = np.zeros(num_outFeatures)
        new_row[y[i]*num_inFeatures:(y[i]+1)*num_inFeatures] = X[i]
        phi[i,:] = new_row

    return phi




def sgd(X, y, num_outFeatures, subgd, eta = 0.01, T = 10000):
    
    '''
    Runs subgradient descent, and outputs resulting parameter vector.
    @param X: array-like, shape = [n_samples,n_features], input training data 
    @param y: array-like, shape = [n_samples,], class labels
    @param num_outFeatures: number of class-sensitive features
    @param subgd: function taking x,y and giving subgradient of objective
    @param eta: learning rate for SGD
    @param T: maximum number of iterations
    @return: vector of weights
    '''
    num_samples = X.shape[0]
    
    w = np.zeros(num_outFeatures)
    meanw = np.zeros(num_outFeatures)

    ind_arr = list(range(num_samples))

    for i in range(T):
        
        # np.random.shuffle(ind_arr)
        j = np.random.randint(num_samples)
        # for j in ind_arr: 
            
        gradient = subgd(X[j],y[j],w)
        w = w - eta * gradient
        meanw += w
    
    return meanw/T

class MulticlassSVM(BaseEstimator, ClassifierMixin):
    '''
    Implements a Multiclass SVM estimator.
    '''
    def __init__(self, num_outFeatures, lam=1.0, num_classes=3, Delta=zeroOne, Psi=featureMap):       
        '''
        Creates a MulticlassSVM estimator.
        @param num_outFeatures: number of class-sensitive features produced by Psi
        @param lam: l2 regularization parameter
        @param num_classes: number of classes (assumed numbered 0,..,num_classes-1)
        @param Delta: class-sensitive loss function taking two arguments (i.e., target margin)
        @param Psi: class-sensitive feature map taking two arguments
        '''
        self.num_outFeatures = num_outFeatures
        self.lam = lam
        self.num_classes = num_classes
        self.Delta = Delta
        self.Psi = lambda X,y : Psi(X,y,num_classes)
        self.fitted = False
    
    def subgradient(self,x,y,w):
        '''
        Computes the subgradient at a given data point x,y
        @param x: sample input
        @param y: sample class
        @param w: parameter vector
        @return returns subgradient vector at given x,y,w
        '''
        #Your code goes here and replaces the following return statement

        y_cap = np.zeros(self.num_classes)

        for p in range(self.num_classes):

            y_cap[p] = self.Delta(y, p) + np.dot(w, (self.Psi(x,p) - self.Psi(x,y)))

        res = 2 * self.lam * w + self.Psi(x, np.argmax(y_cap)) - self.Psi(x,y)

        return res
    def fit(self,X,y,eta=0.1,T=10000):
        '''
        Fits multiclass SVM
        @param X: array-like, shape = [num_samples,num_inFeatures], input data
        @param y: array-like, shape = [num_samples,], input classes
        @param eta: learning rate for SGD
        @param T: maximum number of iterations
        @return returns self
        '''
        self.coef_ = sgd(X,y,self.num_outFeatures,self.subgradient,eta,T)
        self.fitted = True
        return self
    
    def decision_function(self, X):
        '''
        Returns the score on each input for each class. Assumes
        that fit has been called.
        @param X : array-like, shape = [n_samples, n_inFeatures]
        @return array-like, shape = [n_samples, n_classes] giving scores for each sample,class pairing
        '''
        if not self.fitted:
            raise RuntimeError("You must train classifer before predicting data.")

        s = (X.shape[0], self.num_classes)
        dec = np.zeros(s)

        for i in range(X.shape[0]):

            new_row = np.zeros(self.num_classes)

            for j in range(self.num_classes):
    
                new_row[j] = np.dot(self.coef_, self.Psi(X[i],j))

            dec[i] = new_row

        return(dec)

    def predict(self, X):
        '''
        Predict the class with the highest score.
        @param X: array-like, shape = [n_samples, n_inFeatures], input data to predict
        @return array-like, shape = [n_samples,], class labels predicted for each data point
        '''

        dec = self.decision_function(X)
        prediction = np.zeros(X.shape[0])

        for i in range(X.shape[0]):

            prediction[i] = np.argmax(dec[i])

        return prediction


#the following code tests the MulticlassSVM and sgd
#will fail if MulticlassSVM is not implemented yet
np.random.seed(2)
X, y = make_blobs(n_samples=300,cluster_std=.25, centers=np.array([(-3,1),(0,2),(3,1)]))
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50)
# plt.show()

svm_estimator = svm.LinearSVC(loss='hinge', fit_intercept=False, C=200)
# clf_onevsall = OneVsAllClassifier(svm_estimator, n_classes=3)
# clf_onevsall.fit(X,y)

# for i in range(3) :

#     print("Coeffs %d"%i)
#     clf_onevsall.estimators[i].coef_ = sklearn.preprocessing.normalize(clf_onevsall.estimators[i].coef_)
#     print(clf_onevsall.estimators[i].coef_) #Will fail if you haven't implemented fit yet

# create a mesh to plot in
h = .02  # step size in the mesh
x_min, x_max = min(X[:,0])-3,max(X[:,0])+3
y_min, y_max = min(X[:,1])-3,max(X[:,1])+3
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
mesh_input = np.c_[xx.ravel(), yy.ravel()]

# Z = clf_onevsall.predict(mesh_input)
# Z = Z.reshape(xx.shape)
# plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
# # Plot also the training points
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
# plt.show()

# metrics.confusion_matrix(y, clf_onevsall.predict(X))



est = MulticlassSVM(6,lam=1)
est.fit(X,y)
print("w:")
print(est.coef_)
Z = est.predict(mesh_input)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.show()
metrics.confusion_matrix(y, est.predict(X))