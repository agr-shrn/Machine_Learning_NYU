import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets.samples_generator import make_blobs
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn import metrics
from sklearn import svm

# %matplotlib inline


class OneVsAllClassifier(BaseEstimator, ClassifierMixin):  
    """
    One-vs-all classifier
    We assume that the classes will be the integers 0,..,(n_classes-1).
    We assume that the estimator provided to the class, after fitting, has a "decision_function" that 
    returns the score for the positive class.
    """
    def __init__(self, estimator, n_classes):      
        """
        Constructed with the number of classes and an estimator (e.g. an
        SVM estimator from sklearn)
        @param estimator : binary base classifier used
        @param n_classes : number of classes
        """
        self.n_classes = n_classes 
        self.estimators = [clone(estimator) for _ in range(n_classes)]
        self.fitted = False

    def fit(self, X, y=None):
        """
        This should fit one classifier for each class.
        self.estimators[i] should be fit on class i vs rest
        @param X: array-like, shape = [n_samples,n_features], input data
        @param y: array-like, shape = [n_samples, class labels]
        @return returns self
        """
        #Your code goes here

        for i in range(self.n_classes):
            
            tempy = np.copy(y)
            
            for j in range(len(tempy)):

                if tempy[j] == i:
                    tempy[j] = 1

                else:
                    tempy[j] = -1

                self.estimators[i].fit(X, tempy)

        self.fitted = True  
        return self   

    def decision_function(self, X):
        """
        Returns the score of each input for each class. Assumes
        that the given estimator also implements the decision_function method (which sklearn SVMs do), 
        and that fit has been called.
        @param X : array-like, shape = [n_samples, n_features] input data
        @return array-like, shape = [n_samples, n_classes]
        """
        if not self.fitted:
            raise RuntimeError("You must train classifer before predicting data.")

        if not hasattr(self.estimators[0], "decision_function"):
            raise AttributeError(
                "Base estimator doesn't have a decision_function attribute.")
        
        #Replace the following return statement with your code
        
        s = (X.shape[0], self.n_classes)
        dec = np.zeros(s)
        # dec[:,0] = X

        for i in range(self.n_classes):

            dec[:,i] = self.estimators[i].decision_function(X)

        return(dec)





    def predict(self, X):
        """
        Predict the class with the highest score.
        @param X: array-like, shape = [n_samples,n_features] input data
        @returns array-like, shape = [n_samples,] the predicted classes for each input
        """
        #Replace the following return statement with your code

        prediction = np.zeros(X.shape[0])
        dec = self.decision_function(X)

        for i in range(X.shape[0]):

            prediction[i] = np.argmax(dec[i])

        return prediction


np.random.seed(2)
X, y = make_blobs(n_samples=300,cluster_std=.25, centers=np.array([(-3,1),(0,2),(3,1)]))
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50)
# plt.show()

svm_estimator = svm.LinearSVC(loss='hinge', fit_intercept=False, C=200)
clf_onevsall = OneVsAllClassifier(svm_estimator, n_classes=3)
clf_onevsall.fit(X,y)

for i in range(3) :

    print("Coeffs %d"%i)
    clf_onevsall.estimators[i].coef_ = sklearn.preprocessing.normalize(clf_onevsall.estimators[i].coef_)
    print(clf_onevsall.estimators[i].coef_) #Will fail if you haven't implemented fit yet

# create a mesh to plot in
h = .02  # step size in the mesh
x_min, x_max = min(X[:,0])-3,max(X[:,0])+3
y_min, y_max = min(X[:,1])-3,max(X[:,1])+3
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
mesh_input = np.c_[xx.ravel(), yy.ravel()]

Z = clf_onevsall.predict(mesh_input)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.show()

metrics.confusion_matrix(y, clf_onevsall.predict(X))
