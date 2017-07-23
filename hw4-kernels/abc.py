class Kernel_Machine(object):
    def _init_(self, kernel, prototype_points, weights):
        """
        Args:
            kernel(W,X) - a function return the cross-kernel matrix between rows of W and rows of X for kernel k
            prototype_points - an Rxd matrix with rows mu_1,...,mu_R
            weights - a vector of length R
        """

        self.kernel = kernel
        self.prototype_points = prototype_points
        self.weights = weights
        
    def predict(self, X):
        """
        Evaluates the kernel machine on the points given by the rows of X
        Args:
            X - an nxd matrix with inputs x_1,...,x_n in the rows
        Returns:
            Vector of kernel machine evaluations on the n points in X.  Specifically, jth entry of return vector is
                Sum_{i=1}^R w_i k(x_j, mu_i)
        """
        # TODO
        # PLot kernel machine functions
        prototypes = self.prototype_points
        xpts = X
        K = self.kernel(prototypes, xpts) 
        return np.dot(np.transpose(K), self.weights)

sigma = 1
k = functools.partial(RBF_kernel, sigma = sigma)
prototypes = np.array([-1,0,1]).reshape(-1,1)
weights = np.array([1,-1,1]).reshape(-1,1)
RBF_K = Kernel_Machine(k, prototypes, weights)
plot_step = .01
xpts = np.arange(-6.0, 6, plot_step).reshape(-1,1)
fx = RBF_K.predict(xpts)
plt.plot(xpts, fx)
plt.xlabel("Xpts")
plt.ylabel("Predicted value")
plt.show()