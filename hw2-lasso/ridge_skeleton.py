import numpy
from scipy.optimize import minimize

X = numpy.loadtxt("X.txt")
y = numpy.loadtxt("y.txt")

(N, D) = X.shape

w = numpy.random.rand(D, 1)


def ridge(Lambda):
    def ridge_obj(theta):
        return ((numpy.linalg.norm(numpy.dot(X, theta) - y))**2) / (2 * N) +\
            Lambda * (numpy.linalg.norm(theta))**2
    return ridge_obj


def compute_loss(Lambda, theta):
    return ((numpy.linalg.norm(numpy.dot(X, theta) - y))**2) / (2 * N)

for i in range(-5, 6):
    Lambda = 10**i
    w_opt = minimize(ridge(Lambda), w)
    print(Lambda, compute_loss(Lambda, w_opt.x))

def lasso_shooting(X,y,lambda_reg=0.1,max_steps = 1000,tolerence = 1e-5):
    start_time = time.time()
    converge = False
    steps = 0
    #Get dimension info
    n = X.shape[0]
    d = X.shape[1]
    #initializing theta
    w = np.linalg.inv(X.T.dot(X)+lambda_reg*np.identity(d)).dot(X.T).dot(y) # result w dimension: d
    def soft(a,delta):
        sign_a = np.sign(a)
        if np.abs(a)-delta <0:
            return 0 
        else:
            return sign_a*(abs(a)-delta)
    while converge==False and steps<max_steps:
        a = []
        c = []
        old_w = w
    ####For loop for computing aj cj w
        for j in range(d):
            aj = 0
            cj = 0
            for i in range(n):
                xij = X[i,j]
                aj += 2*xij*xij
                cj += 2*xij*(y[i]-w.T.dot(X[i,:])+w[j]*xij)
            w[j] = soft(cj/aj,lambda_reg/aj)
            convergence = np.sum(np.abs(w-old_w))<tolerence
            a.append(aj)
            c.append(cj)
        steps +=1
        a = np.array(a)
        c = np.array(c)
    run_time = time.time()-start_time
    print('lambda:',lambda_reg,'run_time:',run_time,'steps_taken:',steps)
    return w,a,c