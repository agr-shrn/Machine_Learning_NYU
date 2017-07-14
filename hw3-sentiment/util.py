# Taken from http://web.stanford.edu/class/cs221/ Assignment #2 Support Code
from collections import Counter
import pickle
import time


def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.

    NOTE: This function does not return anything, but rather
    increments d1 in place. We do this because it is much faster to
    change elements of d1 in place than to build a new dictionary and
    return it.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale

def optimal_lambda(lambdas,review_counters,labels):

    loss_history = []
    loss_min = 100
    
    for l in lambdas:
        
        w  = peg_improved(review_counters,labels,l)
        per_loss = per_err(review_counters,labels,w)
        loss_history.append(per_loss)
        
        if loss<loss_min:

            loss_min = loss
            best_lambda = l        
    
    return l


def stepsize_search(X, y, theta, loss_func, grad_func, epsilon=1e-6):
    
    alpha = 1.0
    gamma = 0.5
    loss = loss_func(X, y, theta)
    gradient = grad_func(X, y, theta)

    while True:
        
        theta_next = theta - alpha*grad_func(X,y, theta)
        loss_next = loss_func(X, y, theta_next)
        
        if loss_next > loss-epsilon:
            alpha = alpha*gamma
        else:
            return alpha


def loss(review_counters, labels, w, l):
    
    first_term = 0
    second_term = 0

    for key in w:
        first_term = first_term + w[key]**2

    first_term *= l/(2.0)

    for i in range(len(review_counters)):

        second_term += max(0, 1 - (labels[i] * dotProduct(w, review_counters[i])))  
    
    return first_term + second_term


def per_error(review_counters, labels, w):
    
    err = 0
    wrong_counters = []
    wrong_w = []    
    for i in range(len(review_counters)):

        if dotProduct(w,review_counters[i]) * labels[i] < 0.9:
                print(review_counters[i])
                print("Weights are:")
                print(w) 
                err+=1

    return float(err/len(review_counters)*100)


def grad(review_counters, labels, w, l):
    
    w_copy = w.copy()
    
    for key in w:
        w_copy[key] *= l
        
    if labels * dotProduct(w, review_counters) < 1:
        increment(w_copy, -labels, review_counters)
    else:
        pass
    
    return w_copy


def peg_improved(review_counters, labels, l):

    s = 1
    t = 2
    max_epoch = 20
    new_w = Counter()
    # w = Counter()

    for epoch_no in range(max_epoch):
        
        ts1 = time.time()
        j = 0
    
        for j in range(len(review_counters)):
            
            review = review_counters[j]
            label = labels[j]
            eta = 1/(t * l)

            # if s == 0:
            #     s = 1
            #     w = Counter()

            
            if label * dotProduct(new_w, review) < 1:
                s = (1 - (eta * l)) * s
                temp = eta*label/s
                increment(new_w, temp, review)
        
            t += 1

        ts2 = time.time()
        w = Counter()
        increment(w, s, new_w)

        # print(ts2 - ts1)
        
    per_error(review_counters, labels, w)

    return w


def main():

    reviews = pickle.load(open("training.p", "rb"))

    review_counters = []
    labels = []
    w = Counter()

    for word_list in reviews:
        
        temp = Counter()
        labels.append(word_list[- 1])

        del word_list[- 1]


        # for word in word_list:
        #     temp[word]+=1

        review_counters.append(Counter(word_list))

    l = 0.215443469003

    w = peg_improved(review_counters, labels, l)

    # epochs = 2
    # diff = 0

    # for i in range(epochs):

    #     ts1 = time.time()    

    #     for j in range(len(review_counters)):

    #         eta = 1/((j + 1) * l)

    #         peg_grad = grad(review_counters[j], labels[j], w, l)

    #         for key in peg_grad:
    #             w[key] = w[key] - eta * peg_grad[key]  

    #     ts2 = time.time()
    #     diff += ts2 - ts1

    
    # print(loss(review_counters, labels, w,l))

if __name__ == "__main__":
    main()
