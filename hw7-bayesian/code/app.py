import scipy
from scipy.optimize import minimize
from scipy.misc import comb
from scipy.special import beta,betaln
import numpy as np


num_clicks = [50,160,180,0,0,1]
num_impressions = [10000,20000,60000,100,5,2]

def optimize_beta(prior):

    res = 0
    
    for i in range(len(num_clicks)):

        res = res + (betaln(num_clicks[i] + prior[0], num_impressions[i] - num_clicks[i] + prior[1]) - betaln(prior[0],prior[1]))

    res = res * (-1)

    return res


prior = [10,1200]
print(minimize(fun = optimize_beta, x0 = prior, method = 'Nelder-Mead'))
