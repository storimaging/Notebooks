import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm, kurtosis
from scipy.special import gamma, gammainc, gammaincinv, psi
from statsmodels.graphics.gofplots import qqplot_2samples

# 1.2 Comparision of quantiles

def CompareQuantiles(x,y, p_values, xlabel, ylabel):
    quantile_x = np.quantile(x, p_values)
    quantile_y = np.quantile(y, p_values)
    qqplot_2samples(quantile_x, quantile_y, xlabel, ylabel)
    plt.show()

# 2.1 Estimation by the method of moments

def equation2(alpha):
  return (gamma(1/alpha)*gamma(5/alpha)/gamma(3/alpha)**2)

def findAlpha(k, k_hat, alphas):
    array_k = np.asarray(k)
    idx = (np.abs(array_k - k_hat)).argmin()
    return(alphas[idx])

def equation1(sigma, alpha):
  return gamma(3/alpha)/(sigma**2*gamma(1/alpha))

def distribution_f(x, alpha, eta):
    C = (alpha*eta/2)*(1/gamma(1/alpha))
    return C * np.exp(- (np.abs(eta*x))**alpha)

# 2.2 Estimation by maximum likelihood method

def LogLikelihood(X, alpha, eta):
  
    n = len(X)
    C = (alpha*eta/2)*(1/gamma(1/alpha))
    aux = np.sum(np.abs(eta*X)**alpha)
    return -n * np.log(C) + aux


def update_parameters(X, alpha, eta, n, coef = 1e-8):

    der_respect_alpha = - n/alpha - (n/(alpha**2))*psi(1/alpha) + np.sum( np.log(np.abs(eta*X)) * np.abs(eta*X)**alpha)  
    alpha = alpha - coef * der_respect_alpha
    eta = (np.size(X)/(np.sum(alpha *np.abs(X)**alpha )))**(1/alpha )

    return alpha, eta
    

def Estimation_MaxLikelihood(X, epsilon):
    n = len(X)
    alpha, eta = 0.8, 200

    LL = LogLikelihood(X, alpha, eta)
    LL_old = LL

    while (True):

        alpha, eta = update_parameters(X, alpha, eta, n)
        LL = LogLikelihood(X, alpha, eta)

        if np.abs(LL -LL_old) < epsilon:
           break
        else:
           LL_old = LL
    
    return alpha, eta