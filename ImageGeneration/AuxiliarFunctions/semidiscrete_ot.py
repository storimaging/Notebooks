
# Copyright Arthur Leclaire (c), 2019.

import numpy as np

def asgd(sample, y, nu, niter, C):
    #  Estimate Semi discrete optimal transport map
    #   by Average stochastic gradient descent.
    #
    # v = asgd(sample,y,nu,niter,C)
    #
    # Input:
    # - sample: function to compute a sample of source distribution
    # - y: set of Dirac locations of target distribution
    # - nu: masses of Diracs
    # - niter: number of iterations
    # - C: parameter (gradient step)
    #
    # Ouput:
    # - v: final potential values
    #
    # NB: 
    #   - we use c(x,y) = |x-y|^2 as cost function
    
    J = y.shape[0]
    v = np.zeros(J)
    vt = np.zeros(J)
    for k in range(1, niter+1):
        X = sample()
        X = X[np.newaxis, 0]
        r = np.sum((np.tile(X, (J, 1))-y)**2, axis=1)-vt
        irmin = np.argmin(r)
        grad = np.copy(nu)
        grad[irmin] = grad[irmin]-1
        vt = vt + np.double(C)/np.sqrt(k)*grad
        v = 1/k*vt + (k-1)/k*v
    return v


def map(x, y, v):
    # Compute a conditional sample of the optimal transport plan given x .
    #
    # Y,j = sample_transport_plan( x,y,v )
    #
    # Input:
    # - x: sample of source distribution
    # - y: set of Dirac locations of target distribution
    # - v: potentials defining the optimal transport plan
    #
    # Output:
    # - Y: transformed samples
    # - j: corresponding indices (chosen samples in y)
    #
    # NB: we use c(x,y) = |x-y|^2 as cost function
    
    vt = v - np.sum(y**2, axis=1)
#    print(x.shape)
#    print(y.T.shape)
#    print(vt.shape)
    r = -2 * x @ y.T - vt
    j = np.argmin(r, axis=1)
    Y = y[j, :]
    return Y, j
