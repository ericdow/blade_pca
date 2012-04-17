from numpy import *
from scipy import special
from scipy import stats
import pylab

def norm_test(X,mu,sigma):
    # perform the Wilk-Shapiro and Anderson-Darling tests
    # inputs
    # X     : array of sample data
    # mu    : mean of assumed distribution
    # sigma : stand dev of assumed distribution
    # outputs

    Xs = (X - mu) / sigma

    # Anderson-Darling
    A2, crit, sig = stats.anderson(X,dist='norm')
    ad_pass = (A2 < crit)

    # Wilks-Shapiro
    W, p = stats.shapiro(X)
    ws_pass = (W > p)

    return ad_pass, sig, ws_pass

def qq_plot(X,mu,sigma,title):
    # create a Q-Q plot for the data in X
    # inputs
    # X     : array of sample data
    # mu    : mean of assumed distribution
    # sigma : stand dev of assumed distribution
    # outputs

    # shift data to be standard normal
    Xs = (X - mu) / sigma

    # calculate order statistic medians and ordered responses
    res = stats.probplot(Xs)
    osm = res[0][0]
    osr = res[0][1]

    # plot
    delta = max(osm)-min(osm)
    line = array((min(osm)-0.1*delta,max(osm)+0.1*delta))
    pylab.figure()
    pylab.plot(osm,osr,'*')
    pylab.plot(line,line)
    pylab.xlabel('Normal theoretical quantiles')
    pylab.ylabel('Data quantiles')
    pylab.grid(True)
    pylab.title(title)

'''
mu = 0.0
sigma = 1.0
nsamples = 100
X = random.normal(mu,sigma,(nsamples))

ad_pass, sig, qs_pass = norm_test(X,mu,sigma)
qq_plot(X,mu,sigma)
'''
