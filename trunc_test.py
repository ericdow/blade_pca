import pylab, os, math, string
from numpy import *
from read_blades import *
from pca import *
from stats_tests import *
from scipy.interpolate import splprep, splev, sproot, splrep
from project_PCA import *

# R5 data for performing PCA
mpath = '/home/ericdow/code/blade_pca/R5/a-blades/'
npath = '/home/ericdow/code/blade_pca/R5/R5_ablade_averfoil_all.as'

# read measured blades
xyz = read_all(mpath)
# number of blades 
n = xyz.shape[0]
# number of sections
nsec = xyz.shape[1] # don't use last section
# number of points per section
npps = xyz.shape[2]
# number of sections x number of points per section
p = nsec*npps

# read the nominal blade
xyzn = read_coords(npath)
nn, tau1n, tau2n = calcNormals3d(xyzn)
nsec = xyzn.shape[0]
npps = xyzn.shape[1]

# perform the PCA
xe = zeros((n,nsec,npps))
for i in range(n):
    nm, tau1m, tau2m = calcNormals3d(xyz[i,:,:,:])

    # calculate the error in the normal direction for this blade
    xe[i,:,:] = calcError(xyzn,nn,xyz[i,:,:,:],nm)

# ensemble average of errors
xa = mean(xe,axis=0)

# centered
x  = zeros((n,nsec,npps))
for i in range(n):
    x[i,:,:] = xe[i,:,:] - xa # [n x nsec x npps]

# insert x into correct spot in X matrix
for isec in arange(nsec):
    if isec == 0:
        X = x[:,isec,:]
    else:
        X = hstack((X,x[:,isec,:]))

# compute SVD of X
U, S, V0 = linalg.svd(X,full_matrices=False)

# normalize the singular values
# NOTE : need eigenvalues of covariance matrix, which are sqrt(sigma^2/n)
S /= sqrt(n)

# reorganize V to give x, y components of each slice
# in this case, V gives the error in the normal direction
V = zeros((n,nsec,npps))
for isec in arange(nsec):
    V[:,isec,:] = V0[:,isec*npps:(isec+1)*npps]

# project samples onto modes to reconstruct measured geometry
# Z[i,:] are the components for the ith measured blade
Z = zeros((n,n))
for i in range(n):
    for j in range(n):
        Z[i,j] = dot(V0[j,:],X[i,:]) / S[j]

# test reconstruction vs the number of PCA modes
iblade = 0
xe_rec = zeros((n,nsec,npps))
for i in range(n):
    xe_rec[i,:,:] = xa
    for ii in range(i+1):
        xe_rec[i,:,:] += S[ii]*Z[iblade,ii]*V[ii,:,:]

isec = 5
ipps = 30
pylab.plot(arange(n)+1,abs(xe[iblade,isec,ipps]-xe_rec[:,isec,ipps])/abs(xe[iblade,isec,ipps]),'*')
pylab.show()
