import pylab, os, math, string
from numpy import *
from read_blades import *
from pca import *
from stats_tests import *
from scipy.interpolate import splprep, splev, sproot, splrep
from project_PCA import *

# directory for PCA modes, amplitudes
data_dir = 'pca/'

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
# spatial dimensions
m = 2

# read the nominal blade
xyzn = read_coords(npath)
nn, tau1n, tau2n = calcNormals3d(xyzn)

# center the blades
xyzn_tmp = reshape(xyzn,(nsec*npps,3))
th_n = mean(arctan2(xyzn_tmp[:,2],xyzn_tmp[:,1]))
x_n = mean(xyzn_tmp[:,0])
for i in range(n):

    # rotate the measured blades so the average angle agrees
    # with the nominal blade
    xyzm = copy(xyz[i,:,:,:])
    xm = reshape(xyzm[:,:,0],(nsec*npps))
    ym = reshape(xyzm[:,:,1],(nsec*npps))
    zm = reshape(xyzm[:,:,2],(nsec*npps))
    th_m = arctan2(zm,ym)
    r_m = sqrt(ym**2 + zm**2)
    dth = th_n - mean(th_m)
    ym = r_m*cos(th_m+dth)
    zm = r_m*sin(th_m+dth)
    xm = xm - mean(xm) + x_n
    xyz[i,:,:,0] = reshape(xm,(nsec,npps))
    xyz[i,:,:,1] = reshape(ym,(nsec,npps))
    xyz[i,:,:,2] = reshape(zm,(nsec,npps))    

xe = zeros((n,nsec,npps))
for i in range(n):
    nm, tau1m, tau2m = calcNormals3d(xyz[i,:,:,:])

    # calculate the error in the normal direction for this blade
    xe[i,:,:] = calcError(xyzn,nn,xyz[i,:,:,:],nm)

# ensemble average of errors
xa = mean(xe,axis=0)

# plot the error
# z location of slices
zs = xyzn[:,0,2]
Zs = tile(zs,(npps,1))
# chordwise location of slices
C = zeros((nsec,npps))
for isec in arange(nsec):
    xyn = xyzn[isec,:,:]
    tck,sn = splprep([xyn[:,0],xyn[:,1]],s=0,per=1)
    C[isec,:] = sn

for iblade in range(10):
    pylab.figure()
    pylab.contourf(C,Zs.T,xe[iblade,:,:],50)
    pylab.ylabel('Spanwise Coordinate')
    pylab.text(0.03,Zs[0,0]-0.05,'LE')
    pylab.text(0.5,Zs[0,0]-0.05,'TE')
    pylab.text(0.97,Zs[0,0]-0.05,'LE')
    pylab.colorbar()

pylab.show()    
