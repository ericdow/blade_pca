import pylab, os, math, string
from numpy import *
from read_blades import *
from pca import *
from stats_tests import *
from scipy.interpolate import splprep, splev, sproot, splrep
from project_PCA import *

# where to save figures
figdir = 'plots/3d_pca/'

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

# read nominal blade surface
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

# compute the chord at the hub to normalize by
x0 = xyzn[0,:,:-1]
chord = sqrt((x0[0,0]-x0[(npps+1)/2,0])**2 + (x0[0,1]-x0[(npps+1)/2,1])**2)

xe = zeros((n,nsec,npps))
for i in range(n):
    nm, tau1m, tau2m = calcNormals3d(xyz[i,:,:,:])

    # calculate the error in the normal direction for this blade
    xe[i,:,:] = calcError(xyzn,nn,xyz[i,:,:,:],nm)

# average of the errors over all blades
xe_ave = mean(xe,axis=0)

# standard deviation of the errors over all blades
xe_std = std(xe,axis=0)

# z location of slices
zs = xyzn[:,0,2]
Zs = tile(zs,(npps,1))
# chordwise location of slices
C = zeros((nsec,npps))
for isec in arange(nsec):
    xyn = xyzn[isec,:,:]
    tck,sn = splprep([xyn[:,0],xyn[:,1]],s=0,per=1)
    C[isec,:] = sn

C = 2.0*(C - 0.5)
CT = (1./pi*arccos(1.0-2.0*(-abs(C)+1.0)) - 1.0)*0.5*(1.0-sign(C))\
   + 1./pi*arccos(1.0-2.0*abs(C))*0.5*(1.0+sign(C))
CT = 0.5*CT + 0.5
C  = 0.5*C  + 0.5

# plot the mean 
pylab.figure()
pylab.contourf(C,Zs.T,xe_ave,50)
pylab.ylabel('Spanwise Coordinate')
pylab.text(0.03,Zs[0,0]-0.05,'LE')
pylab.text(0.5,Zs[0,0]-0.05,'TE')
pylab.text(0.97,Zs[0,0]-0.05,'LE')
pylab.title('Mean Error Field')
pylab.colorbar()

pylab.figure()
pylab.contourf(CT,Zs.T,xe_ave,50)
pylab.ylabel('Spanwise Coordinate')
pylab.text(0.03,Zs[0,0]-0.05,'LE')
pylab.text(0.5,Zs[0,0]-0.05,'TE')
pylab.text(0.97,Zs[0,0]-0.05,'LE')
pylab.title('Mean Error Field (stretched)')
pylab.colorbar()

# plot the variance
pylab.figure()
pylab.contourf(C,Zs.T,xe_std,50)
pylab.ylabel('Spanwise Coordinate')
pylab.text(0.03,Zs[0,0]-0.05,'LE')
pylab.text(0.5,Zs[0,0]-0.05,'TE')
pylab.text(0.97,Zs[0,0]-0.05,'LE')
pylab.title('Standard Deviation Field')
pylab.colorbar()

pylab.figure()
pylab.contourf(CT,Zs.T,xe_std,50)
pylab.ylabel('Spanwise Coordinate')
pylab.text(0.03,Zs[0,0]-0.05,'LE')
pylab.text(0.5,Zs[0,0]-0.05,'TE')
pylab.text(0.97,Zs[0,0]-0.05,'LE')
pylab.title('Standard Deviation Field (stretched)')
pylab.colorbar()

pylab.show()
