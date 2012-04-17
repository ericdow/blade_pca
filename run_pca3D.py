import pylab, os, math, string
from numpy import *
from read_blades import *
from pca import *
from stats_tests import *
from scipy.interpolate import splprep, splev, sproot, splrep
from project_PCA import *
import write_tecplot

# where to save figures
figdir = 'plots/3d_pca/'

# R5 data for performing PCA
mpath = '/home/ericdow/code/blade_pca/R5/a-blades/'
npath = '/home/ericdow/code/blade_pca/R5/R5_ablade_averfoil_all.as'

xyzn = read_coords(npath)
nsec = xyzn.shape[0]
npps = xyzn.shape[1]

# perform the PCA
U, S, V, V0, Z, max_error = calc_pca3D(mpath, npath)
n = max_error.shape[0]

xnom = zeros((npps+1,nsec))
ynom = zeros((npps+1,nsec))
znom = zeros((npps+1,nsec))
xnom[:-1,:] = xyzn[:,:,0].T
ynom[:-1,:] = xyzn[:,:,1].T
znom[:-1,:] = xyzn[:,:,2].T
xnom[-1,:] = xnom[0,:]
ynom[-1,:] = ynom[0,:]
znom[-1,:] = znom[0,:]
Vp = zeros((n,npps+1,nsec))
for i in range(n):
    Vp[i,:-1,:] = V[i,:,:].T
Vp[:,-1,:] = Vp[:,0,:]
write_tecplot.write_blade_surf(xnom,ynom,znom,Vp,'pca_modes_r5.dat')

'''
n = max_error.shape[0]
# find max error over all blades and plot
max_error = (max_error[:,:,0]).max(0)

# determine how many modes are needed
e_thresh = 5.0e-4
nmodes = nonzero((max_error - e_thresh) <= 0.0)[0][0]+1
print 'Total modes required: ', nmodes

pylab.semilogy(arange(max_error.shape[0])+1,max_error,'*')
pylab.ylim([1.0e-5,2.0*max(max_error)])
pylab.grid(True)
pylab.xlabel('Number of modes in reconstruction')
pylab.ylabel('Maximum error/chord')
pylab.savefig(figdir+'max_error.png')

# test if the modes are normally distributed
for imode in range(Z.shape[1]):
    ad_pass, sig, ws_pass = norm_test(Z[:,imode],0,1)
    print 'Mode '+str(imode+1)+':' 
    print ad_pass
    print ws_pass
print 'Levels for Anderson-Darling Test:'
print sig

# make a Q-Q plot of Z data
for imode in range(4):
    qq_plot(Z[:,imode],0.0,1.0,'Mode '+str(imode+1))
    pylab.savefig(figdir+'qq_mode'+str(imode+1)+'.png')
 
# z location of slices
zs = xyzn[:,0,2]
Zs = tile(zs,(npps,1))
# chordwise location of slices
C = zeros((nsec,npps))
for isec in arange(nsec):
    xyn = xyzn[isec,:,:]
    tck,sn = splprep([xyn[:,0],xyn[:,1]],s=0,per=1)
    C[isec,:] = sn

# plot the scatter fraction 
pylab.figure()
pylab.semilogy(arange(S.shape[0]-1)+1,S[:-1]**2,'*')
pylab.grid(True)
pylab.xlabel('PCA Mode Index')
pylab.ylabel('PCA Eigenvalue')
pylab.savefig(figdir+'pca_eig.png')

pylab.figure()
# pylab.plot(arange(S.shape[0]),0.99*ones((S.shape[0])),'r--')
sf = cumsum(S**2)/sum(S**2)
nn = 10
ind = arange(nn)+1
width = 0.35
pylab.bar(ind,sf[:nn],width)
pylab.xlim((0,nn+1))
for i in range(nn):
    pylab.text(ind[i]-1.8*width,0.95*sf[i],'%1.2f' % sf[i])
pylab.xlabel('PCA Mode Index')
pylab.ylabel('Scatter Fraction')
pylab.savefig(figdir+'pca_scatter.png')

C = 2.0*(C - 0.5)
CT = (1./pi*arccos(1.0-2.0*(-abs(C)+1.0)) - 1.0)*0.5*(1.0-sign(C))\
   + 1./pi*arccos(1.0-2.0*abs(C))*0.5*(1.0+sign(C))
CT = 0.5*CT + 0.5
C  = 0.5*C  + 0.5
# mode to plot
for im in [0,1,2,3]:
    # plot the raw modes
    pylab.figure()
    pylab.contourf(C,Zs.T,V[im,:,:],50)
    pylab.title('Mode '+str(im+1)+' (Original)')
    pylab.ylabel('Spanwise Coordinate')
    pylab.text(0.03,Zs[0,0]-0.05,'LE')
    pylab.text(0.5,Zs[0,0]-0.05,'TE')
    pylab.text(0.97,Zs[0,0]-0.05,'LE')
    pylab.colorbar()
    pylab.savefig(figdir+'mode'+str(im+1)+'.png')
    
    # plot the stretched modes
    pylab.figure()
    pylab.contourf(CT,Zs.T,V[im,:,:],50)
    pylab.title('Mode '+str(im+1)+' (Stretched)')
    pylab.ylabel('Spanwise Coordinate')
    pylab.text(0.03,Zs[0,0]-0.05,'LE')
    pylab.text(0.5,Zs[0,0]-0.05,'TE')
    pylab.text(0.97,Zs[0,0]-0.05,'LE')
    pylab.colorbar()
    pylab.savefig(figdir+'mode'+str(im+1)+'_stretch.png')
'''
