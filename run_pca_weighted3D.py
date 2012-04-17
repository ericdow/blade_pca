import pylab, os, math, string
from numpy import *
from read_blades import *
from pca import *
from stats_tests import *
from scipy.interpolate import splprep, splev, sproot, splrep
from project_PCA import *

# where to save figures
figdir = 'plots/3d_pca_weighted/'

# R5 data for performing PCA
mpath = '/home/ericdow/code/blade_pca/R5/a-blades/'
npath = '/home/ericdow/code/blade_pca/R5/R5_ablade_averfoil_all.as'

# nominal blade
xyzn = read_coords(npath)
nsec = xyzn.shape[0]
npps = xyzn.shape[1]

# perform the PCA
w_1 = 2.*0.005
i_2 = 56
w_2 = 2.*0.005
h   = 0.0025
PCA1, PCA2, PCA3, max_error = calc_pca_weighted3D(mpath, npath, w_1, i_2, w_2, h)

# find max error over all blades and plot
n = max_error.shape[1]
max_error_le = (max_error[0,:,:,0]).max(0)
max_error_te = (max_error[1,:,:,0]).max(0)
max_error_bb = (max_error[2,:,:,0]).max(0)

# determine how many modes are needed
e_thresh = 5.0e-4
nmodes_le = nonzero((max_error_le - e_thresh) <= 0.0)[0][0]+1
nmodes_te = nonzero((max_error_te - e_thresh) <= 0.0)[0][0]+1
nmodes_bb = nonzero((max_error_bb - e_thresh) <= 0.0)[0][0]+1
print 'LE modes required: ', nmodes_le
print 'TE modes required: ', nmodes_te
print 'BB modes required: ', nmodes_bb
print 'Total modes required: ', nmodes_le+nmodes_te+nmodes_bb

pylab.figure()
pylab.semilogy(arange(n)+1,max_error_le,'*')
pylab.ylim([1.0e-5,2.0*max(max_error_le)])
pylab.grid(True)
pylab.title('Leading edge')
pylab.xlabel('Number of modes in reconstruction')
pylab.ylabel('Maximum error/chord')
pylab.savefig(figdir+'max_error_le.png')

pylab.figure()
pylab.semilogy(arange(n)+1,max_error_te,'*')
pylab.ylim([1.0e-5,2.0*max(max_error_te)])
pylab.grid(True)
pylab.title('Trailing edge')
pylab.xlabel('Number of modes in reconstruction')
pylab.ylabel('Maximum error/chord')
pylab.savefig(figdir+'max_error_te.png')

pylab.figure()
pylab.semilogy(arange(n)+1,max_error_bb,'*')
pylab.ylim([1.0e-5,2.0*max(max_error_bb)])
pylab.grid(True)
pylab.title('Blade body')
pylab.xlabel('Number of modes in reconstruction')
pylab.ylabel('Maximum error/chord')
pylab.savefig(figdir+'max_error_bb.png')

error

# plot some modes
modes_to_plot = [0,1,2]
modes_to_plot = []

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
S = PCA1[1]
V = PCA1[2]
pylab.figure()
pylab.semilogy(arange(S.shape[0]-1)+1,S[:-1]**2,'*')
pylab.grid(True)
pylab.xlabel('PCA Mode Index')
pylab.ylabel('PCA Eigenvalue')
pylab.savefig(figdir+'pca_eig_le.png')

pylab.figure()
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
pylab.savefig(figdir+'pca_scatter_le.png')

C = 2.0*(C - 0.5)
CT = (1./pi*arccos(1.0-2.0*(-abs(C)+1.0)) - 1.0)*0.5*(1.0-sign(C))\
   + 1./pi*arccos(1.0-2.0*abs(C))*0.5*(1.0+sign(C))
CT = 0.5*CT + 0.5
C  = 0.5*C  + 0.5
for im in modes_to_plot:
    # plot the raw modes
    pylab.figure()
    pylab.contourf(C,Zs.T,V[im,:,:],50)
    pylab.title('Mode '+str(im+1)+' (Original)')
    pylab.ylabel('Spanwise Coordinate')
    pylab.text(0.03,Zs[0,0]-0.05,'LE')
    pylab.text(0.5,Zs[0,0]-0.05,'TE')
    pylab.text(0.97,Zs[0,0]-0.05,'LE')
    pylab.colorbar()
    pylab.savefig(figdir+'mode'+str(im+1)+'_le.png')
    
    # plot the stretched modes
    pylab.figure()
    pylab.contourf(CT,Zs.T,V[im,:,:],50)
    pylab.title('Mode '+str(im+1)+' (Stretched)')
    pylab.ylabel('Spanwise Coordinate')
    pylab.text(0.03,Zs[0,0]-0.05,'LE')
    pylab.text(0.5,Zs[0,0]-0.05,'TE')
    pylab.text(0.97,Zs[0,0]-0.05,'LE')
    pylab.colorbar()
    pylab.savefig(figdir+'mode'+str(im+1)+'_stretch_le.png')

S = PCA2[1]
V = PCA2[2]
pylab.figure()
pylab.semilogy(arange(S.shape[0]-1)+1,S[:-1]**2,'*')
pylab.grid(True)
pylab.xlabel('PCA Mode Index')
pylab.ylabel('PCA Eigenvalue')
pylab.savefig(figdir+'pca_eig_te.png')

pylab.figure()
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
pylab.savefig(figdir+'pca_scatter_te.png')

C = 2.0*(C - 0.5)
CT = (1./pi*arccos(1.0-2.0*(-abs(C)+1.0)) - 1.0)*0.5*(1.0-sign(C))\
   + 1./pi*arccos(1.0-2.0*abs(C))*0.5*(1.0+sign(C))
CT = 0.5*CT + 0.5
C  = 0.5*C  + 0.5
# mode to plot
for im in modes_to_plot:
    # plot the raw modes
    pylab.figure()
    pylab.contourf(C,Zs.T,V[im,:,:],50)
    pylab.title('Mode '+str(im+1)+' (Original)')
    pylab.ylabel('Spanwise Coordinate')
    pylab.text(0.03,Zs[0,0]-0.05,'LE')
    pylab.text(0.5,Zs[0,0]-0.05,'TE')
    pylab.text(0.97,Zs[0,0]-0.05,'LE')
    pylab.colorbar()
    pylab.savefig(figdir+'mode'+str(im+1)+'_te.png')
    
    # plot the stretched modes
    pylab.figure()
    pylab.contourf(CT,Zs.T,V[im,:,:],50)
    pylab.title('Mode '+str(im+1)+' (Stretched)')
    pylab.ylabel('Spanwise Coordinate')
    pylab.text(0.03,Zs[0,0]-0.05,'LE')
    pylab.text(0.5,Zs[0,0]-0.05,'TE')
    pylab.text(0.97,Zs[0,0]-0.05,'LE')
    pylab.colorbar()
    pylab.savefig(figdir+'mode'+str(im+1)+'_stretch_te.png')

S = PCA3[1]
V = PCA3[2]
pylab.figure()
pylab.semilogy(arange(S.shape[0]-1)+1,S[:-1]**2,'*')
pylab.grid(True)
pylab.xlabel('PCA Mode Index')
pylab.ylabel('PCA Eigenvalue')
pylab.savefig(figdir+'pca_eig_body.png')

pylab.figure()
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
pylab.savefig(figdir+'pca_scatter_body.png')

C = 2.0*(C - 0.5)
CT = (1./pi*arccos(1.0-2.0*(-abs(C)+1.0)) - 1.0)*0.5*(1.0-sign(C))\
   + 1./pi*arccos(1.0-2.0*abs(C))*0.5*(1.0+sign(C))
CT = 0.5*CT + 0.5
C  = 0.5*C  + 0.5
# mode to plot
for im in modes_to_plot:
    # plot the raw modes
    pylab.figure()
    pylab.contourf(C,Zs.T,V[im,:,:],50)
    pylab.title('Mode '+str(im+1)+' (Original)')
    pylab.ylabel('Spanwise Coordinate')
    pylab.text(0.03,Zs[0,0]-0.05,'LE')
    pylab.text(0.5,Zs[0,0]-0.05,'TE')
    pylab.text(0.97,Zs[0,0]-0.05,'LE')
    pylab.colorbar()
    pylab.savefig(figdir+'mode'+str(im+1)+'_body.png')
    
    # plot the stretched modes
    pylab.figure()
    pylab.contourf(CT,Zs.T,V[im,:,:],50)
    pylab.title('Mode '+str(im+1)+' (Stretched)')
    pylab.ylabel('Spanwise Coordinate')
    pylab.text(0.03,Zs[0,0]-0.05,'LE')
    pylab.text(0.5,Zs[0,0]-0.05,'TE')
    pylab.text(0.97,Zs[0,0]-0.05,'LE')
    pylab.colorbar()
    pylab.savefig(figdir+'mode'+str(im+1)+'_stretch_body.png')
