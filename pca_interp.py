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

# directory for PCA modes, amplitudes
data_dir = 'pca/'

# R5 data for performing PCA
mpath = '/home/ericdow/code/blade_pca/R5/a-blades/'
npath = '/home/ericdow/code/blade_pca/R5/R5_ablade_averfoil_all.as'

# mesh surface file for interpolating modes to
mesh_path = '/home/ericdow/code/blade_pca/blade_surf.dat'
# number of points in span that are in tip clearance
ntip = 21

# read the surface mesh
xyz_mesh = read_mesh_surf(mesh_path)

xyzn = read_coords(npath)
nsec = xyzn.shape[0]
npps = xyzn.shape[1]

# perform the PCA
U, S, V, V0, Z = calc_pca3D(mpath, npath)

# test if the modes are normally distributed
'''
for imode in range(Z.shape[1]):
    ad_pass, sig, ws_pass = norm_test(Z[:,imode],0,1)
    print 'Mode '+str(imode+1)+':' 
    print ad_pass
    print ws_pass
print 'Levels for Anderson-Darling Test:'
print sig
'''

# make a Q-Q plot of Z data
'''
for imode in range(4):
    qq_plot(Z[:,imode],0.0,1.0,'Mode '+str(imode+1))
pylab.show()
'''

# write out PCA singular values
'''
f = open(data_dir+'S.dat','w')
for s in S:
    f.write('%e\n' % s)
f.close()
'''

# write out the first pca mode to tecplot format
xmesh = xyz_mesh[:,:,0]
ymesh = xyz_mesh[:,:,1]
zmesh = xyz_mesh[:,:,2]
V2 = transform_mode(xyzn,V[0,:,:],xyz_mesh,ntip)
write_tecplot.write_blade_surf(xmesh,ymesh,zmesh,V2,'pca_mode1_r37.dat')

xnom = zeros((npps+1,nsec))
ynom = zeros((npps+1,nsec))
znom = zeros((npps+1,nsec))
xnom[:-1,:] = xyzn[:,:,0].T
ynom[:-1,:] = xyzn[:,:,1].T
znom[:-1,:] = xyzn[:,:,2].T
xnom[-1,:] = xnom[0,:]
ynom[-1,:] = ynom[0,:]
znom[-1,:] = znom[0,:]
Vp = zeros((npps+1,nsec))
Vp[:-1,:] = V[0,:,:].T
Vp[-1,:] = Vp[0,:]
write_tecplot.write_blade_surf(xnom,ynom,znom,Vp,'pca_mode1_meas.dat')

# interpolate the modes and write them to a file
'''
for i in range(V.shape[0]):
    V2 = transform_mode(xyzn,V[i,:,:],xyz_mesh,ntip)
    fname = 'V'+str(i+1)+'.dat'
    f = open(data_dir+fname,'w')
    f.write('CDIM:       %d\n' % V2.shape[1])
    f.write('SDIM:       %d\n' % V2.shape[0])
    for i in arange(V2.shape[1]):
        for j in arange(V2.shape[0]):
            f.write('%e\n' % V2[j,i])
    f.close()
'''

'''
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
