import pylab, os, math, string
from numpy import *
from read_blades import *
from pca import *
from stats_tests import *
from scipy.interpolate import splprep, splev, sproot, splrep
from project_PCA import *
import write_tecplot

# R37 data for performing PCA
mpath = '/home/ericdow/code/blade_pca/measured_blades/'

# directory for PCA modes, amplitudes
data_dir = 'pca_modes/'

# perform the PCA
U, S, V, V0, Z = calc_pca3D_meas(mpath)

# write out the pca modes and amplitudes
f = open(data_dir+'S.dat','w')
for s in S:
    f.write('%e\n' % s)
f.close()

for n in range(V.shape[0]):
    fname = 'V'+str(n+1)+'.dat'
    f = open(data_dir+fname,'w')
    f.write('CDIM:       %d\n' % V[n,:,:].shape[1])
    f.write('SDIM:       %d\n' % V[n,:,:].shape[0])
    for i in arange(V[n,:,:].shape[1]):
        for j in arange(V[n,:,:].shape[0]):
            f.write('%e\n' % V[n,j,i])
    f.close()

# write out the Z values
for n in range(Z.shape[0]):
    fname = 'Z_blade'+str(n+1)+'.dat'
    f = open(data_dir+fname,'w')
    for i in arange(Z.shape[1]):
        f.write('%e\n' % Z[n,i])
    f.close()

# mesh surface file for interpolating modes to
mesh_path = '/home/ericdow/code/blade_pca/blade_surf.dat'

# read the surface mesh
xyz_mesh = read_mesh_surf(mesh_path)

xmesh = xyz_mesh[:,:,0]
ymesh = xyz_mesh[:,:,1]
zmesh = xyz_mesh[:,:,2]
write_tecplot.write_blade_surf(xmesh,ymesh,zmesh,V,'pca_modes_r37.dat')

 
'''
# plot the scatter fraction 
pylab.figure()
pylab.semilogy(arange(S.shape[0]-1)+1,S[:-1]**2,'*')
pylab.grid(True)
pylab.xlabel('PCA Mode Index')
pylab.ylabel('PCA Eigenvalue')

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

pylab.show()
'''
