import pylab, os, math, string
from numpy import *
from read_blades import *
from pca import *
from stats_tests import *
from scipy.interpolate import splprep, splev, sproot, splrep
import write_tecplot

# directory for the ob modes
ob_dir = 'ob_modes_combined/'
nob = 33

# directory for the measured data
meas_dir = 'measured_blades/'
nb = 34

# mesh surface file for interpolating modes to
mesh_path = '/home/ericdow/code/blade_pca/blade_surf.dat'
# number of points in span that are in tip clearance
ntip = 21
# read the surface mesh
xyz_mesh = read_mesh_surf(mesh_path)
s,t = xyz2st(xyz_mesh[:,:,0].T,xyz_mesh[:,:,1].T,xyz_mesh[:,:,2].T)
s /= s[-1]
t /= t[-1]

# read the output based modes
cdim,sdim,V = read_mode(ob_dir+'V1.dat')
V_ob = zeros((nob,cdim*sdim))
for i in range(nob):
    V_ob[i,:] = reshape(read_mode(ob_dir+'V'+str(i+1)+'.dat')[2], cdim*sdim)

# read the measured errors
es = zeros((nb,cdim*sdim))
for i in range(nb):
    es[i,:] = reshape(read_mode(meas_dir+'blade'+str(i+1)+'.dat')[2], cdim*sdim)

# form the projection matrix
A = zeros((nob,nob))
for i in range(nob):
    for j in range(i+1):
        A[i,j] = dot(V_ob[i,:],V_ob[j,:])
        A[j,i] = dot(V_ob[i,:],V_ob[j,:])

W = zeros((nb,nob))
for i in range(nb):
    b = dot(V_ob,es[i,:])
    W[i,:] = linalg.solve(A,b)

# write output based amplitudes to a file
f = open('W.dat','w')
for i in range(nb):
    for j in range(nob):
        f.write('%e\t' % W[i,j])
    f.write('\n')

f.close()

toplot = 20
img = pylab.contourf(s,t,reshape(es[toplot,:],(cdim,sdim)).T,50)
clim = img.get_clim()
pylab.colorbar()
pylab.title('Measured Error')
pylab.xlabel('Chordwise location')
pylab.ylabel('Spanwise location')
pylab.text(0.03,t[0]-0.07,'LE')
pylab.text(0.5,t[0]-0.07,'TE')
pylab.text(0.97,t[0]-0.07,'LE')
pylab.text(0.25,t[0]-0.07,'PS')
pylab.text(0.75,t[0]-0.07,'SS')
pylab.ylim([t[0]-0.1,t[-1]])

pylab.figure()
img = pylab.contourf(s,t,reshape(dot(V_ob.T,W[toplot,:]),(cdim,sdim)).T,50)
img.set_clim(clim)
pylab.colorbar()
pylab.title('Projected Error')
pylab.xlabel('Chordwise location')
pylab.ylabel('Spanwise location')
pylab.text(0.03,t[0]-0.07,'LE')
pylab.text(0.5,t[0]-0.07,'TE')
pylab.text(0.97,t[0]-0.07,'LE')
pylab.text(0.25,t[0]-0.07,'PS')
pylab.text(0.75,t[0]-0.07,'SS')
pylab.ylim([t[0]-0.1,t[-1]])

pylab.show()
