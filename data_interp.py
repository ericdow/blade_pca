import pylab, os, math, string
from numpy import *
from read_blades import *
from pca import *
from stats_tests import *
from scipy.interpolate import splprep, splev, sproot, splrep
from project_PCA import *
import write_tecplot

# directory for measured blade files
data_dir = 'measured_blades/'

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
nn, tau1n, tau2n = calcNormals3d(xyzn)
nsec = xyzn.shape[0]
npps = xyzn.shape[1]

# read measured blades
xyz = read_all(mpath)
# number of blades 
n = xyz.shape[0]

# interpolate the measured blades and write them to a file
for i in range(n):

    # compute the error 
    nm, tau1m, tau2m = calcNormals3d(xyz[i,:,:,:])
    # calculate the error in the normal direction for this blade
    xe = calcError(xyzn,nn,xyz[i,:,:,:],nm)

    V2 = transform_mode(xyzn,xe,xyz_mesh,ntip)
    fname = 'blade'+str(i+1)+'.dat'
    f = open(data_dir+fname,'w')
    f.write('CDIM:       %d\n' % V2.shape[1])
    f.write('SDIM:       %d\n' % V2.shape[0])
    for i in arange(V2.shape[1]):
        for j in arange(V2.shape[0]):
            f.write('%e\n' % V2[j,i])
    f.close()

