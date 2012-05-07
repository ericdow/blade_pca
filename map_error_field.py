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

xyzn_tmp = reshape(xyzn,(nsec*npps,3))
th_n = mean(arctan2(xyzn_tmp[:,2],xyzn_tmp[:,1]))
x_n = mean(xyzn_tmp[:,0])

nt = xyz_mesh.shape[0]
ns = xyz_mesh.shape[1]

# interpolate the measured blades and write them to a file
xe_r37 = zeros((n,nt,ns))
xe_r5 = zeros((n,nsec,npps))

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
    xyzm[:,:,0] = reshape(xm,(nsec,npps))
    xyzm[:,:,1] = reshape(ym,(nsec,npps))
    xyzm[:,:,2] = reshape(zm,(nsec,npps))    

    # compute the error 
    nm, tau1m, tau2m = calcNormals3d(xyzm)
    # calculate the error in the normal direction for this blade
    xe = calcError(xyzn,nn,xyzm,nm)
    xe_r5[i,:,:] = xe

    V2 = transform_mode(xyzn,xe,xyz_mesh,ntip)
    xe_r37[i,:,:] = V2

    fname = 'blade'+str(i+1)+'.dat'
    f = open(data_dir+fname,'w')
    f.write('CDIM:       %d\n' % V2.shape[1])
    f.write('SDIM:       %d\n' % V2.shape[0])
    for i in arange(V2.shape[1]):
        for j in arange(V2.shape[0]):
            f.write('%e\n' % V2[j,i])
    f.close()

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
    Vp[i,:-1,:] = xe_r5[i,:,:].T
Vp[:,-1,:] = Vp[:,0,:]
write_tecplot.write_blade_surf(xnom,ynom,znom,Vp,'errors_r5.dat')
xmesh = xyz_mesh[:,:,0]
ymesh = xyz_mesh[:,:,1]
zmesh = xyz_mesh[:,:,2]
write_tecplot.write_blade_surf(xmesh,ymesh,zmesh,xe_r37,'errors_r37.dat')
