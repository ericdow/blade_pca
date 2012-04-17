import pylab, os, math, string
from numpy import *
from read_blades import *
from pca import *
from stats_tests import *
from scipy.interpolate import splprep, splev, sproot, splrep

def transform_mode(xyz1,V1,xyz2,ntip):
    # interpolate V1 from xyz1 blade to xyz2
    # inputs
    # xyz1 : coordinates of blade for which PCA is calculated
    # V1   : components of the PCA mode [nsec1 x npps1]
    # xyz2 : coordinates of blade to transform to (from mesh)
    # output
    # V2_mesh : components of PCA mode interpolated on mesh

    V2 = zeros((xyz2.shape[0],xyz2.shape[1]))
    
    # make r37 points go in same orientation
    # remove the tip clearance part
    nspan = xyz2.shape[0]-ntip+1
    xyz2 = xyz2[:nspan,:,:]
    npps = xyz2.shape[1]
    xyz_ss = xyz2[:,:(npps+1)/2,:]
    xyz_ps = xyz2[:,(npps+1)/2:-1,:]
    xyz2 = zeros((nspan,npps-1,3))
    xyz2[:,:,0] = hstack((xyz_ps[:,:,0],xyz_ss[:,:,0]))
    xyz2[:,:,1] = hstack((xyz_ps[:,:,1],xyz_ss[:,:,1]))
    xyz2[:,:,2] = hstack((xyz_ps[:,:,2],xyz_ss[:,:,2]))

    nsec1 = xyz1.shape[0]
    npps1 = xyz1.shape[1]
    nsec2 = xyz2.shape[0]
    npps2 = xyz2.shape[1]
    
    # pressure side
    xyz1_ps = xyz1[:,:npps1/2+1,:]
    xyz2_ps = xyz2[:,:npps2/2+1,:]
    V2_ps = zeros((nsec2,npps2/2+1))
 
    # spline sections of PCA geometry
    u1 = zeros((nsec1,npps1/2+1))
    for isec in range(nsec1):
        tck_tmp,u1[isec,:] = splprep([xyz1_ps[isec,:,0],xyz1_ps[isec,:,1],\
                                      xyz1_ps[isec,:,2]],s=0,per=0)
        if isec == 0:
            tck1 = tck_tmp
        else:
            tck1 = tck1+tck_tmp

    # spline sections of mesh geometry
    u2 = zeros((nsec2,npps2/2+1))
    for isec in range(nsec2):
        tck_tmp,u2[isec,:] = splprep([xyz2_ps[isec,:,0],xyz2_ps[isec,:,1],\
                                      xyz2_ps[isec,:,2]],s=0,per=0)
        if isec == 0:
            tck2 = tck_tmp
        else:
            tck2 = tck2+tck_tmp

    # s coordinates of mesh hub 
    s_hub = u2[0,:]

    # interpolate mode on pressure side
    V1_tmp = zeros((nsec2,npps2/2+1)) 
    for ichord2 in range(npps2/2+1):
        xyz1_interp = zeros((nsec1,3))
        V1_interp = zeros((nsec1)) 
        for isec1 in range(nsec1):
            # interpolate mode
            tck = splrep(u1[isec1,:],V1[isec1,:npps1/2+1],s=0)
            V1_interp[isec1] = splev(s_hub[ichord2],tck)
            # interpolate coordinate
            tck,u = splprep([xyz1_ps[isec1,:,0],xyz1_ps[isec1,:,1],\
                             xyz1_ps[isec1,:,2]],s=0,per=0)
            xyz1_interp[isec1,0],xyz1_interp[isec1,1],\
            xyz1_interp[isec1,2] = splev(s_hub[ichord2],tck)
        # calculate t coordinate of PCA geom
        tck,t1 = splprep([xyz1_interp[:,0],xyz1_interp[:,1],xyz1_interp[:,2]],s=0,per=0)
        # calculate t coordinate of mesh geom
        xyz2_interp = zeros((nsec2,3))
        for isec2 in range(nsec2):
            # interpolate coordinate
            tck,u = splprep([xyz2_ps[isec2,:,0],xyz2_ps[isec2,:,1],\
                             xyz2_ps[isec2,:,2]],s=0,per=0)
            xyz2_interp[isec2,0],xyz2_interp[isec2,1],\
            xyz2_interp[isec2,2] = splev(s_hub[ichord2],tck)
        tck,t2 = splprep([xyz2_interp[:,0],xyz2_interp[:,1],xyz2_interp[:,2]],s=0,per=0)
        # interpolate mode
        tck = splrep(t1,V1_interp)
        V1_tmp[:,ichord2] = splev(t2,tck)
    # interpolate on each spanwise section of the mesh
    for isec2 in range(nsec2):
        tck = splrep(s_hub,V1_tmp[isec2,:],s=0)
        V2_ps[isec2,:] = splev(u2[isec2,:],tck)

    '''
    pylab.figure()    
    pylab.contourf(u1[0,:],t1,V1[:,:npps1/2+1],50)
    pylab.colorbar()
    pylab.figure()
    pylab.contourf(u2[0,:],t2,V2[:,:npps2/2+1],50)
    pylab.colorbar()
    pylab.show()
    '''

    # suction side
    xyz1_ss = xyz1[:,npps1/2-1:,:]
    xyz2_ss = xyz2[:,npps2/2-1:,:]
    V2_ss = zeros((nsec2,npps/2+1))

    # spline sections of PCA geometry
    u1 = zeros((nsec1,npps1/2+1))
    for isec in range(nsec1):
        tck_tmp,u1[isec,:] = splprep([xyz1_ss[isec,:,0],xyz1_ss[isec,:,1],\
                                      xyz1_ss[isec,:,2]],s=0,per=0)
        if isec == 0:
            tck1 = tck_tmp
        else:
            tck1 = tck1+tck_tmp

    # spline sections of mesh geometry
    u2 = zeros((nsec2,npps2/2+1))
    for isec in range(nsec2):
        tck_tmp,u2[isec,:] = splprep([xyz2_ss[isec,:,0],xyz2_ss[isec,:,1],\
                                      xyz2_ss[isec,:,2]],s=0,per=0)
        if isec == 0:
            tck2 = tck_tmp
        else:
            tck2 = tck2+tck_tmp

    # s coordinates of mesh hub 
    s_hub = u2[0,:]

    # interpolate mode on pressure side
    V1_tmp = zeros((nsec2,npps2/2+1)) 
    for ichord2 in range(npps2/2+1):
        xyz1_interp = zeros((nsec1,3))
        V1_interp = zeros((nsec1)) 
        for isec1 in range(nsec1):
            # interpolate mode
            tck = splrep(u1[isec1,:],V1[isec1,npps1/2-1:],s=0)
            V1_interp[isec1] = splev(s_hub[ichord2],tck)
            # interpolate coordinate
            tck,u = splprep([xyz1_ss[isec1,:,0],xyz1_ss[isec1,:,1],\
                             xyz1_ss[isec1,:,2]],s=0,per=0)
            xyz1_interp[isec1,0],xyz1_interp[isec1,1],\
            xyz1_interp[isec1,2] = splev(s_hub[ichord2],tck)
        # calculate t coordinate of PCA geom
        tck,t1 = splprep([xyz1_interp[:,0],xyz1_interp[:,1],xyz1_interp[:,2]],s=0,per=0)
        # calculate t coordinate of mesh geom
        xyz2_interp = zeros((nsec2,3))
        for isec2 in range(nsec2):
            # interpolate coordinate
            tck,u = splprep([xyz2_ss[isec2,:,0],xyz2_ss[isec2,:,1],\
                             xyz2_ss[isec2,:,2]],s=0,per=0)
            xyz2_interp[isec2,0],xyz2_interp[isec2,1],\
            xyz2_interp[isec2,2] = splev(s_hub[ichord2],tck)
        tck,t2 = splprep([xyz2_interp[:,0],xyz2_interp[:,1],xyz2_interp[:,2]],s=0,per=0)
        # interpolate mode
        tck = splrep(t1,V1_interp)
        V1_tmp[:,ichord2] = splev(t2,tck)
    # interpolate on each spanwise section of the mesh
    for isec2 in range(nsec2):
        tck = splrep(s_hub,V1_tmp[isec2,:],s=0)
        V2_ss[isec2,:] = splev(u2[isec2,:],tck)

    '''
    pylab.figure()
    # pylab.contourf(u1[0,:],t1,V1[:,npps1/2-1:],50)
    pylab.contourf(u1[0,:],t1,V1[:,:npps1/2+1],50)
    pylab.title('Original')
    pylab.colorbar()
    pylab.figure()
    # pylab.contourf(u2[0,:],t2,V2_ss,50)
    pylab.contourf(u2[0,:],t2,V2_ps,50)
    pylab.title('Mesh')
    pylab.colorbar()
    pylab.show()
    '''

    # reorient V2 to agree with mesh
    V2[:nspan,:] = hstack((V2_ss,V2_ps[:,1:]))
    # extend perturbation to tip clearance region
    for i in range(nspan,V2.shape[0]):
        V2[i,:] = V2[nspan-1,:]

    # make sure things match at the cut
    V2[:,0] = V2[:,-1]

    return V2

'''
r5_path = '/home/ericdow/code/blade_pca/R5/R5_ablade_averfoil_all.as'
r37_path = '/home/ericdow/code/blade_pca/blade_surf.dat'
xyz_r5 = read_coords(r5_path)
xyz_r37 = read_mesh_surf(r37_path)
# number of points in span that are in tip clearance
ntip = 21

# project modes onto rotor 37
V = zeros((xyz_r5.shape[0],xyz_r5.shape[1]))
for i in range(V.shape[0]):
    for j in range(V.shape[1]):
        V[i,j] = cos(2.0*pi*i/V.shape[0])*cos(2.0*pi*j/V.shape[1])
V2 = transform_mode(xyz_r5,V,xyz_r37,ntip)
'''
