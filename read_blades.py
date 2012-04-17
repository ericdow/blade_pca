import pylab, os, math, string
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import *

def read_mode(path):
    lines = file(path).readlines()

    cdim = int(lines[0].split()[1])
    sdim = int(lines[1].split()[1])

    lines = lines[2:]

    V = array([line.strip().split()[0] for line in lines], float).T 
    nc = V.size
    
    if (nc != cdim*sdim):
        print 'Error: number of coordinates read'
        
    V = reshape(V, (cdim,sdim))
    
    return cdim, sdim, V   

def read_mesh_surf(path):
    lines = file(path).readlines()

    cdim = int(lines[0].split()[1])
    sdim = int(lines[1].split()[1])

    lines = lines[2:]
    
    x = array([line.strip().split()[0] for line in lines], float).T
    y = array([line.strip().split()[1] for line in lines], float).T
    z = array([line.strip().split()[2] for line in lines], float).T
    nc = x.size

    if (nc != cdim*sdim):
        print 'Error: number of coordinates read'

    x = reshape(x, (cdim,sdim))
    y = reshape(y, (cdim,sdim))
    z = reshape(z, (cdim,sdim))

    xyz = zeros((sdim,cdim,3))
    xyz[:,:,0] = x.T
    xyz[:,:,1] = y.T
    xyz[:,:,2] = z.T
 
    return xyz

def read_coords(path):
    lines = file(path).readlines()

    nsec = 0
    npps = 0
    for line in lines:
        if line[0:24] == 'TOTAL NUMBER OF SECTIONS':
            nsec = int(line.split()[-1])
        if line[0:28] == 'NUMBER OF POINTS PER SECTION':
            npps = int(line.split()[-1])
            
    xyz = zeros((nsec,npps,3)) 
   
    isec = 0
    il = 0
    for line in lines:
        if line[0:27] == 'SECTION COORDINATES (X,Y,Z)':
            xyz[isec,:,:] = array([ll.strip().split()[-3:] for ll in lines[il:il+npps]], float)
            isec += 1
        il += 1

    return xyz

def read_all(path):

    nb = len(os.listdir(path))

    # xyz is nblades x nsections x npoints x 3 
    xyz = read_coords(path+os.listdir(path)[0])
    xyz = zeros((nb,xyz.shape[0],xyz.shape[1],xyz.shape[2]))
    i = 0
    for ifile in os.listdir(path):
        xyz[i,:,:,:] = read_coords(path+ifile)
        i += 1
    
    return xyz

def cut_blade(xyz,z_cut):

    npps = xyz.shape[1]
    
    # shift points down to cut location
    xyz[:,:,2] -= z_cut

    xy = zeros((npps,2))

    for ipps in arange(npps):
        tck,u = splprep([xyz[:,ipps,0],xyz[:,ipps,1],xyz[:,ipps,2]],s=0)
        # find where this spline intersects specific z plane
        u0 = sproot(tck)[2][0]
        x,y,z0 = splev(u0,tck)
        xy[ipps,0] = x
        xy[ipps,1] = y

    return xy

def splineLineInter(xy,n,p0):
    # find intersection between xy and line through p0 in direction n
    xy[:,0] = xy[:,0] - p0[0]
    xy[:,1] = xy[:,1] - p0[1]
    # rotation angle 
    th = math.atan2(n[0],n[1])
    # rotate the xy points
    xy_c = xy.copy()
    xy[:,0] = xy_c[:,0]*cos(th) - xy_c[:,1]*sin(th)
    xy[:,1] = xy_c[:,0]*sin(th) + xy_c[:,1]*cos(th)
    tck,u = splprep([xy[:,0],xy[:,1]],s=0,per=1)
    u0 = sproot(tck)[0]
    # return only closest intersection point
    if u0.size < 1:
        print '\nno intersection found\n'
        pylab.figure()
        pylab.plot(xy_c[:,0]+p0[0],xy_c[:,1]+p0[1])
        pylab.plot(p0[0],p0[1],'*')
        pylab.plot(p0[0]+linspace(-1,1)*n[0],p0[1]+linspace(-1,1)*n[1])
        pylab.axis('equal')
        pylab.show()
    elif (u0.size == 1):
        xi,yi = splev(u0[0],tck)
    else:
        xi1,yi1 = splev(u0[0],tck)
        xi2,yi2 = splev(u0[1],tck)
        xi = xi2
        yi = yi2
        if ((xi1**2+yi1**2) < (xi2**2+yi2**2)):
            xi = xi1
            yi = yi1
    # rotate intersection points back
    xi_r = xi*cos(th) + yi*sin(th)
    yi_r = -xi*sin(th) + yi*cos(th)
    xi = xi_r + p0[0]
    yi = yi_r + p0[1]

    return xi,yi

def calcNormals2D(xy):
    xyp = vstack((vstack((xy[-1,:],xy[:,:])),xy[0,:]))
    mpts = 0.5*(xyp[:-1,:] + xyp[1:,:])
    v = mpts[1:,:] - mpts[:-1,:]
    n = (vstack((-v[:,1],v[:,0]))).T
    mag = sqrt(n[:,0]**2 + n[:,1]**2)
    n[:,0] = n[:,0]/mag
    n[:,1] = n[:,1]/mag

    return n

def calcNormals3d(xyz):
    # create bivariate b-spline representation of surface
    nsec = xyz.shape[0]
    npps = xyz.shape[1]
    # tau1, tau2 : tangent vectors along blade surface
    tau1 = zeros(xyz.shape)
    tau2 = zeros(xyz.shape)
    # n : normal vector 
    n = zeros(xyz.shape)
    for isec in range(nsec):
        tck,u = splprep([xyz[isec,:,0],xyz[isec,:,1],xyz[isec,:,2]],s=0,per=1)
        deriv = splev(u,tck,der=1)
        tau1[isec,:,:] = vstack((vstack((deriv[0],deriv[1])),deriv[2])).T
        norm = sqrt(tau1[isec,:,0]**2 + tau1[isec,:,1]**2 + tau1[isec,:,2]**2)
        tau1[isec,:,0] /= norm
        tau1[isec,:,1] /= norm
        tau1[isec,:,2] /= norm
    for ipps in range(npps):
        tck,u = splprep([xyz[:,ipps,0],xyz[:,ipps,1],xyz[:,ipps,2]],s=0)
        deriv = splev(u,tck,der=1)
        tau2[:,ipps,:] = vstack((vstack((deriv[0],deriv[1])),deriv[2])).T
        norm = sqrt(tau2[:,ipps,0]**2 + tau2[:,ipps,1]**2 + tau2[:,ipps,2]**2)
        tau2[:,ipps,0] /= norm
        tau2[:,ipps,1] /= norm
        tau2[:,ipps,2] /= norm
    for isec in range(nsec):
        n[isec,:,:] = cross(tau1[isec,:,:],tau2[isec,:,:]) 
        norm = sqrt(n[isec,:,0]**2 + n[isec,:,1]**2 + n[isec,:,2]**2)
        n[isec,:,0] /= norm
        n[isec,:,1] /= norm
        n[isec,:,2] /= norm

    return n,tau1,tau2

def ray_plane_inter(pn,nn,pm,nm,tau1m,tau2m):
    w0 = pn - pm
    a = -dot(nm,w0)
    b = dot(nm,nn)
    r = a / b

def calcError(xyzn,nn,xyzm,nm):
    # inputs
    # xyzn  : points on the nominal surface
    # nn    : normal to the nominal surface
    # xyzm  : points on the measured surface
    # nm    : normal to the measured surface
    # returns
    # e     : error (measured - nominal)
    nsec = xyzn.shape[0]
    npps = xyzn.shape[1]
    xyzn = reshape(xyzn,(nsec*npps,3))
    nn = reshape(nn,(nsec*npps,3))
    xyzm = reshape(xyzm,(nsec*npps,3))
    nm = reshape(nm,(nsec*npps,3))

    w0 = xyzn - xyzm
    a = -(nm[:,0]*w0[:,0] + nm[:,1]*w0[:,1] + nm[:,2]*w0[:,2])
    b =  (nm[:,0]*nn[:,0] + nm[:,1]*nn[:,1] + nm[:,2]*nn[:,2])
    e = a / b
    
    e = reshape(e,(nsec,npps))

    return e
 
def mapBlades(xyzn,xyz,icut,N):
    zcut = xyzn[icut,0,2]
    xyn = xyzn[icut,:,:-1]
    n = calcNormals2D(xyn)
    npps = xyzn.shape[1]

    nb = xyz.shape[0]
    xym = xyz[:,icut,:,:-1]
    # xym = alignBladesLinear(xyn,xym)
    xym = alignBlades(xyn,xym,N)

    for i in arange(nb):
        # xy = cut_blade(xyz[i,:,:,:],zcut)
        xy = xyz[i,icut,:,:-1]
        for j in arange(npps):
            xi,yi = splineLineInter(xy.copy(),n[j,:],xyn[j,:])
            xym[i,j,0] = xi
            xym[i,j,1] = yi

    return xym

def alignBladesLinear(xyn,xym):
    npps = xyn.shape[0]
    xyn_le = xyn[0,:]
    xyn_te = xyn[npps/2,:]
    xyn_mp = 0.5*(xyn_le + xyn_te)
    theta_n = math.atan2(xyn_te[1]-xyn_le[1],xyn_te[0]-xyn_le[0])

    nb = xym.shape[0]
    for i in arange(nb):
        xy = xym[i,:,:].copy()
        xy_le = xy[0,:]
        xy_te = xy[npps/2,:]
        xy_mp = 0.5*(xy_le + xy_te)
        theta = math.atan2(xy_te[1]-xy_le[1],xy_te[0]-xy_le[0])
        # translate midpoints to origin
        xy[:,0] -= xy_mp[0]
        xy[:,1] -= xy_mp[1]
        xym[i,:,0] -= xy_mp[0]
        xym[i,:,1] -= xy_mp[1]
        # rotate measured blade
        dth = theta - theta_n
        xym[i,:,0] = xy[:,0]*cos(dth) + xy[:,1]*sin(dth)
        xym[i,:,1] = xy[:,1]*cos(dth) - xy[:,0]*sin(dth)
        # translate so midpoints coincide
        xym[i,:,0] += xyn_mp[0]
        xym[i,:,1] += xyn_mp[1]

    return xym

def alignBlades(xyn,xym,N):
    if N > 0:
        xyn_c = xyn.copy()
        # (N-1) is the degree of the polynomial to be fit
        npps = xyn.shape[0]

        xyn_le = xyn[0,:]
        xyn_te = xyn[npps/2,:]
        # translate leading edge to origin
        xyn_c[:,0] -= xyn_le[0]
        xyn_c[:,1] -= xyn_le[1]
        # rotate to x axis
        theta_n = math.atan2(xyn_te[1]-xyn_le[1],xyn_te[0]-xyn_le[0])
        tmp = xyn_c.copy()
        xyn_c[:,0] = tmp[:,0]*cos(theta_n) + tmp[:,1]*sin(theta_n)
        xyn_c[:,1] = tmp[:,1]*cos(theta_n) - tmp[:,0]*sin(theta_n)
        # scale blade to have unit length along x-axis
        s_n = 1/xyn_c[npps/2,0]
        xyn_c *= s_n
        # find (N+1) points on x-axis for interpolation
        x_fit = linspace(0.05,0.95,N+1)
        # spline to find mean camber points
        tck = splrep(xyn_c[5:npps/2-5,0],xyn_c[5:npps/2-5,1],s=0)
        yu_fit = splev(x_fit,tck)
        xx = xyn_c[npps/2+5:-5,0]
        yy = xyn_c[npps/2+5:-5,1]
        tck = splrep(xx[::-1],yy[::-1],s=0)
        yl_fit = splev(x_fit,tck)
        # calculate midpoints and transorm back
        mp_n = zeros((N+1,2))
        mp_n[:,0] = x_fit
        mp_n[:,1] = 0.5*(yl_fit+yu_fit)
        mp_n /= s_n
        tmp = mp_n.copy()
        mp_n[:,0] = tmp[:,0]*cos(-theta_n) + tmp[:,1]*sin(-theta_n)
        mp_n[:,1] = tmp[:,1]*cos(-theta_n) - tmp[:,0]*sin(-theta_n)
        mp_n[:,0] += xyn_le[0]
        mp_n[:,1] += xyn_le[1]

        # repeat the process for the measured blades
        nb = xym.shape[0]
        for i in arange(nb):
            xy = xym[i,:,:].copy()
            xy_le = xym[i,0,:]
            xy_te = xym[i,npps/2,:]
            # translate leading edge to origin
            xy[:,0] -= xy_le[0]
            xy[:,1] -= xy_le[1]
            # rotate to x axis
            theta = math.atan2(xy_te[1]-xy_le[1],xy_te[0]-xy_le[0])
            tmp = xy.copy()
            xy[:,0] = tmp[:,0]*cos(theta) + tmp[:,1]*sin(theta)
            xy[:,1] = tmp[:,1]*cos(theta) - tmp[:,0]*sin(theta)
            # scale blade to have unit length along x-axis
            s = 1/xy[npps/2,0]
            xy *= s
            # find (N+1) points on x-axis for interpolation
            x_fit = linspace(0.05,0.95,N+1)
            # spline to find mean camber points
            tck = splrep(xy[5:npps/2-5,0],xy[5:npps/2-5,1],s=0)
            yu_fit = splev(x_fit,tck)
            xx = xy[npps/2+5:-5,0]
            yy = xy[npps/2+5:-5,1]
            tck = splrep(xx[::-1],yy[::-1],s=0)
            yl_fit = splev(x_fit,tck)
            # calculate midpoints and transorm back
            mp = zeros((N+1,2))
            mp[:,0] = x_fit
            mp[:,1] = 0.5*(yl_fit+yu_fit)
            mp /= s
            tmp = mp.copy()
            mp[:,0] = tmp[:,0]*cos(-theta) + tmp[:,1]*sin(-theta)
            mp[:,1] = tmp[:,1]*cos(-theta) - tmp[:,0]*sin(-theta)
            mp[:,0] += xy_le[0]
            mp[:,1] += xy_le[1]

            # fit dx, dy to polynomial
            dx = polyfit(x_fit,mp[:,0] - mp_n[:,0],N)
            dy = polyfit(x_fit,mp[:,1] - mp_n[:,1],N)

            # translate all points according to dx, dy
            for j in arange(npps):
                xym[i,j,0] -= polyval(dx,xy[j,0])
                xym[i,j,1] -= polyval(dy,xy[j,0])

    return xym    

def calcMeanCamber(xy):
    npps = xy.shape[0]
    xyu = xy[0:npps/2,:]
    xyl = vstack((xy[npps/2:,:],xy[0,:]))
    # spline of lower surface
    tck,u = splprep([xyl[:,0],xyl[:,1]],s=0)
    xls,yls = splev(linspace(0.0,1.0,1000),tck)
    # loop over upper surface, find closest point on lower surface
    xymc = zeros(xyu.shape)
    xymc[0,:] = xyu[0,:]
    xymc[-1,:] = xyu[-1,:]
    thick = zeros((xyu.shape[0],1))
    for i in range(1,npps/2-1):
        mind = 1000
        d = (xls-xyu[i,0])**2 + (yls-xyu[i,1])**2
        j = argmin(d)
        xymc[i,0] = 0.5*(xyu[i,0]+xls[j])
        xymc[i,1] = 0.5*(xyu[i,1]+yls[j])
        thick[i] = d[j]/2

    return xymc, thick
    
def xyz2st(x,y,z):
    s0 = sum(sqrt((x[1:,0]-x[0:-1,0])**2 +\
                  (y[1:,0]-y[0:-1,0])**2 +\
                  (z[1:,0]-z[0:-1,0])**2))
    sN = sum(sqrt((x[1:,-1]-x[0:-1,-1])**2 +\
                  (y[1:,-1]-y[0:-1,-1])**2 +\
                  (z[1:,-1]-z[0:-1,-1])**2))

    if (s0 > sN):
        s = cumsum(sqrt((x[1:,0]-x[0:-1,0])**2 +\
                        (y[1:,0]-y[0:-1,0])**2 +\
                        (z[1:,0]-z[0:-1,0])**2))
        s = hstack((0.,s))
    else: 
        s = cumsum(sqrt((x[1:,-1]-x[0:-1,-1])**2 +\
                        (y[1:,-1]-y[0:-1,-1])**2 +\
                        (z[1:,-1]-z[0:-1,-1])**2))
        s = hstack((0.,s))

    t0 = sum(sqrt((x[0,1:]-x[0,0:-1])**2 +\
                  (y[0,1:]-y[0,0:-1])**2 +\
                  (z[0,1:]-z[0,0:-1])**2))
    tN = sum(sqrt((x[-1,1:]-x[-1,0:-1])**2 +\
                  (y[-1,1:]-y[-1,0:-1])**2 +\
                  (z[-1,1:]-z[-1,0:-1])**2))

    if (t0 > tN):
        t = cumsum(sqrt((x[0,1:]-x[0,0:-1])**2 +\
                        (y[0,1:]-y[0,0:-1])**2 +\
                        (z[0,1:]-z[0,0:-1])**2))
        t = hstack((0.,t))
    else: 
        t = cumsum(sqrt((x[-1,1:]-x[-1,0:-1])**2 +\
                        (y[-1,1:]-y[-1,0:-1])**2 +\
                        (z[-1,1:]-z[-1,0:-1])**2))
        t = hstack((0.,t))

    return s, t
 
def calcCorrelation(icut,mpath,npath,N):
    # read measured blades
    xyz = read_all(mpath)
    nb = xyz.shape[0]
    nsec = xyz.shape[1]
    npps = xyz.shape[2]
    
    # read nominal blade surface
    xyzn = read_coords(npath)
    
    # look at spatial correlation of data
    # xym is [nblades] x [npps] x 2
    # xyn is [npps] x 2
    
    xym = mapBlades(xyzn,xyz,icut,N)
    
    xyn = xyzn[icut,:,:-1]
    tck,u = splprep([xyn[:,0],xyn[:,1]],s=0,per=1)
 
    # error is discrepancy in the normal direction
    n = calcNormals2D(xyn)
    e = zeros((nb,npps))
    for i in arange(nb):
        tmp = xym[i,:,:]-xyn
        e[i,:] = tmp[:,0]*n[:,0] + tmp[:,1]*n[:,1]

    # mean/std dev of error
    e_mean = mean(e,axis=0)
    e_stdev = std(e,axis=0)

    # correlation coefficient 
    corr = zeros((len(u),len(u)))
    cov = zeros((len(u),len(u)))
    for i in arange(len(u)):
        for j in arange(i+1):
            corr[i,j] = mean((e[:,i]-e_mean[i])*(e[:,j]-e_mean[j]))/e_stdev[i]/e_stdev[j]
            corr[j,i] = corr[i,j]
            cov[i,j] = mean((e[:,i]-e_mean[i])*(e[:,j]-e_mean[j]))
            cov[j,i] = cov[i,j]
    
    '''    
    pylab.plot(xyzn[icut,:,0],xyzn[icut,:,1],linewidth=2)
    for i in arange(nb):
        pylab.plot(xym[i,:,0],xym[i,:,1])
    pylab.axis('equal')
    '''
    '''
    pylab.figure()
    for i in arange(5):
        pylab.plot(u,e[i,:])
    pylab.plot(u,e_mean)
    '''
    '''
    pylab.plot(u,e_stdev)
    for i in arange(4):
        pylab.plot(u,e[i,:])
    pylab.show()
    '''
    '''
    pylab.figure()
    pylab.contourf(u,u,corr,50)
    pylab.text(u[0],u[0]-0.05*(u[-1]-u[0]),'LE')
    pylab.text(u[npps/2],u[0]-0.05*(u[-1]-u[0]),'TE')
    pylab.plot(u[0],u[0],'k.',markersize=10)
    pylab.plot(u[npps/2],u[0],'k.',markersize=10)
    pylab.colorbar()
    pylab.axis('equal')
    '''
    '''
    cc = 0.99
    pylab.contour(u,u,corr,array((cc,cc)))
    pylab.contour(u+u[-1],u+u[-1],corr,array((cc,cc)))
    pylab.contourf(u+u[-1],u,corr,30)
    pylab.contourf(u,u+u[1],corr,30)
    pylab.contourf(u+u[-1],u+u[-1],corr,30)
    pylab.colorbar()
    '''
    '''
    pylab.plot(u,e_stdev)
    pylab.plot(u,e_mean)
    '''

    return u, corr, cov, e_mean, e_stdev

# calcCorrelation()

'''
# periodic spline around the profile
tck,u = splprep([xy[:,0],xy[:,1]],s=0,per=1)
x,y = splev(linspace(0,1,100),tck)
pylab.plot(x,y)

pylab.plot(xy[:,0],xy[:,1])
pylab.plot(xyzn[icut,:,0],xyzn[icut,:,1])
pylab.axis('equal')
pylab.show()

fig = pylab.figure()
ax = Axes3D(fig)
ax.plot3D(x,y,z)
ax.plot3D(xyz[0,:,ipps,0],xyz[0,:,ipps,1],xyz[0,:,ipps,2],'*')
pylab.show()

# plot a section
isec = 0
for i in arange(nb):
    pylab.plot(xyz[i,isec,:,0],xyz[i,isec,:,1])

fig = pylab.figure()
ax = Axes3D(fig)
for i in arange(nsec):
    ax.plot3D(xyz[0,i,:,0],xyz[0,i,:,1],xyz[0,i,:,2])   

pylab.axis('equal')
pylab.show()
'''
