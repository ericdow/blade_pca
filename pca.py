import pylab, os, math, string
from numpy import *
from read_blades import *

def calc_pca(mpath, npath, icut):
    # read measured blades
    xyz = read_all(mpath)
    # number of blades 
    n = xyz.shape[0]
    # number of points per section
    p = xyz.shape[2]
    # spatial dimensions
    m = 2
    
    # read nominal blade surface
    xyzn = read_coords(npath)
    
    # map measured to nominal through normal
    xm = mapBlades(xyzn,xyz,icut,1) # [n x p x m]
    x0 = xyzn[icut,:,:-1] # [p x m]

    # calculate the normals
    norm = calcNormals2D(x0)

    # error
    xe = zeros((n,p,m))
    for i in arange(n):
        xe[i,:,:] = xm[i,:,:] - x0

    # ensemble average of errors
    xa = mean(xe,axis=0)

    # centered
    x = zeros((n,p,m))
    for i in arange(n):
        x[i,:,:] = xe[i,:,:] - xa

    X = zeros((n,m*p))
    for j in arange(p):
        for k in arange(m):
            X[:,m*j+k] = x[:,j,k]
        
    # compute SVD of X
    U, S, V0 = linalg.svd(X,full_matrices=False)

    S /= sqrt(n)

    # reorganize V to give x, y components
    V = zeros((n,p,m))
    for j in arange(p):
        for k in arange(m):
            V[:,j,k] = V0[:,m*j+k]

    # plot a mode
    for im in []:
        scale = 50
        pylab.figure()
        pylab.plot(x0[:,0]+xa[:,0],x0[:,1]+xa[:,1],linewidth=2.0)
        for i in arange(p):
            pylab.arrow(x0[i,0]+xa[i,0],x0[i,1]+xa[i,1],\
                        scale*S[im]*V[im,i,0],\
                        scale*S[im]*V[im,i,1],linewidth=1.0,color='r')
        ymin = x0[:,1].min()
        ymax = x0[:,1].max()
        pylab.axis('Equal')
        pylab.ylim((ymin-0.03*(ymax-ymin),ymax+0.03*(ymax-ymin)))
        pylab.title('Mode '+str(im+1)+', Scale = '+str(scale))
        # leading edge
        pylab.figure()
        pylab.plot(x0[:,0]+xa[:,0],x0[:,1]+xa[:,1],linewidth=2.0)
        for i in arange(p):
            pylab.arrow(x0[i,0]+xa[i,0],x0[i,1]+xa[i,1],\
                        scale*S[im]*V[im,i,0],\
                        scale*S[im]*V[im,i,1],linewidth=1.0,color='r')
        pylab.axis('Equal')
        pylab.xlim((-0.24,-0.14))
        pylab.ylim((0.25,0.33))
        pylab.title('Mode '+str(im+1)+', Scale = '+str(scale)+', LE')
        # trailing edge
        pylab.figure()
        pylab.plot(x0[:,0]+xa[:,0],x0[:,1]+xa[:,1],linewidth=2.0)
        for i in arange(p):
            pylab.arrow(x0[i,0]+xa[i,0],x0[i,1]+xa[i,1],\
                        scale*S[im]*V[im,i,0],\
                        scale*S[im]*V[im,i,1],linewidth=1.0,color='r')
        pylab.axis('Equal')
        pylab.xlim((0.24,0.31))
        pylab.ylim((-0.42,-0.36))
        pylab.title('Mode '+str(im+1)+', Scale = '+str(scale)+', TE')

    '''    
    # plot the eigenvalues and scatter fraction
    pylab.figure()
    pylab.semilogy(arange(S.shape[0]-1)+1,S[:-1],'*')
    pylab.grid(True)
    pylab.xlabel('Mode Index')
    pylab.ylabel('Mode Amplitude ('+ u'\u03c3)')

    pylab.figure()
    # pylab.plot(arange(S.shape[0]),0.99*ones((S.shape[0])),'r--')
    sf = cumsum(S**2)/sum(S**2)
    nn = 10
    ind = arange(nn)+1
    width = 0.35
    pylab.bar(ind,sf[:nn],width)
    pylab.xlim((0,nn+1))
    pylab.xticks(ind+width/2., ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
    for i in range(nn):
        pylab.text(ind[i]-1.8*width,0.95*sf[i],'%1.2f' % sf[i])
    pylab.xlabel('Mode Index')
    pylab.ylabel('Scatter Fraction')
    
    pylab.show()
    '''

    return U, S, V, V0, x0, xa, xm, norm
  
def calc_pca3D(mpath, npath):
    # read measured blades
    xyz = read_all(mpath)
    # number of blades 
    n = xyz.shape[0]
    # number of sections
    nsec = xyz.shape[1]-1 # don't use last section
    # number of points per section
    npps = xyz.shape[2]
    # number of sections x number of points per section
    p = nsec*npps
    # spatial dimensions
    m = 2
    
    # read nominal blade surface
    xyzn = read_coords(npath)

    for isec in arange(nsec):
        # map measured to nominal through normal
        xm = mapBlades(xyzn,xyz,isec,0) # [n x npps x m]
        x0 = xyzn[isec,:,:-1] # [npps x m]

        # calculate the normals
        norm = calcNormals2D(x0)

        # error
        xe = zeros((n,npps,m))
        for i in arange(n):
            xe[i,:,:] = xm[i,:,:] - x0

        # ensemble average of errors
        xa = mean(xe,axis=0)

        # centered
        x = zeros((n,npps,m))
        for i in arange(n):
            x[i,:,:] = xe[i,:,:] - xa # [n x npps x m]

        # insert x into correct spot in X matrix
        Xisec = zeros((n,m*npps))
        for j in arange(npps):
            for k in arange(m):
                Xisec[:,m*j+k] = x[:,j,k]
        if isec == 0:
            X = Xisec
        else:
            X = hstack((X,Xisec))
        
    # compute SVD of X
    U, S, V0 = linalg.svd(X,full_matrices=False)

    # normalize the singular values
    S /= sqrt(n)

    # reorganize V to give x, y components of each slice
    V = zeros((n,nsec,npps,m))
    for isec in arange(nsec):
        Visec = V0[:,isec*m*npps:(isec+1)*m*npps]
        for j in arange(npps):
            for k in arange(m):
                V[:,isec,j,k] = Visec[:,m*j+k]

    Vn = zeros((n,nsec,npps))
    for isec in arange(nsec):
        x0 = xyzn[isec,:,:-1] # [npps x m]
        norm = calcNormals2D(x0)
        Vn[:,isec,:] = V[:,isec,:,0]*norm[:,0] + V[:,isec,:,1]*norm[:,1]

    
    return U, S, V, V0, Vn
