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
    xm = mapBlades(xyzn,xyz,icut,0) # [n x p x m]
    x0 = xyzn[icut,:,:-1] # [p x m]

    # center the measured data
    x0_center = mean(x0,axis=0)
    for i in range(n):
        xm_center = mean(xm[i,:,:],axis=0)
        dx = x0_center - xm_center
        xm[i,:,:] += dx

    # compute the chord of this section to normalize by
    chord = sqrt((x0[0,0]-x0[(p+1)/2,0])**2 + (x0[0,1]-x0[(p+1)/2,1])**2)

    # calculate the normals
    norm = calcNormals2D(x0)

    # error in the normal direction
    xe = zeros((p,n))
    for i in arange(n):
        xe[:,i] = (xm[i,:,0]-x0[:,0])*norm[:,0] + (xm[i,:,1]-x0[:,1])*norm[:,1]

    # ensemble average of errors
    xa = mean(xe,axis=1)

    # centered
    x = zeros((p,n))
    for i in arange(n):
        x[:,i] = xe[:,i] - xa
 
    # compute SVD of X
    U, S, V = linalg.svd(x,full_matrices=True)

    S /= sqrt(n-1)

    # project samples onto modes to reconstruct measured geometry
    # Z[i,:] are the components for the ith measured blade
    Z = zeros((n,n))
    for i in range(n):
        for j in range(n):
            Z[i,j] = dot(U[:,j],x[:,i]) / S[j]

    # compute the maximum error in the truncated PCA
    # max_error[iblade,nmodes,0] - value of maximum error on iblade truncated to nmodes
    # max_error[iblade,nmodes,1] - index of maximum error 
    max_error = zeros((n,n,2))
    for iblade in range(n):
        for i in range(n):
            xe_rec = copy(xa)
            for ii in range(i+1):
                xe_rec += S[ii]*Z[iblade,ii]*U[:,ii]
            # maximum difference between xe and xe_rec
            max_error[iblade,i,0] = max(abs((xe[:,iblade]-xe_rec)/chord))
            max_error[iblade,i,1] = argmax(abs((xe[:,iblade]-xe_rec)/chord))
 
    return U, S, V, x0, xa, xm, norm, Z, max_error
    
def calc_pca_weighted(mpath, npath, icut, w_1, i_2, w_2, h):
    # inputs
    # icut : the slice to be analyzed
    # w_1  : width at s = 0 (where profile starts)
    # i_2  : index of s ~= 0.5
    # w_2  : width of the weighting function at s ~= 0.5
    # h    : how quickly the weighting function falls off
    
    # read measured blades
    xyz = read_all(mpath)
    # number of blades 
    n = xyz.shape[0]
    # number of points per section
    p = xyz.shape[2]
    # spatial dimensions
    m = 2

    xyzn = read_coords(npath)
    x0 = xyzn[icut,:,:-1] # [p x m]

    # map measured to nominal through normal
    xm = mapBlades(xyzn,xyz,icut,0) 

    # center the measured data
    x0_center = mean(x0,axis=0)
    for i in range(n):
        xm_center = mean(xm[i,:,:],axis=0)
        dx = x0_center - xm_center
        xm[i,:,:] += dx

    # compute the chord of this section to normalize by
    chord = sqrt((x0[0,0]-x0[(p+1)/2,0])**2 + (x0[0,1]-x0[(p+1)/2,1])**2)

    # calculate the normals
    norm = calcNormals2D(x0)

    # compute the weighting functions
    tck,s = splprep([x0[:,0],x0[:,1]],s=0,per=1)
    a1 = zeros((len(s))) # at s = 0
    a2 = zeros((len(s))) # at s ~= 0.5
    a3 = ones((len(s))) # blade body

    for i in range(len(s)):
        if s[i] > w_1/2.:
            a1[i] += exp(-(s[i]-w_1/2.)**2/h**2)
        if s[i] < 1.0-w_1/2.:
            a1[i] += exp(-(s[i]-1.0+w_1/2.)**2/h**2)
        if s[i] > s[i_2]+w_2/2.:
            a2[i] += exp(-(s[i]-s[i_2]-w_2/2.)**2/h**2)
        if s[i] < s[i_2]-w_2/2.:
            a2[i] += exp(-(s[i]-s[i_2]+w_2/2.)**2/h**2)

        if s[i] <= w_1/2.:
            a1[i] = 1.0
        if s[i] >= 1.0-w_1/2.:
            a1[i] = 1.0
        if (s[i] >= s[i_2]-w_2/2.) and (s[i] <= s[i_2]+w_2/2.):
            a2[i] = 1.0
   
    a3 -= (a1+a2)
    
    '''
    pylab.plot(s,a1) 
    pylab.plot(s,a2) 
    pylab.plot(s,a3)
    pylab.figure()
    pylab.plot(x0[:,0],x0[:,1])
    pylab.plot(x0[:,0]+0.01*a1*norm[:,0],x0[:,1]+0.01*a1*norm[:,1])
    pylab.plot(x0[:,0]+0.01*a2*norm[:,0],x0[:,1]+0.01*a2*norm[:,1])
    pylab.axis('Equal')
    pylab.show()  
    '''

    # error in the normal direction
    xe_orig = zeros((p,n))
    for i in arange(n):
        xe_orig[:,i] = (xm[i,:,0]-x0[:,0])*norm[:,0] + (xm[i,:,1]-x0[:,1])*norm[:,1]
    xe1 = copy(xe_orig)
    xe2 = copy(xe_orig)
    xe3 = copy(xe_orig)

    # scale by weighting function
    for i in arange(n):
        xe1[:,i] = a1*xe_orig[:,i]
        xe2[:,i] = a2*xe_orig[:,i]
        xe3[:,i] = a3*xe_orig[:,i]

    # ensemble average of errors
    xa1 = mean(xe1,axis=1)
    xa2 = mean(xe2,axis=1)
    xa3 = mean(xe3,axis=1)

    # centered
    x1 = zeros((p,n))
    x2 = zeros((p,n))
    x3 = zeros((p,n))
    for i in arange(n):
        x1[:,i] = xe1[:,i] - xa1
        x2[:,i] = xe2[:,i] - xa2
        x3[:,i] = xe3[:,i] - xa3
 
    # compute SVD of X
    U1, S1, V1 = linalg.svd(x1,full_matrices=True)
    U2, S2, V2 = linalg.svd(x2,full_matrices=True)
    U3, S3, V3 = linalg.svd(x3,full_matrices=True)

    S1 /= sqrt(n-1)
    S2 /= sqrt(n-1)
    S3 /= sqrt(n-1)

    # project samples onto modes to reconstruct measured geometry
    # Z[i,:] are the components for the ith measured blade
    Z1 = zeros((n,n))
    Z2 = zeros((n,n))
    Z3 = zeros((n,n))
    for i in range(n):
        for j in range(n):
            Z1[i,j] = dot(U1[:,j],x1[:,i]) / S1[j]
            Z2[i,j] = dot(U2[:,j],x2[:,i]) / S2[j]
            Z3[i,j] = dot(U3[:,j],x3[:,i]) / S3[j]

    # compute the maximum error in the truncated PCA
    # max_error[iblade,nmodes,0] - value of maximum error on iblade truncated to nmodes
    # max_error[iblade,nmodes,1] - index of maximum error 
    max_error = zeros((3,n,n,2))
    for iblade in range(n):
        # error reconstructed from PCA
        for i in range(n):
            xe_rec1 = copy(xa1)
            xe_rec2 = copy(xa2)
            xe_rec3 = copy(xa3)
            for ii in range(i+1):
                xe_rec1 += S1[ii]*Z1[iblade,ii]*U1[:,ii]
                xe_rec2 += S2[ii]*Z2[iblade,ii]*U2[:,ii]
                xe_rec3 += S3[ii]*Z3[iblade,ii]*U3[:,ii]
            # maximum difference between xe and xe_rec
            max_error[0,iblade,i,0] = max(abs((xe1[:,iblade]-xe_rec1)/chord))
            max_error[0,iblade,i,1] = argmax(abs((xe1[:,iblade]-xe_rec1)/chord))
            max_error[1,iblade,i,0] = max(abs((xe2[:,iblade]-xe_rec2)/chord))
            max_error[1,iblade,i,1] = argmax(abs((xe2[:,iblade]-xe_rec2)/chord))
            max_error[2,iblade,i,0] = max(abs((xe3[:,iblade]-xe_rec3)/chord))
            max_error[2,iblade,i,1] = argmax(abs((xe3[:,iblade]-xe_rec3)/chord))

    # test the reconstruction of the error vs number of modes used to reconstruct
    '''
    iblade = 0
    xe_rec = zeros((n,p))
    for i in range(n):
        xe_rec[i,:] = xa1 + xa2 + xa3
        for ii in range(i+1):
            xe_rec[i,:] += S1[ii]*Z1[iblade,ii]*U1[:,ii]
            xe_rec[i,:] += S2[ii]*Z2[iblade,ii]*U2[:,ii]
            xe_rec[i,:] += S3[ii]*Z3[iblade,ii]*U3[:,ii]

    pylab.figure()
    pylab.semilogy(arange(15)+1,S1[:15]**2,'*')
    pylab.grid(True)
    pylab.ylabel('Eigenvalue')
    pylab.xlabel('Index')
    pylab.title('PCA eigenvalues for LE')
    pylab.savefig('plots/weighted_pca/LE_eig.png')

    pylab.figure()
    pylab.semilogy(arange(15)+1,S2[:15]**2,'*')
    pylab.grid(True)
    pylab.ylabel('Eigenvalue')
    pylab.xlabel('Index')
    pylab.title('PCA eigenvalues for TE')
    pylab.savefig('plots/weighted_pca/TE_eig.png')

    pylab.figure()
    pylab.semilogy(arange(n-1)+1,S3[:-1]**2,'*')
    pylab.grid(True)
    pylab.ylabel('Eigenvalue')
    pylab.xlabel('Index')
    pylab.title('PCA eigenvalues for PS/SS')
    pylab.savefig('plots/weighted_pca/PSSS_eig.png')

    pylab.figure()
    pylab.plot(x0[:,0],x0[:,1])
    pylab.plot(x0[:,0]+0.01*a1*norm[:,0],x0[:,1]+0.01*a1*norm[:,1])
    pylab.axis('Equal')
    pylab.xlim((-0.24,-0.14))
    pylab.ylim((0.25,0.33))
    pylab.title('LE weighting function')
    pylab.savefig('plots/weighted_pca/LE_weight.png')

    pylab.figure()
    pylab.plot(x0[:,0],x0[:,1])
    pylab.plot(x0[:,0]+0.01*a2*norm[:,0],x0[:,1]+0.01*a2*norm[:,1])
    pylab.axis('Equal')
    pylab.xlim((0.24,0.31))
    pylab.ylim((-0.42,-0.36))
    pylab.title('TE weighting function')
    pylab.savefig('plots/weighted_pca/TE_weight.png')

    pylab.figure()
    pylab.plot(x0[:,0],x0[:,1])
    pylab.plot(x0[:,0]+0.01*a3*norm[:,0],x0[:,1]+0.01*a3*norm[:,1])
    pylab.axis('Equal')
    pylab.title('PS/SS weighting function')
    pylab.savefig('plots/weighted_pca/PSSS_weight.png')

    ipps = 84
    pylab.figure()
    pylab.plot(arange(n)+1,100*abs(xe_orig[ipps,iblade]-xe_rec[:,ipps])/abs(xe_orig[ipps,iblade]),'*')
    pylab.grid(True)
    pylab.title('Percent error at SS, weighted PCA')
    pylab.ylabel('Percent error')
    pylab.xlabel('Number of modes')
    pylab.savefig('plots/weighted_pca/SS_weighted_error.png')

    pylab.figure()
    pylab.plot(x0[:,0],x0[:,1])
    pylab.plot(x0[ipps,0],x0[ipps,1],'o')
    pylab.savefig('plots/weighted_pca/SS_error_loc.png')
    pylab.axis('Equal')
    '''

    PCA1 = (U1, S1, V1, Z1)
    PCA2 = (U2, S2, V2, Z2)
    PCA3 = (U3, S3, V3, Z3)

    return PCA1, PCA2, PCA3, x0, xa1, xa2, xa3, xm, norm, max_error
  
def calc_pca3D(mpath, npath):
    # read measured blades
    xyz = read_all(mpath)
    # number of blades 
    n = xyz.shape[0]
    # number of sections
    nsec = xyz.shape[1] # don't use last section
    # number of points per section
    npps = xyz.shape[2]
    # number of sections x number of points per section
    p = nsec*npps
    # spatial dimensions
    m = 2
    
    # read nominal blade surface
    xyzn = read_coords(npath)
    nn, tau1n, tau2n = calcNormals3d(xyzn)

    xyzn_tmp = reshape(xyzn,(nsec*npps,3))
    th_n = mean(arctan2(xyzn_tmp[:,2],xyzn_tmp[:,1]))
    x_n = mean(xyzn_tmp[:,0])

    # interpolate the measured blades and write them to a file
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
        xyz[i,:,:,0] = reshape(xm,(nsec,npps))
        xyz[i,:,:,1] = reshape(ym,(nsec,npps))
        xyz[i,:,:,2] = reshape(zm,(nsec,npps))    

    '''
    # center the measured data
    xyzn_center = array([mean(xyzn[:,:,0]),mean(xyzn[:,:,1])])
    for i in range(n):
        xyz_center = array([mean(xyz[i,:,:,0]),mean(xyz[i,:,:,1])])
        dx = xyzn_center[0] - xyz_center[0] 
        dy = xyzn_center[1] - xyz_center[1]
        xyz[i,:,:,0] += dx
        xyz[i,:,:,1] += dy
    '''

    # compute the chord at the hub to normalize by
    x0 = xyzn[0,:,:-1]
    chord = sqrt((x0[0,0]-x0[(npps+1)/2,0])**2 + (x0[0,1]-x0[(npps+1)/2,1])**2)

    xe = zeros((n,nsec,npps))
    for i in range(n):
        nm, tau1m, tau2m = calcNormals3d(xyz[i,:,:,:])

        # calculate the error in the normal direction for this blade
        xe[i,:,:] = calcError(xyzn,nn,xyz[i,:,:,:],nm)

    # print maximum error relative to chord
    print 'Max error/chord: ',abs(xe).max()/chord

    '''    
    # ensemble average of errors
    xa = mean(xe,axis=0)

    # centered
    x  = zeros((n,nsec,npps))
    for i in range(n):
        x[i,:,:] = xe[i,:,:] - xa # [n x nsec x npps]
    '''
    x = copy(xe)

    # insert x into correct spot in X matrix
    for isec in arange(nsec):
        if isec == 0:
            X = x[:,isec,:]
        else:
            X = hstack((X,x[:,isec,:]))
 
    # compute SVD of X
    U, S, V0 = linalg.svd(X,full_matrices=False)

    # normalize the singular values
    # NOTE : need eigenvalues of covariance matrix, which are sqrt(sigma^2/n)
    S /= sqrt(n-1)

    # reorganize V to give x, y components of each slice
    # in this case, V gives the error in the normal direction
    V = zeros((n,nsec,npps))
    for isec in arange(nsec):
        V[:,isec,:] = V0[:,isec*npps:(isec+1)*npps]

    # project samples onto modes to reconstruct measured geometry
    # Z[i,:] are the components for the ith measured blade
    Z = zeros((n,n))
    for i in range(n):
        for j in range(n):
            Z[i,j] = dot(V0[j,:],X[i,:]) / S[j]

    # compute the maximum error in the truncated PCA
    # max_error[iblade,nmodes,0] - value of maximum error on iblade truncated to nmodes
    # max_error[iblade,nmodes,1] - section where the maximum error occurs
    # max_error[iblade,nmodes,2] - point where the maximum error occurs
    max_error = zeros((n,n,3))
    
    '''
    tmp = zeros((nsec,2))
    for iblade in range(n):
        # error reconstructed from PCA
        for i in range(n):
            xe_rec = copy(xa)
            for ii in range(i+1):
                xe_rec += S[ii]*Z[iblade,ii]*V[ii,:,:]

            # maximum difference between xe and xe_rec
            for isec in range(nsec):
                tmp[isec,0] = max(abs((xe[iblade,isec,:]-xe_rec[isec,:])/chord))
                tmp[isec,1] = argmax(abs((xe[iblade,isec,:]-xe_rec[isec,:])/chord))
            max_error[iblade,i,0] = max(tmp[:,0])
            max_error[iblade,i,1] = argmax(tmp[:,0])
            max_error[iblade,i,2] = tmp[argmax(tmp[:,0]),1]
    '''
 
    return U, S, V, V0, Z, max_error


def calc_pca_weighted3D(mpath, npath, w_1, i_2, w_2, h):
    # inputs
    # w_1  : width at s = 0 (where profile starts)
    # i_2  : index of s ~= 0.5
    # w_2  : width of the weighting function at s ~= 0.5
    # h    : how quickly the weighting function falls off
    
    # read measured blades
    xyz = read_all(mpath)
    # number of blades 
    n = xyz.shape[0]
    # number of sections
    nsec = xyz.shape[1] # don't use last section
    # number of points per section
    npps = xyz.shape[2]
    # number of sections x number of points per section
    p = nsec*npps
    # spatial dimensions
    m = 2
    
    # read nominal blade surface
    xyzn = read_coords(npath)
    nn, tau1n, tau2n = calcNormals3d(xyzn)

    # center the measured data
    xyzn_center = array([mean(xyzn[:,:,0]),mean(xyzn[:,:,1])])
    for i in range(n):
        xyz_center = array([mean(xyz[i,:,:,0]),mean(xyz[i,:,:,1])])
        dx = xyzn_center[0] - xyz_center[0] 
        dy = xyzn_center[1] - xyz_center[1]
        xyz[i,:,:,0] += dx
        xyz[i,:,:,1] += dy

    # compute the chord at the hub to normalize by
    x0 = xyzn[0,:,:-1]
    chord = sqrt((x0[0,0]-x0[(npps+1)/2,0])**2 + (x0[0,1]-x0[(npps+1)/2,1])**2)

    # weight each section separately
    for isec in arange(nsec):

        # compute the weighting functions
        tck,s = splprep([xyzn[isec,:,0],xyzn[isec,:,1]],s=0,per=1)
        a1 = zeros((len(s))) # at s = 0
        a2 = zeros((len(s))) # at s ~= 0.5
        a3 = ones((len(s))) # blade body

        for i in range(len(s)):
            if s[i] > w_1/2.:
                a1[i] += exp(-(s[i]-w_1/2.)**2/h**2)
            if s[i] < 1.0-w_1/2.:
                a1[i] += exp(-(s[i]-1.0+w_1/2.)**2/h**2)
            if s[i] > s[i_2]+w_2/2.:
                a2[i] += exp(-(s[i]-s[i_2]-w_2/2.)**2/h**2)
            if s[i] < s[i_2]-w_2/2.:
                a2[i] += exp(-(s[i]-s[i_2]+w_2/2.)**2/h**2)

            if s[i] <= w_1/2.:
                a1[i] = 1.0
            if s[i] >= 1.0-w_1/2.:
                a1[i] = 1.0
            if (s[i] >= s[i_2]-w_2/2.) and (s[i] <= s[i_2]+w_2/2.):
                a2[i] = 1.0
   
        a3 -= (a1+a2)

        # error in the normal direction for each blade
        xe_orig = zeros((n,nsec,npps))
        for i in range(n):
            nm, tau1m, tau2m = calcNormals3d(xyz[i,:,:,:])
            # calculate the error in the normal direction for this blade
            xe_orig[i,:,:] = calcError(xyzn,nn,xyz[i,:,:,:],nm)
        xe1 = copy(xe_orig)
        xe2 = copy(xe_orig)
        xe3 = copy(xe_orig)

        # scale by weighting function
        for i in arange(n):
            for isec in arange(nsec):
                xe1[i,isec,:] = a1*xe_orig[i,isec,:]
                xe2[i,isec,:] = a2*xe_orig[i,isec,:]
                xe3[i,isec,:] = a3*xe_orig[i,isec,:]
        
        # ensemble average of errors
        xa1 = mean(xe1,axis=0)
        xa2 = mean(xe2,axis=0)
        xa3 = mean(xe3,axis=0)
    
        # centered
        x1  = zeros((n,nsec,npps))
        x2  = zeros((n,nsec,npps))
        x3  = zeros((n,nsec,npps))
        for i in range(n):
            x1[i,:,:] = xe1[i,:,:] - xa1 
            x2[i,:,:] = xe2[i,:,:] - xa2
            x3[i,:,:] = xe3[i,:,:] - xa3 
    
        # insert x into correct spot in X matrix
        for isec in arange(nsec):
            if isec == 0:
                X1 = x1[:,isec,:]
                X2 = x2[:,isec,:]
                X3 = x3[:,isec,:]
            else:
                X1 = hstack((X1,x1[:,isec,:]))
                X2 = hstack((X2,x2[:,isec,:]))
                X3 = hstack((X3,x3[:,isec,:]))
 
    # compute SVD of X
    U1, S1, V01 = linalg.svd(X1,full_matrices=False)
    U2, S2, V02 = linalg.svd(X2,full_matrices=False)
    U3, S3, V03 = linalg.svd(X3,full_matrices=False)

    S1 /= sqrt(n-1)
    S2 /= sqrt(n-1)
    S3 /= sqrt(n-1)
 
    # reorganize V to give x, y components of each slice
    # in this case, V gives the error in the normal direction
    V1 = zeros((n,nsec,npps))
    V2 = zeros((n,nsec,npps))
    V3 = zeros((n,nsec,npps))
    for isec in arange(nsec):
        V1[:,isec,:] = V01[:,isec*npps:(isec+1)*npps]
        V2[:,isec,:] = V02[:,isec*npps:(isec+1)*npps]
        V3[:,isec,:] = V03[:,isec*npps:(isec+1)*npps]

    # project samples onto modes to reconstruct measured geometry
    # Z[i,:] are the components for the ith measured blade
    Z1 = zeros((n,n))
    Z2 = zeros((n,n))
    Z3 = zeros((n,n))
    for i in range(n):
        for j in range(n):
            Z1[i,j] = dot(V01[j,:],X1[i,:]) / S1[j]
            Z2[i,j] = dot(V02[j,:],X2[i,:]) / S2[j]
            Z3[i,j] = dot(V03[j,:],X3[i,:]) / S3[j]

    PCA1 = (U1, S1, V1, V01, Z1)
    PCA2 = (U2, S2, V2, V02, Z2)
    PCA3 = (U3, S3, V3, V03, Z3)

    # compute the maximum error in the truncated PCA
    # max_error[iblade,nmodes,0] - value of maximum error on iblade truncated to nmodes
    # max_error[iblade,nmodes,1] - section where the maximum error occurs
    # max_error[iblade,nmodes,2] - point where the maximum error occurs
    max_error = zeros((3,n,n,3))
    
    tmp = zeros((nsec,2))
    for iblade in range(n):
        # error reconstructed from PCA
        for i in range(n):
            xe_rec1 = copy(xa1)
            xe_rec2 = copy(xa2)
            xe_rec3 = copy(xa3)
            for ii in range(i+1):
                xe_rec1 += S1[ii]*Z1[iblade,ii]*V1[ii,:,:]
                xe_rec2 += S2[ii]*Z2[iblade,ii]*V2[ii,:,:]
                xe_rec3 += S3[ii]*Z3[iblade,ii]*V3[ii,:,:]
            # maximum difference between xe and xe_rec
            for isec in range(nsec):
                tmp[isec,0] = max(abs((xe1[iblade,isec,:]-xe_rec1[isec,:])/chord))
                tmp[isec,1] = argmax(abs((xe1[iblade,isec,:]-xe_rec1[isec,:])/chord))
            max_error[0,iblade,i,0] = max(tmp[:,0])
            max_error[0,iblade,i,1] = argmax(tmp[:,0])
            max_error[0,iblade,i,2] = tmp[argmax(tmp[:,0]),1]
            for isec in range(nsec):
                tmp[isec,0] = max(abs((xe2[iblade,isec,:]-xe_rec2[isec,:])/chord))
                tmp[isec,1] = argmax(abs((xe2[iblade,isec,:]-xe_rec2[isec,:])/chord))
            max_error[1,iblade,i,0] = max(tmp[:,0])
            max_error[1,iblade,i,1] = argmax(tmp[:,0])
            max_error[1,iblade,i,2] = tmp[argmax(tmp[:,0]),1]
            for isec in range(nsec):
                tmp[isec,0] = max(abs((xe3[iblade,isec,:]-xe_rec3[isec,:])/chord))
                tmp[isec,1] = argmax(abs((xe3[iblade,isec,:]-xe_rec3[isec,:])/chord))
            max_error[2,iblade,i,0] = max(tmp[:,0])
            max_error[2,iblade,i,1] = argmax(tmp[:,0])
            max_error[2,iblade,i,2] = tmp[argmax(tmp[:,0]),1]

    return PCA1, PCA2, PCA3, max_error

def calc_pca3D_meas(mpath):
    # read measured errors
    # number of blades 
    n = len(os.listdir(mpath))
    npps,nsec,xe = read_mode(mpath+os.listdir(mpath)[0])
    xe = zeros((n,nsec,npps))
    i = 0
    for ifile in os.listdir(mpath):
        npps,nsec,tmp = read_mode(mpath+ifile)
        xe[i,:,:] = tmp.T
        i += 1
 
    # insert x into correct spot in X matrix
    for isec in arange(nsec):
        if isec == 0:
            X = xe[:,isec,:]
        else:
            X = hstack((X,xe[:,isec,:]))
 
    # compute SVD of X
    U, S, V0 = linalg.svd(X,full_matrices=False)

    # normalize the singular values
    # NOTE : need eigenvalues of covariance matrix, which are sqrt(sigma^2/n)
    S /= sqrt(n-1)

    # reorganize V to give x, y components of each slice
    # in this case, V gives the error in the normal direction
    V = zeros((n,nsec,npps))
    for isec in arange(nsec):
        V[:,isec,:] = V0[:,isec*npps:(isec+1)*npps]

    # project samples onto modes to reconstruct measured geometry
    # Z[i,:] are the components for the ith measured blade
    Z = zeros((n,n))
    for i in range(n):
        for j in range(n):
            Z[i,j] = dot(V0[j,:],X[i,:]) / S[j]

    # test reconstruction
    iblade = 10
    xe_rec = zeros((nsec,npps))
    for i in range(n):
        xe_rec += S[i]*Z[iblade,i]*V[i,:,:]

    print abs(xe[iblade,:,:] - xe_rec).max()

    return U, S, V, V0, Z
