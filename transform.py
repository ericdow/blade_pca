import pylab, os, math, string
from numpy import *
from read_blades import *
from pca import *
from scipy.interpolate import splprep, splev, sproot, splrep

mpath = '/home/ericdow/code/blade_pca/R5/b-blades/'
npath = '/home/ericdow/code/blade_pca/R5/R5_bblade_averfoil_all.as'
icut = 10

# calculate the PCA
# x0: nominal blade shape
# xa: ensamble average
# xm: measured blades 
U, S, V, V0, x0, xa, xm, norm = calc_pca(mpath, npath, icut)

# determine s-coordinates for nominal blade
xyzn = read_coords(npath)
xyn = xyzn[icut,:,:]
tck,sn = splprep([xyn[:,0],xyn[:,1]],s=0,per=1)

# compute the variance of the measured blades
# error in the normal direction
xe = zeros(xm.shape)
en = zeros((xm.shape[0],xm.shape[1]))
for i in arange(xm.shape[0]):
    xe[i,:,:] = xm[i,:,:] - x0
    en[i,:] = xe[i,:,0]*norm[:,0] + xe[i,:,1]*norm[:,1]
var = std(en,axis=0)**2

# find the normal component of the perturbation
Vn = norm[:,0]*V[:,:,0] + norm[:,1]*V[:,:,1]

# make Vn to be periodic
Vn[:,-1] = Vn[:,0]

# compute the derivative at the midpoints
sn_per = hstack((hstack((sn[0]-(sn[-1]-sn[-2]),sn)),sn[-1]+(sn[1]-sn[0])))
Vn_per = hstack((hstack((Vn[:,-2].reshape(-1,1),Vn[:,:])),Vn[:,1].reshape(-1,1)))
h = sn_per[1:] - sn_per[0:-1]
dVds_mid = (Vn_per[:,1:] - Vn_per[:,0:-1])/2.0/h

# interpolate back to the original nodes
sn_mid = 0.5*(sn_per[1:] + sn_per[:-1])
dVds = zeros((dVds_mid.shape[0],len(sn)))
for i in arange(dVds.shape[0]):
    dVds[i,:] = interp(sn,sn_mid,dVds_mid[i,:])

# average the first nm modes
nm = 10
dVds_ave = zeros((len(sn)))
for i in arange(nm):
    # scale by pca weight
    dVds_ave += (S[i]*dVds[i,:])**2
dVds_ave = sqrt(dVds_ave)

# generate a cdf for the transformation
integ = 0.5*h[1:-1]*(dVds_ave[0:-1]+dVds_ave[1:])
cdf = hstack((0.0,integ))
cdf = cumsum(cdf)
scale = cdf[-1]
cdf = cdf/cdf[-1]

# transform the coordinates according to the cdf
sn_trans = cdf

'''
# compute fft of cdf
n = 1024
cdf_u = cdf.copy() - sn
sn_u = linspace(0.0,1.0,n)
cdf_u = interp(sn_u,sn,cdf_u)
F = fft.fft(cdf_u)

# zero out all but first N frequencies
###############################################
###############################################
f = ones(sn_u.shape)*real(F[0])/n

N = 50
F_filt = zeros(F.shape, dtype=complex)
F_filt[0] = F[0]
F_filt[1:N+1] = F[1:N+1]
F_filt[n-N:] = F[n-N:]
m = arange(n)
freq = fft.fftfreq(n)
for k in arange(1,n):
    f += 1.0*real(F_filt[k])/n*cos(2.0*pi*m*freq[k])
    f += 1.0*imag(F_filt[k])/n*sin(2.0*pi*m*freq[k])

f[1:] = flipud(f[1:].copy())
f += sn_u

# numerical approximation of derivative
ds = sn_u[1]-sn_u[0]
pdf_approx = 0.5*(f[2:] - f[:-2])/ds
sn_approx = sn_u[1:-1]

# differentiate the CDF to get the PDF
pdf = zeros(f.shape)
for k in arange(1,n):
    pdf += 2.0*pi*freq[k]*real(F_filt[k])*sin(2.0*pi*m*freq[k])
    pdf -= 2.0*pi*freq[k]*imag(F_filt[k])*cos(2.0*pi*m*freq[k])

pdf[1:] = flipud(pdf[1:].copy())
pdf += 1.0

# scale the pdf to agree with original data
pdf *= scale
###############################################
###############################################

# evaluate the cdf at the original grid locations to 
# obtain the transformation
sn_trans = ones(sn.shape)*real(F[0])/n
for k in arange(1,n):
    sn_trans += 1.0*real(F_filt[k])/n*cos(2.0*pi*sn*(n-1)*freq[k])
    sn_trans += 1.0*imag(F_filt[k])/n*sin(2.0*pi*sn*(n-1)*freq[k])

sn_trans[1:] = flipud(sn_trans[1:].copy())
sn_trans += sn

# make sure sn_trans is zero at zero and unity at one
sn_trans -= sn_trans[0]
sn_trans /= sn_trans[-1]
'''

# plot the correlation map with and without the transformation
u, corr, cov, e_mean, e_stdev = calcCorrelation(icut,mpath,npath,1) 
npps = len(sn)
    
# plot tensor product of PCA modes
if 0:
    for i in range(2):
        pylab.figure()
        pylab.plot(sn,Vn[i,:], '.-')
        pylab.title('Mode ' + str(i+1))

# plot the cdf/pdf transformation maps
if 1:
    ''' 
    pylab.figure()
    pylab.plot(sn,dVds_ave,'--')
    # pylab.plot(sn_u,f)
    # pylab.plot(sn_approx,pdf_approx)
    pylab.plot(sn_u,pdf)
    '''
    
    pylab.figure()
    pylab.plot(sn,cdf,label='Original')
    pylab.grid('True')
    pylab.xlabel('s')
    pylab.title('Transformation Map')
    
    pylab.figure()
    pylab.semilogy(sn,var)
    pylab.grid('True')
    pylab.xlabel('s')
    pylab.title('Variance')
    
    pylab.figure()
    pylab.plot(sn,zeros((len(sn))),'*',label='Original')
    pylab.plot(sn_trans,ones((len(sn))),'.',label='Transformed')
    pylab.xlabel('s')
    pylab.ylim([-1,2])
    pylab.legend()
    
    pylab.figure()
    pylab.plot(sn,dVds_ave)
    pylab.xlabel('s')
    pylab.ylabel('dV/ds')
    pylab.grid('True')
    pylab.title('RMS dV/ds for first 5 modes')

# plot covariance and correlation coefficient
if 1:
    ff = sn.copy()
    sn = sn_trans
    pylab.figure()
    pylab.contourf(sn,sn,corr,50)
    pylab.text(sn[0],sn[0]-0.05*(sn[-1]-sn[0]),'LE')
    pylab.plot(sn[0],sn[0],'k.',markersize=10)
    pylab.text(sn[npps/2],sn[0]-0.05*(sn[-1]-sn[0]),'TE')
    pylab.plot(sn[npps/2],sn[0],'k.',markersize=10)
    pylab.text(sn[-1],sn[0]-0.05*(sn[-1]-sn[0]),'LE')
    pylab.plot(sn[-1],sn[0],'k.',markersize=10)
    pylab.text(sn[0]-0.05*(sn[-1]-sn[0]),sn[npps/2]-0.05*(sn[-1]-sn[0]),'TE')
    pylab.plot(sn[0],sn[npps/2],'k.',markersize=10)
    pylab.text(sn[0]-0.05*(sn[-1]-sn[0]),sn[-1]-0.05*(sn[-1]-sn[0]),'LE')
    pylab.plot(sn[0],sn[-1],'k.',markersize=10)
    pylab.colorbar()
    pylab.title('Transformed Correlation Coefficient')
    pylab.axis('equal')
    
    pylab.figure()
    pylab.contourf(sn,sn,cov,50)
    pylab.text(sn[0],sn[0]-0.05*(sn[-1]-sn[0]),'LE')
    pylab.plot(sn[0],sn[0],'k.',markersize=10)
    pylab.text(sn[npps/2],sn[0]-0.05*(sn[-1]-sn[0]),'TE')
    pylab.plot(sn[npps/2],sn[0],'k.',markersize=10)
    pylab.text(sn[-1],sn[0]-0.05*(sn[-1]-sn[0]),'LE')
    pylab.plot(sn[-1],sn[0],'k.',markersize=10)
    pylab.text(sn[0]-0.05*(sn[-1]-sn[0]),sn[npps/2]-0.05*(sn[-1]-sn[0]),'TE')
    pylab.plot(sn[0],sn[npps/2],'k.',markersize=10)
    pylab.text(sn[0]-0.05*(sn[-1]-sn[0]),sn[-1]-0.05*(sn[-1]-sn[0]),'LE')
    pylab.plot(sn[0],sn[-1],'k.',markersize=10)
    pylab.colorbar()
    pylab.title('Transformed Covariance')
    pylab.axis('equal')
    sn = ff

'''
# remove first two modes
pylab.figure()
S[0:1] = 0.0
sn = sn_trans
X = dot(U,dot(diag(S),Vn))
pylab.contourf(sn,sn,dot(X.T,X),50)
pylab.text(sn[0],sn[0]-0.05*(sn[-1]-sn[0]),'LE')
pylab.text(sn[npps/2],sn[0]-0.05*(sn[-1]-sn[0]),'TE')
pylab.plot(sn[0],sn[0],'k.',markersize=10)
pylab.plot(sn[npps/2],sn[0],'k.',markersize=10)
pylab.text(sn[0]-0.05*(sn[-1]-sn[0]),sn[0],'LE')
pylab.text(sn[0]-0.05*(sn[-1]-sn[0]),sn[npps/2],'TE')
pylab.plot(sn[0],sn[0],'k.',markersize=10)
pylab.plot(sn[0],sn[npps/2],'k.',markersize=10)
pylab.colorbar()
pylab.title('Covariance with modes 1 and 2 removed')
pylab.axis('equal')
'''

'''
im = 0
pylab.figure()
pylab.contourf(sn,sn,outer(Vn[im,:],Vn[im,:]),50)
pylab.text(sn[0],sn[0]-0.05*(sn[-1]-sn[0]),'LE')
pylab.text(sn[npps/2],sn[0]-0.05*(sn[-1]-sn[0]),'TE')
pylab.plot(sn[0],sn[0],'k.',markersize=10)
pylab.plot(sn[npps/2],sn[0],'k.',markersize=10)
pylab.text(sn[0]-0.05*(sn[-1]-sn[0]),sn[0],'LE')
pylab.text(sn[0]-0.05*(sn[-1]-sn[0]),sn[npps/2],'TE')
pylab.plot(sn[0],sn[0],'k.',markersize=10)
pylab.plot(sn[0],sn[npps/2],'k.',markersize=10)
pylab.colorbar()
pylab.title('Mode ' + str(im+1))
pylab.axis('equal')
'''

'''
pylab.figure()
pylab.plot(sn,cdf)
pylab.plot(sn,sn_trans)

pylab.figure()
pylab.contourf(sn_trans,sn_trans,corr,50)
pylab.contourf(sn_trans+sn_trans[-1],sn_trans,corr,50)
pylab.text(sn_trans[0],sn_trans[0]-0.05*(sn_trans[-1]-sn_trans[0]),'LE')
pylab.text(sn_trans[npps/2],sn_trans[0]-0.05*(sn_trans[-1]-sn_trans[0]),'TE')
pylab.plot(sn_trans[0],sn_trans[0],'k.',markersize=10)
pylab.plot(sn_trans[npps/2],sn_trans[0],'k.',markersize=10)
pylab.colorbar()
pylab.title('Transformed')
pylab.axis('equal')

pylab.figure()
pylab.contourf(sn_trans,sn_trans,cov,50)
pylab.contourf(sn_trans+sn_trans[-1],sn_trans,cov,50)
pylab.text(sn_trans[0],sn_trans[0]-0.05*(sn_trans[-1]-sn_trans[0]),'LE')
pylab.text(sn_trans[npps/2],sn_trans[0]-0.05*(sn_trans[-1]-sn_trans[0]),'TE')
pylab.plot(sn_trans[0],sn_trans[0],'k.',markersize=10)
pylab.plot(sn_trans[npps/2],sn_trans[0],'k.',markersize=10)
pylab.colorbar()
pylab.title('Transformed')
pylab.axis('equal')

pylab.figure()
pylab.plot(sn,dVds_ave,'--')
# pylab.plot(sn_u,f)
# pylab.plot(sn_approx,pdf_approx)
pylab.plot(sn_u,pdf)

pylab.figure()
pylab.plot(sn_u,cdf,label='Original')
pylab.grid('True')
pylab.xlabel('s')
pylab.title('CDF of RMS dV/ds')
pylab.legend()

pylab.figure()
pylab.semilogy(sn,var)
pylab.grid('True')
pylab.xlabel('s')
pylab.title('Variance')

pylab.figure()
pylab.plot(sn,zeros((len(sn))),'*',label='Original')
pylab.plot(sn_trans,ones((len(sn))),'.',label='Transformed')
pylab.xlabel('s')
pylab.ylim([-1,2])
pylab.legend()

pylab.figure()
pylab.plot(sn,dVds_ave)
pylab.xlabel('s')
pylab.ylabel('dV/ds')
pylab.grid('True')
pylab.title('RMS dV/ds for first 5 modes')

pylab.figure()
pylab.plot(sn,Vn[0,:],label='Original')
pylab.plot(sn_trans,Vn[0,:],label='Transformed')
pylab.xlabel('s')
pylab.ylabel('V')
pylab.grid('True')
pylab.legend()
pylab.title('Mode 1')

pylab.figure()
pylab.plot(sn,Vn[1,:],label='Original')
pylab.plot(sn_trans,Vn[1,:],label='Transformed')
pylab.xlabel('s')
pylab.ylabel('V')
pylab.grid('True')
pylab.legend()
pylab.title('Mode 2')

pylab.figure()
pylab.plot(sn,Vn[2,:],label='Original')
pylab.plot(sn_trans,Vn[2,:],label='Transformed')
pylab.xlabel('s')
pylab.ylabel('V')
pylab.grid('True')
pylab.legend()
pylab.title('Mode 3')

pylab.figure()
pylab.plot(sn,Vn[3,:],label='Original')
pylab.plot(sn_trans,Vn[3,:],label='Transformed')
pylab.xlabel('s')
pylab.ylabel('V')
pylab.grid('True')
pylab.legend()
pylab.title('Mode 4')
'''

pylab.show()

'''
im = 0 
pylab.figure()
pylab.plot(sn,Vn[im,:],'-*')
pylab.plot(sn,zeros((len(sn),1)),'-*')
'''

