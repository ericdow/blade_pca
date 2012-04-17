import pylab, os, math, string
from numpy import *
from read_blades import *
from pca import *
from scipy.interpolate import splprep, splev, sproot, splrep

# where to save figures
figdir = 'plots/weighted_pca'

mpath = '/home/ericdow/code/blade_pca/R5/b-blades/'
npath = '/home/ericdow/code/blade_pca/R5/R5_bblade_averfoil_all.as'
icut = 10

# remove the first PCA mode from the data
'''
U, S, V, x0, xa, xm, norm, Z = calc_pca(mpath, npath, icut)

# xyz_rec has the first PCA mode removed
xm_rec = zeros(xm.shape)
for iblade in range(n): 
    xe_rec = xa
    for i in range(1,n):
        xe_rec += S[i]*Z[iblade,i]*U[:,i]
    xm_rec[iblade,:,0] = x0[:,0] + xe_rec*norm[:,0] 
    xm_rec[iblade,:,1] = x0[:,1] + xe_rec*norm[:,1]

xm = xm_rec
'''
 
# calculate the weighted PCA
w_1 = 2.*0.005
i_2 = 56
w_2 = 2.*0.005
h   = 0.0025
PCA1, PCA2, PCA3, x0, xa1, xa2, xa3, xm, norm, max_error = calc_pca_weighted(mpath, npath, icut, w_1, i_2, w_2, h)

# find max error over all blades and plot
n = max_error.shape[1]
max_error_le = (max_error[0,:,:,0]).max(0)
max_error_te = (max_error[1,:,:,0]).max(0)
max_error_bb = (max_error[2,:,:,0]).max(0)

# determine how many modes are needed
e_thresh = 5.0e-4
nmodes_le = nonzero((max_error_le - e_thresh) <= 0.0)[0][0]+1
nmodes_te = nonzero((max_error_te - e_thresh) <= 0.0)[0][0]+1
nmodes_bb = nonzero((max_error_bb - e_thresh) <= 0.0)[0][0]+1
print 'LE modes required: ', nmodes_le
print 'TE modes required: ', nmodes_te
print 'BB modes required: ', nmodes_bb
print 'Total modes required: ', nmodes_le+nmodes_te+nmodes_bb

pylab.figure()
pylab.semilogy(arange(n)+1,max_error_le,'*')
pylab.grid(True)
pylab.title('Leading edge')
pylab.xlabel('Number of modes in reconstruction')
pylab.ylabel('Maximum error/chord')
pylab.ylim([1.0e-6,2.0*max(max_error_le)])
pylab.savefig(figdir+'/max_error_le')

pylab.figure()
pylab.semilogy(arange(n)+1,max_error_te,'*')
pylab.grid(True)
pylab.title('Trailing edge')
pylab.xlabel('Number of modes in reconstruction')
pylab.ylabel('Maximum error/chord')
pylab.ylim([1.0e-6,2.0*max(max_error_te)])
pylab.savefig(figdir+'/max_error_te')

pylab.figure()
pylab.semilogy(arange(n)+1,max_error_bb,'*')
pylab.grid(True)
pylab.title('Blade body')
pylab.xlabel('Number of modes in reconstruction')
pylab.ylabel('Maximum error/chord')
pylab.ylim([1.0e-5,2.0*max(max_error_bb)])
pylab.savefig(figdir+'/max_error_bb')

error

# plot some modes
modes_to_plot = [0,1]
modes_to_plot = []

U = PCA3[0]
xa = xa3
S = PCA3[1]
scales = array((15.,15.))
for im in modes_to_plot:
    # PS/SS
    scale = scales[im]
    pylab.figure()
    pylab.plot(x0[:,0]+xa*norm[:,0],x0[:,1]+xa*norm[:,1],linewidth=2.0)
    for i in arange(p):
        pylab.arrow(x0[i,0]+xa[i]*norm[i,0],x0[i,1]+xa[i]*norm[i,1],\
                    scale*S[im]*U[i,im]*norm[i,0],\
                    scale*S[im]*U[i,im]*norm[i,1],linewidth=1.0,color='r')
    ymin = x0[:,1].min()
    ymax = x0[:,1].max()
    pylab.axis('Equal')
    pylab.ylim((ymin-0.03*(ymax-ymin),ymax+0.03*(ymax-ymin)))
    pylab.title('PS/SS PCA mode '+str(im+1))
    pylab.savefig(figdir+'/PSSS_mode'+str(im+1))

U = PCA1[0]
xa = xa1
S = PCA1[1] 
for im in modes_to_plot:
    # leading edge
    pylab.figure()
    pylab.plot(x0[:,0]+xa*norm[:,0],x0[:,1]+xa*norm[:,1],linewidth=2.0)
    for i in arange(p):
        pylab.arrow(x0[i,0]+xa[i]*norm[i,0],x0[i,1]+xa[i]*norm[i,1],\
                    scale*S[im]*U[i,im]*norm[i,0],\
                    scale*S[im]*U[i,im]*norm[i,1],linewidth=1.0,color='r')
    pylab.axis('Equal')
    pylab.xlim((-0.24,-0.14))
    pylab.ylim((0.25,0.33))
    pylab.title('LE PCA mode '+str(im+1))
    pylab.savefig(figdir+'/LE_mode'+str(im+1))

U = PCA2[0]
xa = xa2
S = PCA2[1] 
for im in modes_to_plot:
    # trailing edge
    pylab.figure()
    pylab.plot(x0[:,0]+xa*norm[:,0],x0[:,1]+xa*norm[:,1],linewidth=2.0)
    for i in arange(p):
        pylab.arrow(x0[i,0]+xa[i]*norm[i,0],x0[i,1]+xa[i]*norm[i,1],\
                    scale*S[im]*U[i,im]*norm[i,0],\
                    scale*S[im]*U[i,im]*norm[i,1],linewidth=1.0,color='r')
    pylab.axis('Equal')
    pylab.xlim((0.24,0.31))
    pylab.ylim((-0.42,-0.36))
    pylab.title('TE PCA mode '+str(im+1))
    pylab.savefig(figdir+'/TE_mode'+str(im+1))

# plot the singular values and scatter fraction
'''
pylab.figure()
pylab.semilogy(arange(S.shape[0]-1)+1,S[:-1],'*')
pylab.grid(True)
pylab.xlabel('PCA mode index')
pylab.ylabel('PCA mode amplitude ('+ u'\u03c3)')

pylab.figure()
sf = cumsum(S**2)/sum(S**2)
nn = 10
ind = arange(nn)+1
width = 0.35
pylab.bar(ind,sf[:nn],width)
pylab.xlim((0,nn+1))
pylab.xticks(ind+width/2., ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
for i in range(nn):
    pylab.text(ind[i]-1.8*width,0.95*sf[i],'%1.2f' % sf[i])
pylab.xlabel('PCA mode index')
pylab.ylabel('Scatter fraction')
pylab.show()
'''

# look at covariances of Zi for weighted PCA
'''
Z1 = PCA1[3]
Z2 = PCA2[3]
Z3 = PCA3[3]

Z1bar = mean(Z1,axis=0) 
Z2bar = mean(Z2,axis=0) 
Z3bar = mean(Z3,axis=0) 

# covariance of Z1 with itself
N = Z1.shape[0]
cov11 = zeros((N,N))
for i in range(N):
    for j in range(i+1):
        cov11[i,j] = sum((Z1[:,i]-Z1bar[i])*(Z1[:,j]-Z1bar[j]))/(N-1.0)
        cov11[j,i] = cov11[i,j]

pylab.matshow(cov11)
pylab.colorbar()

# covariance of Z1 with Z3
N = Z1.shape[0]
cov13 = zeros((N,N))
for i in range(N):
    for j in range(i+1):
        cov13[i,j] = sum((Z1[:,i]-Z1bar[i])*(Z3[:,j]-Z3bar[j]))/(N-1.0)
        cov13[j,i] = cov13[i,j]

pylab.matshow(cov13)
pylab.colorbar()

pylab.show()
'''
