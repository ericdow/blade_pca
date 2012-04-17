import pylab, os, math, string
from numpy import *
from read_blades import *
from pca import *
from scipy.interpolate import splprep, splev, sproot, splrep

# where to save figures
figdir = 'plots/2d_pca'

mpath = '/home/ericdow/code/blade_pca/R5/b-blades/'
npath = '/home/ericdow/code/blade_pca/R5/R5_bblade_averfoil_all.as'
icut = 10

# calculate the PCA
# x0: nominal blade shape
# xa: ensamble average
# xm: measured blades 
U, S, V, x0, xa, xm, norm, Z, max_error = calc_pca(mpath, npath, icut)

n = max_error.shape[0]
# find max error over all blades and plot
max_error = (max_error[:,:,0]).max(0)

# determine how many modes are needed
e_thresh = 5.0e-4
nmodes = nonzero((max_error - e_thresh) <= 0.0)[0][0]+1
print 'Total modes required: ', nmodes

pylab.figure()
pylab.semilogy(arange(n)+1,max_error,'*')
pylab.grid(True)
pylab.title('Combined PCA')
pylab.xlabel('Number of modes in reconstruction')
pylab.ylabel('Maximum error/chord')
pylab.ylim([1.0e-5,2.0*max(max_error)])
pylab.savefig(figdir+'/max_error')

error

p = x0.shape[0]

# plot a mode
'''
scales = array((15.,15.))
for im in [0,1]:
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
    pylab.title('PCA mode '+str(im+1))
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
    pylab.title('PCA mode '+str(im+1)+', LE')
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
    pylab.title('PCA mode '+str(im+1)+', TE')

# plot the singular values and scatter fraction
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
