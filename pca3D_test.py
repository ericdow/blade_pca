import pylab, os, math, string
from numpy import *
from read_blades import *
from pca import *
from scipy.interpolate import splprep, splev, sproot, splrep

mpath = '/home/ericdow/code/blade_pca/R5/a-blades/'
npath = '/home/ericdow/code/blade_pca/R5/R5_ablade_averfoil_all.as'

'''
x = linspace(-1.0,1.0,100)
xT = (0.5*(1 - cos(pi*(x-1)))-1)*0.5*(1-sign(x))\
   +  0.5*(1 - cos(pi*x))*0.5*(1+sign(x))
x = (1./pi*arccos(1.0-2.0*(-abs(xT)+1.0)) - 1.0)*0.5*(1.0-sign(xT))\
  + 1./pi*arccos(1.0-2.0*abs(xT))*0.5*(1.0+sign(xT))

pylab.plot(x,zeros(x.shape),'*')
pylab.plot(xT,ones(x.shape),'*')
pylab.show()

error
'''



# contour plot some modes
xyzn = read_coords(npath)
nsec = xyzn.shape[0]-1
npps = xyzn.shape[1]

# z location of slices
z = xyzn[:-1,0,2]
Z = tile(z,(npps,1))
# chordwise location of slices
C = zeros((nsec,npps))
for isec in arange(nsec):
    xyn = xyzn[isec,:,:]
    tck,sn = splprep([xyn[:,0],xyn[:,1]],s=0,per=1)
    C[isec,:] = sn

# perform the PCA
U, S, V, V0, Vn = calc_pca3D(mpath, npath)

# plot the scatter fraction 
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
# pylab.xticks(ind+width/2., ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
for i in range(nn):
    pylab.text(ind[i]-1.8*width,0.95*sf[i],'%1.2f' % sf[i])
pylab.xlabel('Mode Index')
pylab.ylabel('Scatter Fraction')

C = 2.0*(C - 0.5)
CT = (1./pi*arccos(1.0-2.0*(-abs(C)+1.0)) - 1.0)*0.5*(1.0-sign(C))\
   + 1./pi*arccos(1.0-2.0*abs(C))*0.5*(1.0+sign(C))
CT = 0.5*CT + 0.5
C  = 0.5*C  + 0.5
# mode to plot
for im in [0,1,2,3]:
    # plot the raw modes
    pylab.figure()
    pylab.contourf(C,Z.T,Vn[im,:,:],50)
    pylab.title('Mode '+str(im+1)+' (Original)')
    pylab.ylabel('Spanwise Coordinate')
    pylab.text(0.03,Z[0,0]-0.05,'LE')
    pylab.text(0.5,Z[0,0]-0.05,'TE')
    pylab.text(0.97,Z[0,0]-0.05,'LE')
    pylab.colorbar()
    
    # plot the stretched modes
    pylab.figure()
    pylab.contourf(CT,Z.T,Vn[im,:,:],50)
    pylab.title('Mode '+str(im+1)+' (Transformed)')
    pylab.ylabel('Spanwise Coordinate')
    pylab.text(0.03,Z[0,0]-0.05,'LE')
    pylab.text(0.5,Z[0,0]-0.05,'TE')
    pylab.text(0.97,Z[0,0]-0.05,'LE')
    pylab.colorbar()

pylab.show()
