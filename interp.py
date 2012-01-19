from numpy import *
import pylab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import *

# http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/INT-APP/SURF-INT-global.html
# http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/INT-APP/PARA-surface.html
# http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/INT-APP/PARA-knot-generation.html
# http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/INT-APP/PARA-chord-length.html
# K. Lee, Principles of CAD/CAM/CAE Systems 

def eval_curve(k,t,P,u):
    # evaluate B-spline at u
    # inputs
    # k : B-spline order
    # t : sequence of knots
    # P : control points
    # u : evaluation parameter value
    # returns
    # A : coordinates of eval point

    # l : t[l] <= u < t[l+1]
    # first and last k repeat, so look at t[k:-k]
    l = searchsorted(t[k:-k],u,'right')+k-1

    A = zeros((k,3))
    for j in range(k):
        A[j,:] = P[l-k+1+j,:]

    for r in range(1,k):
        for j in range(k-1,r-1,-1):
            i = l-k+1+j
            d1 = u-t[i]
            d2 = t[i+k-r]-u
            A[j,:] = (d1*A[j,:]+d2*A[j-1,:])/(d1+d2)

    return A[k-1,:]

def eval_surf(k,l,s,t,P,u,v):
    # evaluate B-spline at u
    # inputs
    # k,l : B-spline order
    # s,t : sequence of knots
    # P   : control points
    # u,v : evaluation parameter value
    # returns
    # A : coordinates of eval point
    
    # m+1 : number of control points in u-direction
    m = P.shape[0]-1
    # n+1 : number of control points in v-direction
    n = P.shape[1]-1

    # calculate temporary control points C
    C = zeros((m+1,3))
    for i in range(m+1):
        C[i,:] = eval_curve(l,t,P[i,:],v)

    return eval_curve(k,s,C,u)

def eval_basis(k,t,i,n,u):
    # n+1 : number of control points
    # evaluate N_{i,k}(u)
    P = zeros((n+1,3))
    P[i,:] = 1.0

    return eval_curve(k,t,P,u)

def interp_curve(Q,k,u=-1,t=-1):
    # fit a B-spline curve to data points Q
    # inputs
    # Q : data points to interpolate
    # k : B-spline order
    # u : parameter values (optional)
    # t : knot locations (optional)
    # returns
    # P : control points
    # u : parameter values
    # t : knot locations
    
    # n+1 : number of control points
    n = Q.shape[0]-1

    if t == -1:
        d = sqrt((Q[1:,0]-Q[0:-1,0])**2 + (Q[1:,1]-Q[0:-1,1])**2 + (Q[1:,2]-Q[0:-1,2])**2)
        t = zeros((n+k+1))
        t[0:k] = 0.0
        t[n+1:] = 1.0
        for i in range(k,n+1):
            den = 0.0
            for m in range(k,n+2):
                den += sum(d[m-k:m-1])
            t[i] = t[i-1] + sum(d[i-k:i-1]) / den
    
    if u == -1: 
        u = zeros((n+1))
        for j in range(n+1):
            u[j] = sum(t[j+1:j+k]) / (k-1)
    
    # form linear system
    B = zeros((n+1,n+1))
    for i in range(n+1):
        for j in range(n+1):
            B[i,j] = eval_basis(k,t,j,n,u[i])[0]
    
    # solve to compute control points
    P = linalg.solve(B,Q)
    
    return P, u, t

def interp_surf(Q,k,l):
    # fit a B-spline surface to data points Q
    # inputs
    # Q : data points to interpolate (m+1)x(n+1)x3
    # k : B-spline order in u-direction
    # l : B-spline order in v-direction
    # returns
    # P   : control points
    # u,v : parameter values
    # s,t : knot values
    
    # m+1 : number of control points in u-direction
    m = Q.shape[0]-1
    # n+1 : number of control points in v-direction
    n = Q.shape[1]-1
    
    # calculate parameters using the chord length method
    tmp = zeros((m+1,n+1))
    for j in range(0,n+1):
        d = sqrt((Q[1:,j,0]-Q[0:-1,j,0])**2\
               + (Q[1:,j,1]-Q[0:-1,j,1])**2\
               + (Q[1:,j,2]-Q[0:-1,j,2])**2)
        L = sum(d)
        tmp[0,j]  = 0.0
        tmp[-1,j] = 1.0
        for i in range(1,m):
            tmp[i,j] = sum(d[:i]) / L
    
    u = average(tmp,1)
    
    for i in range(0,m+1):
        d = sqrt((Q[i,1:,0]-Q[i,0:-1,0])**2\
               + (Q[i,1:,1]-Q[i,0:-1,1])**2\
               + (Q[i,1:,2]-Q[i,0:-1,2])**2)
        L = sum(d)
        tmp[i,0]  = 0.0
        tmp[i,-1] = 1.0
        for j in range(1,n):
            tmp[i,j] = sum(d[:j]) / L
    
    v = average(tmp,0)
    
    # calculate knots from parameters
    s = zeros((m+k+1))
    s[0:k] = 0.0
    s[m+1:] = 1.0
    for j in range(1,m-k+2):
        s[j+k-1] = sum(u[j:j+k-1]) / (k-1)
    
    t = zeros((n+l+1))
    t[0:l] = 0.0
    t[n+1:] = 1.0
    for j in range(1,n-l+2):
        t[j+l-1] = sum(v[j:j+l-1]) / (l-1)
    
    # solve for temporary control points D
    D = zeros((m+1,n+1,3))
    for j in range(n+1):
        B = zeros((m+1,m+1))
        for x in range(m+1):
            for y in range(m+1):
                B[x,y] = eval_basis(k,s,y,m,u[x])[0]
        D[:,j,:] = linalg.solve(B,Q[:,j,:])
    
    # solve for control points P
    P = zeros((m+1,n+1,3))
    for i in range(m+1):
        B = zeros((n+1,n+1))
        for x in range(n+1):
            for y in range(n+1):
                B[x,y] = eval_basis(l,t,y,n,v[x])[0]
        P[i,:,:] = linalg.solve(B,Q[i,:,:])

    return P, u, v, s, t

m = 5
n = 10
Q = zeros((m+1,n+1,3))
s = linspace(0.,1.,m+1)
t = linspace(0.,1.,n+1)
Q[:,:,0], Q[:,:,1] = meshgrid(t,s)
for i in range(m+1):
    for j in range(n+1):
        Q[i,j,2] = s[i]*t[j]

k = l = 4

P, u, v, s, t = interp_surf(Q,k,l)

uu = linspace(0.,1.,20)
vv = linspace(0.,1.,20)
S = zeros((len(uu),len(vv),3))
for i in range(len(uu)):
    for j in range(len(vv)):
        S[i,j,:] = eval_surf(k,l,s,t,P,uu[i],vv[j])

fig = pylab.figure()
ax = Axes3D(fig)
ax.plot_surface(S[:,:,0], S[:,:,1], S[:,:,2], rstride=1, cstride=1, cmap = cm.jet)
for i in range(m+1):
    ax.plot(Q[i,:,0],Q[i,:,1],Q[i,:,2],'k*')
pylab.show()

'''
k = 4
np = 15
Q = zeros((np,3))
s = linspace(0.,1.,np)
for i in range(np):
    Q[i,0] = cos(s[i])
    Q[i,1] = sin(s[i])
    Q[i,2] = s[i]

P, u, t = interp_curve(Q,k)

uu = linspace(0.,1.,100)
S = zeros((len(uu),3)) 
for i in range(len(uu)):
    S[i,:] = eval_curve(k,t,P,uu[i])
    
fig = pylab.figure()
ax = Axes3D(fig)
ax.plot(Q[:,0], Q[:,1], Q[:,2],'*')
ax.plot(S[:,0], S[:,1], S[:,2])
pylab.show()
'''

