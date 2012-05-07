from numpy import *
from interp import * 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pylab
import time

def triray_inter(P0,rd,V0,V1,V2):
    # determine if a ray intersects a triangle
    # inputs
    # P0 : starting point of ray
    # rd : direction of ray
    # V0,V1,V2 : vertices of triangle
    #              o V2 
    #             /^
    #            / | v,y
    #           /  |
    #          /   |
    #      V1 o<---o V0
    #           u,x
    # returns
    # x,y : parametric location of intersection
    # PI : x,y,z coordinates of intersection
    u = V1 - V0
    v = V2 - V0
    n = cross(u,v)
    w0 = P0 - V0
    a = -dot(n,w0)
    b = dot(n,rd)
    r = a / b
    
    # intersection of line and plane
    PI = P0 + r*rd
    
    uu = dot(u,u)
    uv = dot(u,v)
    vv = dot(v,v)
    w = PI - V0
    wu = dot(w,u)
    wv = dot(w,v)
    den = uv*uv - uu*vv
    
    # outside if x < 0 or x > 1
    x = (uv*wv - vv*wu)/den
    if (x < 0.0 or x > 1.0):
        return 0,0,0,0
    # outside if y < 0 or (x+y) > 1
    y = (uv*wu - uu*wv)/den
    if (y < 0.0 or (x+y) > 1.0):
        return 0,0,0,0

    return 1,x,y,PI

def uv_inter(P0,rd,u0,u1,v0,v1,k,l,s,t,P):
    #         4    3
    # (u0,v1) o----o (u1,v1)
    #         |   /|
    #         |  / |
    #         | /  |
    #         |/   |
    # (u0,v0) o----o (u1,v0)
    #         1    2
    #
    #              o V2 
    #             /^
    #            / | 
    #           /  |
    #          /   |
    #      V1 o<---o V0
    #
    #      V0 o--->o V2
    #         |   /
    #         |  / 
    #         | /  
    #         V/   
    #      V1 o

    # check intersection with triangle 1-2-3
    V0 = eval_surf(k,l,s,t,P,u1,v0)
    V1 = eval_surf(k,l,s,t,P,u0,v0)
    V2 = eval_surf(k,l,s,t,P,u1,v1)
    I,x,y,PI = triray_inter(P0,rd,V0,V1,V2)

    if I==1:
        uI = u1 - x*(u1-u0)
        vI = v0 + y*(v1-v0)
        return 1, uI, vI, PI

    # check intersection with triangle 1-3-4
    if I == 0:
        V0 = eval_surf(k,l,s,t,P,u0,v1)        
        I,x,y,PI = triray_inter(P0,rd,V0,V1,V2)
        if I==1:
            uI = u0 + y*(u1-u0)
            vI = v1 - x*(v1-v0)
            return 1, uI, vI, PI

    return 0,0,0,0

def raysurf_inter(P0,rd,f,k,l,s,t,P):
    # inputs
    # f : factor to subdivide surf by each step
    uI = vI = 0.5
    w = 1.0
    I = 1
    err = 10e6
    # subdivide until convergence
    while I and (err > 1e-8):
        umin = max(0.,uI - 0.5*w)
        umax = min(1.,uI + 0.5*w)
        vmin = max(0.,vI - 0.5*w)
        vmax = min(1.,vI + 0.5*w)
        ug = linspace(umin,umax,f+1)
        vg = linspace(vmin,vmax,f+1)

        i = I = 0
        while ((not I) and (i < f)):
            j = 0
            while ((not I) and (j < f)):
                I,uI,vI,PI = uv_inter(P0,rd,ug[i],ug[i+1],vg[j],vg[j+1],k,l,s,t,P)
                j += 1
            i += 1

        PI_uv = eval_surf(k,l,s,t,P,uI,vI)

        if I == 0:
            print 'Error: No Intersection found'

        w /= f

        err = sqrt((PI[0]-PI_uv[0])**2 + (PI[1]-PI_uv[1])**2 + (PI[2]-PI_uv[2])**2)

    return PI,uI,vI
                    
tic = time.clock()

# construct test surface
m = 5
n = 5
Q = zeros((m+1,n+1,3))
s = linspace(0.,1.,m+1)
t = linspace(0.,1.,n+1)
Q[:,:,0], Q[:,:,1] = meshgrid(t,s)
for i in range(m+1):
    for j in range(n+1):
        # Q[i,j,2] = cos(6*s[i])*sin(4*t[j])*t[j]
        Q[i,j,2] = (1.-s[i])*(1.-t[j])

k = l = 4
P, u, v, s, t = interp_surf(Q,k,l)

toc = time.clock()
print 'Time to construct surface : ', toc-tic

P0 = array([0.25,0.25,0.1])
rd = array([0.2,0.1,1.])

tic = time.clock()

PI,uI,vI = raysurf_inter(P0,rd,2,k,l,s,t,P)

toc = time.clock()
print 'Time to intersect : ', toc-tic
print 'uI, vI : ', uI, vI

uu = linspace(0.,1.,20)
vv = linspace(0.,1.,20)
S = zeros((len(uu),len(vv),3))
for i in range(len(uu)):
    for j in range(len(vv)):
        S[i,j,:] = eval_surf(k,l,s,t,P,uu[i],vv[j])

fig = pylab.figure()
ax = Axes3D(fig)
ax.plot_surface(S[:,:,0], S[:,:,1], S[:,:,2], rstride=1, cstride=1, cmap = cm.jet)
ray = vstack((P0-1*rd,P0+1*rd))
inter = vstack((PI,PI))
ax.plot(inter[:,0],inter[:,1],inter[:,2],'*')
#for i in range(m+1):
#    ax.plot(Q[i,:,0],Q[i,:,1],Q[i,:,2],'k*')
ax.plot(ray[:,0],ray[:,1],ray[:,2])
pylab.show()
