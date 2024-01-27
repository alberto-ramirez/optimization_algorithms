##======================================================================================================
## Book: MATLAB: COMPUTACIÓN METAHEURÍSTICA Y BIO-iNSPIRADA
## Chapter 1: Optimización.
## Description: Segundo algoritmo del capitulo 1: Broyden-Fletcher-Goldfard-Shanno (BFGS)
##              Para optimizacion de funciones continuas
## Authors/Researchers: Dr. Erik Cuevas, Dr. Fernando Fausto, Dr. Jorge Gálvez, Dra. Alma Rodríguez
## Research center: Centro Universitario de Ciencias Exactas e Ingenierías, Universidad de Guadalajara
##======================================================================================================
## Programmer: Alberto Ramirez Bello
## Date: January 26th, 2024
##======================================================================================================

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pylab import *
from drawnow import drawnow, figure
from mpl_toolkits import mplot3d
import random

# Función objetivo
a,b,c,e = 1,1,1,1
d = -1
f = 0
A = np.array([ [2*a, b], [b, 2*c] ])
B = np.array([[d], [e]])
C = f
point_n = 25
x = np.linspace(-3,3,num=point_n)
y = np.linspace(-2,2,num=point_n)
xx,yy = np.meshgrid(x,y)
zz = a*xx**2 + b*xx*yy + c*yy**2 + d*xx + e*yy + f
figura = plt.figure( figsize=(14,9) )
axis   = plt.axes( projection='3d' )
mapaColor = plt.get_cmap('summer')
axis.plot_surface(xx,yy,zz, cmap=mapaColor)
axis.set_title('BFGS')
plt.show()

# Seleccion del punto inicial x0
x0 = -3
y0 = -2
p_count = 300
position = np.zeros((p_count,2))
position[0,0] = x0
position[0,1] = y0
t = np.transpose(position[0])
tmpA = np.dot( np.dot( np.transpose(t), A ), t) 
tmpB = np.dot(np.transpose(B),t)
contour_level = (1/2) * tmpA + tmpB + C

## Gradiente de la funcion f(x_1,x_2) = (x1)^2 + (x2)^2 + (x1*x2) - x1 + x2
siz = A.shape
gtmp = np.dot(A,t)
gtmp.shape = (len(t),1)
g = gtmp + B

## Inicializar M con la matriz hessiana I
InvHess = np.identity(2)

## Iniciando el proceso de minimización
for i in range(1,22):
    dirc = np.dot(-InvHess,g)

    # Determinando la nueva posicion con la siguiente ecuación
    #       x^(k+1) = M^k * (g^(k+1) - g^k) + x^k
    position[i,:] = position[i-1,:] + (0.1*np.transpose(dirc))
    t = position[i,:]
    t.shape = (len(position[i-1,:]), 1)
    tmpA2 = np.dot( np.dot( np.transpose(t), A ), t) 
    tmpB2 = np.dot( np.transpose(B),t)
    contour_level_tmp = (1/2) * tmpA2 + tmpB2 + C
    contour_level = np.append(contour_level,contour_level_tmp)
    g_old = g
    gNtmp = np.dot(A,t)
    gNtmp.shape = (len(t),1)
    g = gNtmp + B
    u = position[i,:]
    k = position[i-1,:]
    u.shape = (len(position[i,:]), 1)
    k.shape = (len(position[i-1,:]), 1)
    d_pos = u - k
    d_g = g - g_old
    
    # Obtener M(k) por medio de la ecuación 
    # M^(k+1) = (I-(delta-x_k*(delta-g_k)^T)/((delta-x_k)^T*delta-g_k)))...
    # *M^k*(I-((delta-g_k*(delta-x_k)^T)/((delta-x_k)^T*delta-g_k)))+...
    # (delta-x_k*(delta-x_k)^T)/((delta-x_k)^T*delta-g_k)
    InvHess_1 = np.dot(np.identity(2) - np.dot(d_pos,np.transpose(d_g))/np.dot(np.transpose(d_pos),d_g),InvHess)
    InvHess_2 = (np.identity(2) - (np.dot(d_g,np.transpose(d_pos)))/(np.dot(np.transpose(d_pos),d_g)) )
    InvHess_3 = np.dot(d_pos,np.transpose(d_pos))/np.dot(np.transpose(d_pos),d_g)
    InvHess = np.dot(InvHess_1,InvHess_2) + InvHess_3

fig, ax = plt.subplots()
cs = ax.contour(xx,yy,zz,sorted(contour_level))
x_last = position[p_count-1,0]
y_last = position[p_count-1,1]
plt.plot(x_last,y_last,'r--')
plt.plot(x0,y0,'bo')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title('(b)')
plt.show()