##==================================================================================
## MATLAB: COMPUTACIÓN METAHEURÍSTICA Y BIO-iNSPIRADA
## Capitulo 1: Optimización.
## Description: Primer algoritmo del capitulo 1: Gradiente Descendiente
## Authors: Erik Cuevas, Fernando Fausto, Jorge Gálvez y Alma Rodríguez
##==================================================================================
## Programmer: Alberto Ramirez Bello
## Date: January 25th, 2024
##==================================================================================

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pylab import *
from drawnow import drawnow, figure
from mpl_toolkits import mplot3d
import random


def funcion_objetivo(x,y):
    '''Esta es la funcion objetivo, para este ejemplo la funcion se define como
       10 - exp( -1 * (x^2 + 3 * y^2) )
    '''
    z = 10 - math.exp( -(x**2 + 3*y**2) )

    return z        


rango = [-1, 1] # Rango de las variables 'x' y 'y' 

## Dibujando la función como referencia
Ndiv = 50
dx = (rango[1] - rango[0]) / Ndiv
dy = (rango[1] - rango[0]) / Ndiv
x_line = np.arange(rango[0],rango[1],dx)
y_line = np.arange(rango[0],rango[1],dy)
x,y = np.meshgrid(x_line,y_line)
z = np.zeros((len(x), len(x[0]) ))

for i in range(len(x)):
    for j in range(len(x[0])):
        z[i][j] = funcion_objetivo(x[i][j], y[i][j])


figura = plt.figure( figsize=(14,9) )
axis   = plt.axes( projection='3d' )
mapaColor = plt.get_cmap('winter')
axis.plot_surface(x,y,z, cmap=mapaColor)
axis.set_title('Gradiente Descendiente')
plt.show()

## Número de iteraciones
k = 0
niter = 200
hstep = 0.001 # pasos del gradiente
alfa = 0.05   # pasos de la optimizacion

# Generacion del punto inicial
xrange = rango[1] - rango[0]
yrange = rango[1] - rango[0]
x1 = random.random() * xrange + rango[0]
x2 = random.random() * yrange + rango[0]
x_list = []
y_list = []
z_list = []

# Proceso iterativo de optimización

while(k < niter):
    zn = funcion_objetivo(x1,x2)

    # Calculando los gradientes
    vx1 = x1 + hstep
    vx2 = x2 + hstep
    gx1 = ( funcion_objetivo(vx1, x2) - zn ) / hstep
    gx2 = ( funcion_objetivo(x1, vx2) - zn ) / hstep

    # Elementos para graficar los puntos de convergencia
    x_list.append(x1)
    y_list.append(x2)
   
    # Calculando el nuevo punto
    x1 = x1 - alfa * gx1
    x2 = x2 - alfa * gx2
    k = k + 1

fig, ax = plt.subplots()
cs = ax.contour(x,y,z)
plt.scatter(x_list, y_list)
ax.clabel(cs, inline=True, fontsize=10)
ax.set_title('Aproximaciones del gradiente descendiente')
plt.show()