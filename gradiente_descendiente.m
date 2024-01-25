%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Libro: MATLAB: COMPUTACIÓN METAHEURÍSTICA Y BIO-iNSPIRADA
% Capitulo 1: Optimización.
% Descripción: Primer algoritmo del capitulo 1: Gradiente Descendiente
% Autores:Erik Cuevas, Fernando Fausto, Jorge Gálvez y Alma Rodríguez
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Programador: Alberto Ramírez Bello
% Fecha: Enero 25, 2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Limpiar memoria
clear all

% Definiendo funcion a optimizar: 10 - e^-( (x1)^2+3(x2)^2 )
funstr = '10-(exp(-1*(x^2+3*y^2)))';
f = vectorize(inline(funstr));
range = [-1, 1 -1 1]; % Valores de 'x' y 'y'

% Dibujando la función como referencia.
Ndiv = 50;
dx = ( range(2) - range(1) ) / Ndiv;
dy = ( range(4) - range(3) ) / Ndiv;
[x,y] = meshgrid( range(1):dx:range(2), range(3):dy:range(4));
z = (f(x,y));
figure(1);
surfc(x,y,z)

% Número de iteraciones
k = 0;
niter = 200;

% Pasos del gradiente
hstep = 0.001;

% paso de la optimización
alfa = 0.05;

% Generación del punto inicial
xrange = range(2) - range(1);
yrange = range(4) - range(3);
x1 = rand * xrange + range(1);
x2 = rand * yrange + range(3);
figure

% Proceso iterativo de optimización
while ( k < niter )
    zn = f(x1,x2); % Evaluando la función

    % Calculo de los gradientes
    vx1 = x1 + hstep;
    vx2 = x2 + hstep;
    gx1 = ( f(vx1, x2) - zn ) / hstep;
    gx2 = ( f(x1, vx2) - zn ) / hstep;

    % Graficando el punto
    contour(x,y,z,15); hold on;
    plot( x1, x2, '.', 'MarkerSize', 10, 'MarkerFaceColor','g' );
    drawnow;
    hold on;

    % Se calcula el nuevo punto
    x1 = x1 - alfa * gx1;
    x2 = x2 - alfa * gx2;
    k = k + 1;
end