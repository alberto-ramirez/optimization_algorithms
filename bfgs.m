%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Libro: MATLAB: COMPUTACIÓN METAHEURÍSTICA Y BIO-iNSPIRADA
% Chapter 1: Optimización.
% Description: Segundo algoritmo del capitulo 1:
%              Broyden-Fletcher-Goldfard-Shanno (BFGS)
%              Para optimización de funciones continuas
% Authors/Researchers: Dr. Erik Cuevas, Dr. Fernando Fausto, 
%                      Dr. Jorge Gálvez, Dra. Alma Rodríguez
% Research center: Centro Universitario de Ciencias Exactas e Ingenierías 
%                  Universidad de Guadalajara
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Programador: Alberto Ramírez Bello
% Fecha: Enero 26th, 2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Limpiar memoria
clear all

% Función objetivo
a =  1; b = 1; c = 1;
d = -1; e = 1; f = 0;
A = [2*a b; b 2*c];
B = [d;e];
C = f;
point_n = 25;
x = linspace(-3, 3, point_n);
y = linspace(-2, 2, point_n);
[xx,yy] = meshgrid(x,y);
zz = a*xx.^2 + b*xx.*yy + c*yy.^2 + d*xx + e*yy + f;

% Graficando la función
%{ 
mesh(xx,yy,zz);
view([-20 25]);
set(gca, 'box', 'on');
axis([-inf inf -inf inf -inf inf]);
xlabel('X_1');
ylabel('X_2');
zlabel('f(X_1,X_2)');
title('(a)');
figure
%}

% Seleccion del punto inicial x0
x0 = -3; y0 = -2;
p_count = 300;
position = zeros(p_count, 2);
position(1,1) = x0; position(1,2) = y0;
t = position(1, :)';
contour_level = 1/2*t'*A*t+B'*t+C;

% Gradiente de la funcion objetivo definida como:
% f(x_1,x_2) = (x1)^2 + (x2)^2 + (x1*x2) - x1 + x2 
siz = size(A);
g = A*t+B;

% Inicializar M con la matriz hessiana I
InvHess = eye(siz);
d_g=0;

% Iniciando el proceso de minimización
for i = 2:10
    dirc = -InvHess * g;
     
    %  Determinando la nueva posicion con la siguiente ecuación
    %        x^(k+1) = M^k * (g^(k+1) - g^k) + x^k
    position(i,:) = position(i-1,:) + 0.1 * dirc';
    t = position(i,:)';
    contour_level = [contour_level; 1/2*t'*A*t+B'*t+C];
    g_old = g;
    g = A*t+B;
    d_pos = position(i,:)' - position(i-1,:)';
    d_g = g - g_old;
     
    % ----- Obtener M(k) por medio de la ecuacion ------
    % M^(k+1) = (I-(delta-x_k*(delta-g_k)^T)/((delta-x_k)^T*delta-g_k)))...
    % *M^k*(I-((delta-g_k*(delta-x_k)^T)/((delta-x_k)^T*delta-g_k)))+...
    % (delta-x_k*(delta-x_k)^T)/((delta-x_k)^T*delta-g_k)
    InvHess = ( eye(siz) - ( (d_pos*d_g')/(d_pos'*d_g) ) ) * InvHess * ...
    ( eye(siz) - (d_g*d_pos')/(d_pos'*d_g) ) + (d_pos*d_pos')/(d_pos'*d_g);

end

% Graficando los resultados
contour(xx, yy, zz, contour_level);
hold on;
x_last = position(p_count,1);
y_last = position(p_count,2);
plot(x_last, y_last, 'x');
plot(x0,y0,'*')
arrow(position(1:3,1), position(1:3,2));
xlabel('x_1');
ylabel('x_2');
title('(b)')
hold off;
axis image;