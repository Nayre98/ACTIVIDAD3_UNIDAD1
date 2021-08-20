from Ipython.display import Image
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import scipy as sp
import sympy as sym

#%matplotlib inline
image(filename= 'assets/seriesfourier.jpg')

n = sym.symbol('n')
t = sym.symbol('t')

tmin = 0
tmax=2*np.pi
t.tmax-tmin
w = 2*np.pi/T

# ft es una fracion simbolica
ft=t

#calculamos la integral para a0
f_integral = ft
a0 = (2/T)*sym.integrate(f_integral,(t,tmin,tmax))
print("a0 =")
sym.pprint(a0)

#calculamos la integral para an
f_integral = ft*sym.cos(n*w*t)
an = (2/T)*sym.integrate(f_integral,(t,tmin,tmax))
an = sym.simplify(an)
print("an = ")
sym.pprint(an)

#calculamos la integral para bn
f_integral =ft*sym.sin(n*w*t)
bn = (2/T)*sym.integrate(f_integral,(t,tmin,tmax))
print("bn = ")
bn = sym.simplify(bn)
sym.pprint(bn)

# Definimos el numero de arminico para la expansion
serie = 0
Armonicos = 5
for i in range(1,Armonicos+1):
    #Evaluamos los coeficientes para cada armonico
    an_c=an.subs(n,i)
    bn_c=bn.subs(n,i)
    if abs(an_c) < 0.0001: an_c=0
    if abs(bn_c) <0.0001:bn_c=0

    serie=serie + an_c*sym.cos(i*w*t) # terminos coseno de la serie
    serie = serie + bn_c*sym.sin(i*w*t) # termino seno de la serie

serie = a0/2+serie # expansion final de la serie

print('f(t)=')
sym.pprint(serie)

#convertimos la expresion sympy a una funcion evaluable
fserie = sym.lambdify(t,serie)
f = sym.iambdify(t,ft)

#creamos un vector de tiempo para la grafica
v_tiempo = np.linspace(Tmin, Tmax,50)

# evaluamos las funciones
fserieG = fserie(v_tiempo)
fg = f(v_tiempo)

plt.plot(v_tiempo,fG,label = 'f(t)')
plt.plot(v_tiempo,fserieG,label = 'Expansion')

plt.xlabel('tiempo')
plt.legend()
plt.title('expansion en series de fourier')
plt.show()

#funcion por tramos
n = sym.simbol('n')
t=sym.symbol('t')

tmin = -2
tmax = 2

t=tmax-tmin
w = 2*np.pi/T

f1=-1
f2=1

# ft es una funcion simbolica por tramos
ft = sym.piecewise((f1,((t<= -1) & (t>=-29))),(f2,((t> -1))))
ft

#calculamos la integral para a0
f_integral = ft
a0 = (2/T)*sym.integrate(f_integral,(t,tmin,tmax))
sym.pprint(a0)

#calculamos la integral para an
f_integral = ft*sym.cos(n*w*t)
an = sym.simplify(an)

#calculamos la integral para bn
f_integral = ft*sym.sin(n*w*t)
bn = (2/T)*sym.integrate(f_integral,(t,tmin,tmax))
print("bn =")
bn = sym.simplify(bn)
sym.pprint(bn)

#definicion el numero de armonicos para la expansion
serie = 0
arminicos = 30

for i in range(1,Armonicos+1):

    #Evaluamos los coeficientes para cada armonico
    an_c = an.subs(n,i)
    bn_c = bn.subs(n,i)

    if abs(an_c)< 0.0001: an_c = 0
    if abs (bn_c)<0.0001:bn_c = 0

    serie= serie + an_c*sym.cos (i*w*t)# terminos coseno de la serie
    serie = serie +bn_c*sym.sin(i*w*t) # termono seno de la serie

serie = a0/2+serie # expansion final de la serie
print('f(t) =')
sym.pprint(serie)

# convertimos la expresion sympy a una funcion evaluable
fserie = sym.lambdify(t,serie)
f = sym.lambdify(t,ft)
# creamos un vector de tiempo para la grafica
v_tiempo = np.linspace(Tmin,Tmax,200)
# evaluacion las funciones
fserieG = fserie(v_tiempo)
fG = f(v_tiempo)
plt.plot(v_tiempo,fG,label = 'f(t)')
plt.plot(v.tiempo,fserieG,label = 'Expansion')

plt.xlabel('tiempo')
plt.legend()
plt.title('expansion en series de Fourier')
plt.show()

