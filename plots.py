
import numpy as np 
from scipy import optimize
import matplotlib.pyplot as plt
import os
from IPython import get_ipython
from scipy.optimize import curve_fit
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
from scipy import signal as sg
plt.rcParams["font.family"] = "serif"
figsize = (7, 5)
fontsize_title=15
from matplotlib import colors

#get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'qt5')

os.chdir (r'C:\Users\Nicolás Molina\Desktop\labo 4\PE')

# Nombre del archivo a analizar entre ' ' con la terminación .txt incluida:

file1 = ' caracterizacion calorimetro2022-11-09 12-03-22.csv'

datos1 = np.loadtxt(file1, delimiter=',', skiprows=1, unpack=True)

V=datos1[2]
t=datos1[1]
t -= t[0]
#%% ajustes para transformar V a T

file = 'tablas termocupla.txt'


Misdatos = np.loadtxt(file, delimiter=',', skiprows=0) 


Misdatos_ordenados = Misdatos[np.argsort(Misdatos[:, 0])]


V0= Misdatos_ordenados[:,3] # Presión detectada en mBar
errorP= V0*0.01

grados = Misdatos_ordenados[:,0] # Tiempo transcurrido en seg
errort = 0

n_res = len(V0) # Guardamos el largo del vector en esta variable
#print(n_res)
#print(len(grados))


#Acá me defino la función que uso para ajustar (el modelo que propongo), con su variable independiente y sus parámetros

def fun(x, A, B):
    return A*x+B


#Acá digo que quiero los resultados "res" y la matriz de covarianza "cov" en el ajuste de mis datos con el modelo propuesto

res, cov = curve_fit(fun, V0, grados)  #Los errores de R y L corresponden a la raíz cuadrada de la diagonal de la matriz de covarianza

A = res[0]
errorA = np.sqrt(cov[0,0])
B = res[1]
errorB = np.sqrt(cov[1,1])

#Defino el dominio que grafico mi modelo

x = np.linspace(V0[0],V0[-1],100)

#Grafico mis datos y el modelo obtenido en la misma gráfica

fig, axes = plt.subplots()
plt.errorbar(V0, grados, xerr=errort,yerr=errorP, fmt=".b")
axes.plot(x, fun(x, A, B), 'r-') #Pongo los valores obtenido en "res"
plt.grid()
#plt.axis([0,2,0,3.5]) #permite seleccionar la escala
plt.ylabel('V');
plt.xlabel('°C');
plt.show()

#Ahora, con estos datos podemos saber el valor de los parámetros Q/V y P0

P0 = Misdatos_ordenados[0,0]
errorP0 = P0 * 0.01

print('a', A, '+/-', errorA) #Valor de Q/V que corresponde al valor de A
print('o.o', B, '+/-', errorB, 'obtenida del ajuste') #Valor de P0 del ajuste que corresponde al valor de B
print('o.o', P0, '+/-', errorP0, 'obtenida de los datos') #Valor de P0 de los datos
#%%


os.chdir (r'C:\Users\Nicolás Molina\Desktop\labo 4\PE')

# Nombre del archivo a analizar entre ' ' con la terminación .txt incluida:

file1 = ' caracterizacion calorimetro2022-11-09 12-03-22.csv'

datos1 = np.loadtxt(file1, delimiter=',', skiprows=1, unpack=True)

V=datos1[2]*1000
t=datos1[1]
t -= t[0]

T=A*V+B+20

T=T[:1900]
t=t[:1900]

#tiempo=np.arange(0,len(p)/10000,0.0001)

#grafico temp en funcion del tiempo caract calorimetro
plt.close('all')
plt.figure(figsize =figsize, dpi=100)
plt.plot(t,T, linewidth = 2,label = 'Ajuste Lineal')
plt.title("Tensión en función del tiempo con resistencia de 500W",fontsize=fontsize_title)
plt.grid()
plt.xlabel("Tiempo[s]")
plt.ylabel("Temperatura [°C]")
plt.legend(['datos crudos','ajuste: C+D*np.exp(-K*x)'])
plt.show()

#%% ajuste para saber calor especifico calorimetro
Q=627*t
plt.close('all')

def fun(x, C,T0):
    return 5*C*(x-T0)


#Acá digo que quiero los resultados "res" y la matriz de covarianza "cov" en el ajuste de mis datos con el modelo propuesto

res, cov = curve_fit(fun, T, Q)  #Los errores de R y L corresponden a la raíz cuadrada de la diagonal de la matriz de covarianza

C = res[0]
errorC = np.sqrt(cov[0,0])
T0 = res[1]
errorB = np.sqrt(cov[1,1])

print('C=', C, '+/-', errorC) #Valor de A
print('T0=', T0, '+/-', errorB) #Valor de B

#Defino el dominio que grafico mi modelo

x = np.linspace(T[0],T[-1],500)

#Grafico mis datos y el modelo obtenido en la misma gráfica
fig, axes = plt.subplots()
plt.plot(T, Q)
axes.plot(x, fun(x,C,T0), 'r-') #Pongo los valores obtenido en "res"
plt.title("T vs Q con P=627 W",fontsize=fontsize_title)
plt.grid()
#plt.axis([0,2,0,3.5]) #permite seleccionar la escala
plt.ylabel('Q (J)');
plt.xlabel('T (°C)')
plt.legend(['datos crudos','ajuste: 5*C*(x-B)'])
plt.show()
#esta dando 4800 el calor especifico del calorimetro

#%% enfriamiento hierro
file2 = '3kg hierro+agua2022-11-09 12-42-18.csv'

datos2 = np.loadtxt(file2, delimiter=',', skiprows=1, unpack=True)

V2=datos2[2]*1000
t2=datos2[1]
t2 -= t2[0]

T2=A*V2+B


def fun(x, K,C,D):
    return C+D*np.exp(-K*x)


#Acá digo que quiero los resultados "res" y la matriz de covarianza "cov" en el ajuste de mis datos con el modelo propuesto

res, cov = curve_fit(fun, t2, T2)  #Los errores de R y L corresponden a la raíz cuadrada de la diagonal de la matriz de covarianza

K = res[0]
errorK = np.sqrt(cov[0,0])
C = res[1]
errorC = np.sqrt(cov[1,1])
D = res[2]
errorD = np.sqrt(cov[2,2])

print('k=', K, '+/-', errorK) #Valor de A
print('C=', C, '+/-', errorC) #Valor de B
print('D=', D, '+/-', errorD)
#Defino el dominio que grafico mi modelo

x = np.linspace(0,t2[-1],500)

fig, axes = plt.subplots()
plt.plot(t2, T2)
axes.plot(x, fun(x, K,C,D), 'r-') #Pongo los valores obtenido en "res"
plt.title("Temperatura en función del tiempo para el hierro",fontsize=fontsize_title)
plt.grid()
#plt.axis([0,2,0,3.5]) #permite seleccionar la escala
plt.ylabel('Temperatura (°C)');
plt.xlabel('Tiempo (s)')
plt.legend(['datos crudos','ajuste: C+D*np.exp(-K*x)'])
plt.show()
#%%
file3 = '27 barras laton+agua2022-11-09 13-57-40.csv'

datos3 = np.loadtxt(file3, delimiter=',', skiprows=1, unpack=True)

V3=datos3[2]*1000
t3=datos3[1]
t3 -= t3[0]

#recorto la info q no sirve

V3=V3[20:]
t3=t3[20:]

T3=A*V3+B+27


def fun(x, K,C,D):
    return C+D*np.exp(-K*x)


#Acá digo que quiero los resultados "res" y la matriz de covarianza "cov" en el ajuste de mis datos con el modelo propuesto

res, cov = curve_fit(fun, t3, T3)  #Los errores de R y L corresponden a la raíz cuadrada de la diagonal de la matriz de covarianza

K = res[0]
errorK = np.sqrt(cov[0,0])
C = res[1]
errorC = np.sqrt(cov[1,1])
D = res[2]
errorD = np.sqrt(cov[2,2])

print('k=', K, '+/-', errorK) #Valor de A
print('C=', C, '+/-', errorC) #Valor de B
print('D=', D, '+/-', errorD)
#Defino el dominio que grafico mi modelo

x = np.linspace(t3[0],t3[-1],500)

fig, axes = plt.subplots()
plt.plot(t3, T3)
axes.plot(x, fun(x, K,C,D), 'r-') #Pongo los valores obtenido en "res"
plt.title("Temperatura en función del tiempo para el latón(27)",fontsize=fontsize_title)
plt.grid()
#plt.axis([0,2,0,3.5]) #permite seleccionar la escala
plt.ylabel('Temperatura (°C)');
plt.xlabel('Tiempo (s)')
plt.legend(['datos crudos','ajuste: C+D*np.exp(-K*x)'])
plt.show()

