from __future__ import division
#import Pyomo
from pyomo.environ import *
from pyomo.opt import SolverFactory
# Cargar el modelo
#from modelo_rf import model
#import random_forest as rf
import numpy as np
import pandas as pd
import pickle
from numpy import linalg as l2
import math
import statistics
from pyts.metrics import dtw

# cambiar directorio
import os


def optimizacion(ind,n_curvas,modelo_opt,leaves, values, restricciones_right, restricciones_left, x, y, y_pred,model,datos_arbol):
    
    S=n_curvas

    x0=x[ind]
    y0=y_pred[ind]
    y_deseada=-y0

    T=x0.shape[0]

    X_candidatos=x[y_pred==y_deseada]

    def dist_candidatos(x_0,X_candidatos,metric):
        dist=[]
        for i in range(X_candidatos.shape[0]):
            if metric=='dtw':
                dist.append(dtw(x_0,X_candidatos[i]))
            elif metric=='euclidean':
                dist.append(np.linalg.norm(x_0-X_candidatos[i]))
            
        return dist

    def derivada(x):
        dx=np.zeros(len(x)-1)
        for i in range(len(x)-1):
            dx[i]=x[i+1]-x[i]
        return dx

    #x0_derivada=derivada(x0)
    #x0_derivada2=derivada(x0_derivada)


    #distancias=dist_candidatos(x0,X_candidatos,'euclidean')
    #distancias=dist_candidatos(x0,X_candidatos,'dtw')

    #si quiero elegir por numero de vecinos
    #umbral=np.sort(distancias)[24]
    #vecinos=X_candidatos[np.where(distancias<=umbral)]


    #si quiero elegir por el valor
    #q=np.percentile(distancias,25)
    #umbral=np.array(q)
    #vecinos=X_candidatos[np.where(distancias<=umbral)]
    #print('median: '+str(q))
    #X_candidatos=vecinos

    N2=X_candidatos.shape[0]


    #vecino,dist=sol_inicial(x0,X_candidatos,'euclidean')
    
    #print(x0)
    #print(y0)

    x_0={}
    for i in range(x0.shape[0]):
        x_0[i]=list(x0)[i] 

    #x_0_d={}
    #for i in range(x0_derivada.shape[0]):
    #    x_0_d[i]=list(x0_derivada)[i]

    #x_0_d2={}
    #for i in range(x0_derivada2.shape[0]):
    #   x_0_d2[i]=list(x0_derivada2)[i]

    #vec={}
    #for i in range(vecino.shape[0]):
    #    vec[i]=list(vecino)[i]

    vec={}
    for i in range(X_candidatos.shape[0]):
        for j in range(X_candidatos.shape[1]):
            vec[i,j]=X_candidatos[i][j]

    #if path_sol==[]:
    #    path_sol=[(i,j) for (i,j) in zip(range(24),range(24))] #la euclidea

    #path={}
    #for i in range(x0.shape[0]):
    #    path[i]=path_sol[i]


    leaf=leaves[0]


    n_left=len(restricciones_left)
    n_right=len(restricciones_right)

    restric_left={}
    for i in range(n_left):
        restric_left[i]=restricciones_left[i]

    restric_right={}
    for i in range(n_right):
        restric_right[i]=restricciones_right[i]



    values_leaf_dict={}
    for l in range(leaf):
        values_leaf_dict[l]=values[l]

    data= {None: dict(
            N = {None : T}, #n de tiempos
            N2 ={None : N2},
            S ={None: S}, #n de curvas para la combinacion lineal
            M1={None:10}, # M1=1-c_s (~1.01) #ajustar
            M2={None:10}, # M2=0+c_s+epsi (~1.01) #ajustar
            M3={None: 1.1}, #para la l0
            epsi={None: 1e-6}, #1e-4
            leaves = {None: leaf},
            values_leaf=values_leaf_dict,
            nleft={None:n_left},
            nright={None:n_right},
            left=restric_left,
            right=restric_right,
            x0=x_0,
            #x0_d=x_0_d,
            #x0_d2=x_0_d2,
            #path=path,
            y={None: y_deseada},
            vec=vec,
            )}


    #print(data)
    #pathinstance = "/".join([path,"prueba3.dat"]) 
    instance = modelo_opt.create_instance(data) 
    #instance.path1.pprint()
    #instance.pathlength.pprint()
    #instance.respath2_n.pprint()
    #instance.obj.pprint()

    #opt = SolverFactory("ipopt",executable=pathsolver)
    opt= SolverFactory('gurobi', solver_io="python")
    #opt= SolverFactory('glpk')


    # Resolver        
    results = opt.solve(instance,tee=False) # tee=True: ver iteraciones por pantalla
    #warmstart=True
    #results=opt.solve(instance,tee=True,timelimit=900) # 15 minutos de timelimit
    # Cargar los resultados
    #instance.solutions.load_from(results)
    print(instance.obj())
    #for v in instance.component_objects(Var, active=True):
    #        print ("Variable",v)
    #        varobject = getattr(instance, str(v))
    #        for index in varobject:
    #            print ("   ",index, varobject[index].value)


    x_sol=np.zeros(x0.shape[0])
    for i in instance.t:
        x_sol[i]=instance.x[i].value


    alphas=np.zeros(X_candidatos.shape[0])
    alpha0=instance.alpha0.value
    for i in instance.vecinos:
        alphas[i]=instance.alpha[i].value
        if alphas[i]<=1e-11:
            alphas[i]=0

    distancia=math.sqrt(instance.obj())

    indices=[i for i, e in enumerate(alphas) if e != 0]

    curvas=X_candidatos[indices] #estos indices no son los originales del dataset x!!!
          
   
    #print('distancia x0 a curva: '+str(np.linalg.norm(x0-curvas)))
    #print(distancia)

    return x0, alpha0, alphas, curvas, indices, x_sol, distancia 
