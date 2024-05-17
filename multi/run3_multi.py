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
# cambiar directorio
import os


def optimizacion(ind,n_curvas,modelo_opt,leaves, values, restricciones_right, restricciones_left, x,J, y, y_pred,model,datos_arbol,M1_aux,M2_aux):
    
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

    #distancias=dist_candidatos(x0,X_candidatos,'euclidean')

    #si quiero elegir por numero de vecinos
    #umbral=np.sort(distancias)[24]
    #vecinos=X_candidatos[np.where(distancias<=umbral)]


    #si quiero elegir por el valor
    #q1=statistics.quantiles(distancias)[1]
    #q=np.percentile(distancias,50)
    #umbral=np.array(q)
    #vecinos=X_candidatos[np.where(distancias<=umbral)]

    #X_candidatos=vecinos

    N2=X_candidatos.shape[0]


    #vecino,dist=sol_inicial(x0,X_candidatos,'euclidean')
    
    #print(x0)
    #print(y0)

    x_0={}
    for i in range(x0.shape[0]):
        x_0[i]=list(x0)[i] 

    #vec={}
    #for i in range(vecino.shape[0]):
    #    vec[i]=list(vecino)[i]

    vec={}
    for i in range(X_candidatos.shape[0]):
        for j in range(X_candidatos.shape[1]):
            vec[i,j]=X_candidatos[i][j]

    n_arboles=len(leaves)
    leaf={}
    for t in range(n_arboles):
        leaf[t]=leaves[t]



    n_left=len(restricciones_left)
    n_right=len(restricciones_right)

    restric_left={}
    for i in range(n_left):
        restric_left[i]=restricciones_left[i]

    restric_right={}
    for i in range(n_right):
        restric_right[i]=restricciones_right[i]

    index_list = [] #lista de indices (arbol, leaf)
    for t in range(n_arboles):
            for l in range(leaves[t]):
                index_list.append((t,l))


    values_leaf_dict={}
    for t in range(n_arboles):
        for l in range(leaves[t]):
            values_leaf_dict[(t,l)]=values[t][l]

    data= {None: dict(
            N = {None : T}, #n de tiempos
            N2 ={None : N2},
            J = {None: J},
            S ={None: S}, #n de curvas para la combinacion lineal
            M1=M1_aux, # M1=max(x(x))-c_s+epsi (~1.01) #ajustar
            M2=M2_aux, # M2=max(x(s))+c_s+epsi (~1.01) #ajustar
            M3=M1_aux, #para la l0
            M4={None: 1},
            epsi={None: 1e-6}, #1e-4
            trees={None: n_arboles},
            leaves = leaf,
            values_leaf=values_leaf_dict,
            nleft={None:n_left},
            nright={None:n_right},
            left=restric_left,
            right=restric_right,
            x0=x_0,
            y={None: y_deseada},
            vec=vec,
            )}


    #print(data)
    #pathinstance = "/".join([path,"prueba3.dat"]) 
    instance = modelo_opt.create_instance(data) 
    #instance.pprint()


    #opt = SolverFactory("ipopt",executable=pathsolver)
    opt= SolverFactory('gurobi', solver_io="python")
    #opt= SolverFactory('glpk')

    #opt.options['TimeLimit'] = 2000
    # Resolver        
    results = opt.solve(instance,tee=True) # tee=True: ver iteraciones por pantalla
    #warmstart=True
    #results=opt.solve(instance,tee=True) # 15 minutos de timelimit

    # Cargar los resultados
    #instance.solutions.load_from(results)
    print(instance.obj())
    #for v in instance.component_objects(Var, active=True):
    #        print ("Variable",v)
    #        varobject = getattr(instance, str(v))
    #        for index in varobject:
    #            print ("   ",index, varobject[index].value)


    x_sol=np.zeros(x0.shape[0])
    for i in instance.times:
        x_sol[i]=instance.x[i].value


   
    alphas={}
    alphas0={}

    for j in instance.Jaux:
        alphas0[j]=instance.alphas0[j].value
        alphas[j]=np.zeros(X_candidatos.shape[0])
        for i in instance.vecinos:
            alphas[j][i]=instance.alphas[j,i].value
            if alphas[j][i]<=1e-6:
                alphas[j][i]=0

    distancia=math.sqrt(instance.obj())

    indices_aux={}
    for j in range(J):
        indices_aux[j]=[i for i, e in enumerate(alphas[j]) if e != 0]
    
    indices=[]
    for j in range(J):
        indices=indices+indices_aux[j]
    indices=list(set(indices))

    curvas=X_candidatos[indices]
            
    #for t in instance.t:
    #    for l in RangeSet(0,instance.leaves[t]-1):
    #        if instance.z[t,l].value==1:
    #            print (str(t)+','+str(l))

    #print(sum(instance.D[t].value for t in instance.t))

    #print(alphas)

    xi_g=[]
    for i in instance.Jaux:
        if instance.xi_g[i].value==1:
            xi_g.append(instance.xi_g[i].value)

    return x0, curvas, indices, x_sol, distancia, xi_g

