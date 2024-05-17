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


def optimizacion(ind,n_curvas,modelo_opt,leaves, values, restricciones_right, restricciones_left, x, y, y_pred,model,datos_arbol,M_aux,sol_inicial):
    
    S=n_curvas

    x0=x[ind]
    y0=y_pred[ind]
    y_deseada=-y0

    T=x0.shape[0]

    X_candidatos=x[y_pred==y_deseada]

    #def dist_candidatos(x_0,X_candidatos,metric):
    #    dist=[]
    #    for i in range(X_candidatos.shape[0]):
    #        if metric=='dtw':
    #            dist.append(dtw(x_0,X_candidatos[i]))
    #        elif metric=='euclidean':
    #            dist.append(np.linalg.norm(x_0-X_candidatos[i]))
            
    #    return dist

    #distancias=dist_candidatos(x0,X_candidatos,'euclidean')

 
    #umbral=np.sort(distancias)[24]
    #vecinos=X_candidatos[np.where(distancias<=umbral)]


    
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
            N = {None : T}, 
            N2 ={None : N2},
            S ={None: S}, 
            M1=M_aux, 
            M2=M_aux, 
            M3={None: 1}, #para la l0
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

    if sol_inicial != []:
        x_ini={}
        for i in range(x0.shape[0]):
            x_ini[i]=list(sol_inicial[2])[i]

        x_inicial=np.array(sol_inicial[2]).reshape(1,-1)
        sol_ini=model.apply(x_inicial)[0] 
        z_sol_ini=list(map(lambda tl:datos_arbol[tl[0]]['index_leaves'].index(tl[1]),enumerate(sol_ini)))

        z={}
        for t in range(n_arboles):
            for l in range(leaves[t]):
                z[t,l]=0
                z[t,z_sol_ini[t]]=1

        D={}
        for t in range(n_arboles):
            D[t]=0
            for l in range(leaves[t]):
                D[t]=D[t]+values[t][l]*z[t,l]

        alpha0_ini=sol_inicial[0]

        alphas_ini={}
        xis_ini={}
        for i in range(N2):
            alphas_ini[i]=list(sol_inicial[1])[i]
            xis_ini[i]=list(sol_inicial[3])[i]


    #print(data)
    #pathinstance = "/".join([path,"prueba3.dat"]) 
    instance = modelo_opt.create_instance(data) 
    #instance.pprint()


    #opt = SolverFactory("ipopt",executable=pathsolver)
    opt= SolverFactory('gurobi', solver_io="python")
    #opt= SolverFactory('glpk')

    opt.options['TimeLimit'] = 1000

    if sol_inicial !=[]:
        for i in range(x0.shape[0]):
            instance.x[i]=x_ini[i]

        for t in range(n_arboles):
            for l in range(leaves[t]):
                instance.z[t,l]=z[t,l]

        for t in range(n_arboles):
            instance.D[t]=D[t]  

        instance.alpha0=alpha0_ini

        for i in range(N2):
            instance.alpha[i]=alphas_ini[i]
            instance.xi[i]=xis_ini[i]


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


    alphas=np.zeros(X_candidatos.shape[0])
    alpha0=instance.alpha0.value
    for i in instance.vecinos:
        alphas[i]=instance.alpha[i].value
        if alphas[i]<=1e-13:
            alphas[i]=0

    xis=np.zeros(X_candidatos.shape[0])
    for i in instance.vecinos:
        xis[i]=instance.xi[i].value
        if xis[i]<=1e-11:
            xis[i]=0

    distancia=math.sqrt(instance.obj())

    indices=[i for i, e in enumerate(alphas) if e != 0]

    curvas=X_candidatos[indices]
            
    #for t in instance.t:
    #    for l in RangeSet(0,instance.leaves[t]-1):
    #        if instance.z[t,l].value==1:
    #            print (str(t)+','+str(l))

    #print(sum(instance.D[t].value for t in instance.t))

    return x0, alpha0, alphas, curvas, indices, x_sol, distancia, xis

