import modelo_opt2
import modelo_opt3
import run2
import run3
import sys
import classifier

import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
from pyts.metrics import dtw

#from scipy.io import arff
pd.options.mode.chained_assignment = None 
##DATOS


#PATH = r'C:\Users\jas_r\OneDrive\Documentos\PhD\datos\UCR_TS_Archive_2015'

dataset='ItalyPowerDemand' 
file_train = PATH +"/"+ str(dataset) + "/" + str(dataset) + "_TRAIN"
file_test = PATH +"/"+ str(dataset) + "/" + str(dataset) + "_TEST"
train = np.genfromtxt(fname=file_train, delimiter=",", skip_header=0)
test = np.genfromtxt(fname=file_test, delimiter=",", skip_header=0)    
X_train, y_train = train[:, 1:], train[:, 0]
X_test, y_test = test[:, 1:], test[:, 0]
min_y=min(set(list(y_train)))
max_y=max(set(list(y_train)))
y_train[y_train==min_y]=-1
y_train[y_train==max_y]=1
y_test[y_test==min_y]=-1
y_test[y_test==max_y]=1

M_aux1=abs(X_train).max(axis=0)
M_aux2=abs(X_test).max(axis=0)
M_aux={}
for i in range(X_train.shape[1]):
    M_aux[i]=max(M_aux1[i],M_aux2[i])+1e-6




###MODEL

tipo_clasificador='RF'


if tipo_clasificador=='DT':
    leaves, values, restricciones_right, restricciones_left, model, datos_arbol =classifier.decision_tree(X_train,y_train,X_test,y_test)


elif tipo_clasificador=='RF':
    n_arboles=200
    max_depth=4

    leaves, values, restricciones_right, restricciones_left, model, datos_arbol =classifier.random_forest(X_train,y_train,X_test,y_test,n_arboles,max_depth)



x=np.concatenate((X_train,X_test))
y=np.concatenate((y_train,y_test))
y_pred=pd.DataFrame(model.predict(x).tolist(),columns=['y'])
#if list(y_pred['y'].unique())==[0,1]:
#    y_pred['y']=y_pred['y'].apply(lambda x: -1 if x<=0 else 1)
#elif list(y_pred['y'].unique())==[1,2]:
#    y_pred['y']=y_pred['y'].apply(lambda x: -1 if x<=1 else 1) 
y_pred=np.array(y_pred['y'])




##optimization model

#modelo_opt2: decision tree
#modelo_opt3: random forest


if tipo_clasificador=='DT':
        
    modelo_opt=modelo_opt2.modelo_opt(leaves, values, restricciones_right, restricciones_left, x, y, y_pred, model, datos_arbol)

elif tipo_clasificador=='RF':
        
    modelo_opt=modelo_opt3.modelo_opt(leaves, values, restricciones_right, restricciones_left, x, y, y_pred, model, datos_arbol)



#plot the explanations

n_curvas=list(range(1,10))
media_dist=[]
dict={}
for n in n_curvas:
    funcion_obj=[]
    for i in range(5):
        print('n_curvas='+str(n)+', individuo='+str(i))
        if tipo_clasificador=='DT':
            x0, alpha0, alphas, curvas, indices,  x_sol, distancia =run2.optimizacion(i,n,modelo_opt,leaves, values, restricciones_right, restricciones_left, x, y, y_pred,model,datos_arbol)
            funcion_obj.append(distancia)
        elif tipo_clasificador=='RF':
            x0, alpha0, alphas, curvas, indices,  x_sol, distancia =run3.optimizacion(i,n,modelo_opt,leaves, values, restricciones_right, restricciones_left, x, y, y_pred,model,datos_arbol,M_aux)
            funcion_obj.append(distancia)

            plt.clf()
            axes = plt.gca()
            axes.set_ylim([-1.7,2.1]) #-1.7, 2.1
            plt.plot(x0,'k',label='instance x0')
            for j in range(curvas.shape[0]):
                plt.plot(curvas[j],label='instance '+str(indices[i]))
            plt.plot(x_sol,'r--', label='counterfactual')
            plt.legend(loc='upper left')
            distancia="{:.3f}".format(distancia)
            plt.text(15, -1.5,'distance: '+str(distancia)) #15,-15 #60,4.8
            plt.savefig(str(dataset)+'_'+str(tipo_clasificador)+'_i='+str(i)+'_ncurvas='+str(n)+'l2.png')


    mean_fobj=statistics.mean(funcion_obj)
    media_dist.append(mean_fobj)
    dict[n]=mean_fobj

plt.scatter(n_curvas,media_dist)
plt.savefig('distancias_ncurvas_rf.png')
with open('resul_'+str(dataset)+'dist_prot_rf.txt','a') as f:
    f.write(str(dict))



""" 
individuo=10
#n_curvas=[1,2,3,4]
n_curvas=list(range(1,10))
dict={}
sol_inicial=[]
for n in n_curvas:
    print('n_curvas='+str(n))
    if tipo_clasificador=='DT':
        modelo_opt=modelo_opt2.modelo_opt(leaves, values, restricciones_right, restricciones_left, x, y, y_pred, model, datos_arbol)
        x0, alpha0, alphas, curvas, indices,  x_sol, distancia =run2.optimizacion(individuo,n,modelo_opt,leaves, values, restricciones_right, restricciones_left, x, y, y_pred,model,datos_arbol)

    elif tipo_clasificador=='RF':
        modelo_opt=modelo_opt3.modelo_opt(leaves, values, restricciones_right, restricciones_left, x, y, y_pred, model, datos_arbol)
        x0, alpha0, alphas, curvas, indices,  x_sol, distancia, xis=run3.optimizacion(individuo,n,modelo_opt,leaves, values, restricciones_right, restricciones_left, x, y, y_pred,model,datos_arbol,M_aux, sol_inicial)

        sol_inicial=[alpha0,alphas,x_sol,xis]
        if n<=4:
            distancia="{:.3f}".format(distancia)
            plt.clf()
            axes = plt.gca()
            axes.set_ylim([-1.7,2.1]) #-1.7, 2.1
            plt.plot(x0,'k',label='instance x0')
            for i in range(curvas.shape[0]):
                plt.plot(curvas[i],label='instance '+str(indices[i]))
            plt.plot(x_sol,'r--', label='counterfactual')
            plt.legend(loc='upper left')
            plt.text(15, -1.5,'distance: '+str(distancia)) #15,-15 #60,4.8
            plt.savefig(str(dataset)+'_'+str(tipo_clasificador)+'200_i='+str(individuo)+'_ncurvas='+str(n)+'_l2.png')

    dict[n]=distancia

with open('resul_'+str(dataset)+'dist_prot_rf_l2.txt','a') as f:
    f.write(str(dict)) 
    """