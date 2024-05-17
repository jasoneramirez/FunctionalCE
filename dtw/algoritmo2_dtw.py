import modelo_opt2_dtw
import modelo_opt3_dtw
import run2_dtw
import run3_dtw
#import run3
import sys
import classifier

import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
from pyts.metrics import dtw

from scipy.io import arff
pd.options.mode.chained_assignment = None 
##DATOS

PATH = r'C:\Users\jas_r\OneDrive\Documentos\PhD\datos\UCR_TS_Archive_2015'
#dataset_list=['BeetleFly','DistalPhalanxOutlineCorrect']
#dataset='Earthquakes'
#dataset='DistalPhalanxOutlineCorrect'
dataset='ItalyPowerDemand' #ver como esta estructurado
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




###MODELO

tipo_clasificador='RF'



if tipo_clasificador=='DT':
    leaves, values, restricciones_right, restricciones_left, model, datos_arbol =classifier.decision_tree(X_train,y_train,X_test,y_test)


elif tipo_clasificador=='RF':
    n_arboles=200
    max_depth=4

    leaves, values, restricciones_right, restricciones_left, model, datos_arbol =classifier.random_forest(X_train,y_train,X_test,y_test,n_arboles,max_depth)


##cojo los datos y calculo su clase

x=np.concatenate((X_train,X_test))
y=np.concatenate((y_train,y_test))
y_pred=pd.DataFrame(model.predict(x).tolist(),columns=['y'])
#if list(y_pred['y'].unique())==[0,1]:
#    y_pred['y']=y_pred['y'].apply(lambda x: -1 if x<=0 else 1)
#elif list(y_pred['y'].unique())==[1,2]:
#    y_pred['y']=y_pred['y'].apply(lambda x: -1 if x<=1 else 1) #cuidado porque hay algunos dataset donde las clases son 1 y 2 no 0 y 1
y_pred=np.array(y_pred['y'])

if all(elem == y_pred[0] for elem in y_pred):
    print('todas las instancias son de la misma clase')
    sys.exit()



#individuo=10
#n_curvas=[1,2,3,4]
#n_curvas=[1,2,3,4]

#for n in n_curvas:

#    if tipo_clasificador=='DT':
#        it=0
#        path_sol=[(i,j) for (i,j) in zip(range(24),range(24))]
#        while True:
#            print('iteracion '+str(it))
#            modelo_opt=modelo_opt2_dtw.modelo_opt(leaves, values, restricciones_right, restricciones_left, x, y, y_pred, model, datos_arbol,path_sol)
#            x0, alpha0, alphas, curvas, indices,  x_sol, distancia =run2_dtw.optimizacion(individuo,n,modelo_opt,leaves, values, restricciones_right, restricciones_left, x, y, y_pred,model,datos_arbol)
#            print('distancia modelo: '+str(distancia))
#            dist_dtw, paths=dtw(x0,x_sol,return_path=True)
#            path_sol=[(u,v) for (u,v) in zip(paths[0],paths[1])]
#            print('distancia dtw: '+str(dist_dtw))
#            if dist_dtw>=distancia:
#                break
#            it+=1

#        distancia="{:.3f}".format(distancia)
#        plt.clf()
#        axes = plt.gca()
#        axes.set_ylim([-1.7,2]) #-1.7, 2.1
#        plt.plot(x0,'k',label='instance x0')
#        for k in range(curvas.shape[0]):
#            plt.plot(curvas[k],label='instance '+str(indices[k]))
#        plt.plot(x_sol,'r--', label='counterfactual')
#        plt.legend(loc='upper left')
#        plt.text(15, -1.5,'distance: '+str(distancia)) #15,-15 #60,4.8
#        plt.savefig(str(dataset)+'_'+str(tipo_clasificador)+'_i='+str(individuo)+'_ncurvas='+str(n)+'dtw.png')

#    elif tipo_clasificador=='RF':
#        it=0
#        path_sol=[(i,j) for (i,j) in zip(range(24),range(24))]
#        while True:
#            print('iteracion '+str(it))
#            modelo_opt=modelo_opt3_dtw.modelo_opt(leaves, values, restricciones_right, restricciones_left, x, y, y_pred, model, datos_arbol,path_sol)
#            x0, alpha0, alphas, curvas, indices,  x_sol, distancia =run3_dtw.optimizacion(individuo,n,modelo_opt,leaves, values, restricciones_right, restricciones_left, x, y, y_pred,model,datos_arbol,M_aux)
#            print('distancia modelo: '+str(distancia))
#            dist_dtw, paths=dtw(x0,x_sol,return_path=True)
#            path_sol=[(u,v) for (u,v) in zip(paths[0],paths[1])]
#            print('distancia dtw: '+str(dist_dtw))
#            if dist_dtw>=distancia:
#                break
#            it+=1

#        distancia="{:.3f}".format(distancia)
#        plt.clf()
#        axes = plt.gca()
#        axes.set_ylim([-1.7,2.1]) #-1.7, 2.1
#        plt.plot(x0,'k',label='instance x0')
#        for k in range(curvas.shape[0]):
#            plt.plot(curvas[k],label='instance '+str(indices[k]))
#        plt.plot(x_sol,'r--', label='counterfactual')
#        plt.legend(loc='upper left')
#        plt.text(15, -1.5,'distance: '+str(distancia)) #15,-15 #60,4.8
#        plt.savefig(str(dataset)+'_'+str(tipo_clasificador)+'_i='+str(individuo)+'_ncurvas='+str(n)+'dtw.png')


n_curvas=list(range(1,9))
#n_curvas=[2,3]
dict={}
individuo=10
sol_inicial=[]
path_sol=[(i,j) for (i,j) in zip(range(24),range(24))]
for n in n_curvas:
    print('n_curvas='+str(n))
    if tipo_clasificador=='RF':
        it=0
        #path_sol=[(i,j) for (i,j) in zip(range(24),range(24))]
        while True:
            print('iteracion '+str(it))
            modelo_opt=modelo_opt3_dtw.modelo_opt(leaves, values, restricciones_right, restricciones_left, x, y, y_pred, model, datos_arbol,path_sol)
            x0, alpha0, alphas, curvas, indices,  x_sol, distancia, xis =run3_dtw.optimizacion(individuo,n,modelo_opt,leaves, values, restricciones_right, restricciones_left, x, y, y_pred,model,datos_arbol,M_aux, sol_inicial)
            print('distancia modelo: '+str(distancia))
            dist_dtw, paths=dtw(x0,x_sol,return_path=True)
            path_sol=[(u,v) for (u,v) in zip(paths[0],paths[1])]
            print('distancia dtw: '+str(dist_dtw))
            if dist_dtw>=distancia:
                break
            it+=1

        sol_inicial=[alpha0,alphas,x_sol,xis]
        
        if n<=4:
            distancia="{:.3f}".format(distancia)
            plt.clf()
            axes = plt.gca()
            axes.set_ylim([-1.7,2.1]) #-1.7, 2.1
            plt.plot(x0,'k',label='instance x0')
            for k in range(curvas.shape[0]):
                plt.plot(curvas[k],label='instance '+str(indices[k]))
            plt.plot(x_sol,'r--', label='counterfactual')
            plt.legend(loc='upper left')
            plt.text(15, -1.5,'distance: '+str(distancia)) #15,-15 #60,4.8
            plt.savefig(str(dataset)+'_'+str(tipo_clasificador)+'_i='+str(individuo)+'_ncurvas='+str(n)+'dtw.png')


    dict[n]=distancia


with open('resul_'+str(dataset)+'dist_prot_rf_dtw.txt','a') as f:
    f.write(str(dict))



    

 


