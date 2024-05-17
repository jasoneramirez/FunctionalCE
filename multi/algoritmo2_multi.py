import modelo_opt2_multi
import run2_multi
import modelo_opt3_multi
import run3_multi
import sys
import classifier

import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
import pickle
#from pyts.metrics import dtw

from scipy.io import arff
pd.options.mode.chained_assignment = None 


##DATOS





PATH = r'C:\Users\jas_r\OneDrive - UNIVERSIDAD DE SEVILLA\PhD\Functional Data Counterfactuals\Multivariate_arff'
dataset='NATOPS'
file_train = PATH +"/"+ str(dataset) + "/" + str(dataset) + "_TRAIN.arff" 
file_test = PATH +"/"+ str(dataset) + "/" + str(dataset) + "_TEST.arff"
train_data = arff.loadarff(file_train)
train =pd.DataFrame(train_data[0])
test_data = arff.loadarff(file_test)
test =pd.DataFrame(test_data[0])
#divido en features
nom_col=train.columns.tolist()
nom_att=nom_col[0]
nom_class=nom_col[1]
n_features=train[nom_att][0].shape[0]
for i in range(n_features):
    train['f'+str(i)]=train[nom_att].apply(lambda x: x[i])
train=train.drop(columns=[nom_att])
for i in range(n_features):
    test['f'+str(i)]=test[nom_att].apply(lambda x: x[i])
test=test.drop(columns=[nom_att])
columnas=train.columns[1:]

clase1=b'2.0'
clase1_n='allclear'
clase2=b'3.0'
clase2_n='notclear'

train=train[(train[nom_class]==clase1) | (train[nom_class]==clase2)]
test=test[(test[nom_class]==clase1) | (test[nom_class]==clase2)]
train.reset_index(inplace=True,drop=True)
test.reset_index(inplace=True,drop=True)

X_train=train.drop(columns=[nom_class])
y_train=train[nom_class]
X_test=test.drop(columns=[nom_class])
y_test=test[nom_class]
J=n_features


#calculo la long de las time series
len_t=[]
for i in range(train.loc[:,columnas[0]].shape[0]):
    len_t.append(len(train.loc[i,columnas[0]]))
long_t=max(len_t)

def sep_columns(x,i):
    try:
        return x[i]
    except:
        return np.nan

for c in columnas:
    for i in range(max(len_t)):
        X_train[c+'_'+str(i)]=X_train[c].apply(lambda x: sep_columns(x,i))

X_train=X_train.drop(columns=columnas)
X_train_f=X_train.dropna(axis=1,how='any')
y_train=y_train.apply(lambda x: -1 if x==clase1 else 1)

for c in columnas:
    for i in range(max(len_t)):
        X_test[c+'_'+str(i)]=X_test[c].apply(lambda x:sep_columns(x,i))
        
X_test=X_test.drop(columns=columnas)
X_test_f=X_test.dropna(axis=1,how='any')
y_test=y_test.apply(lambda x: -1 if x==clase1 else 1)

M1_aux1=abs(X_train).max()
M1_aux2=abs(X_test).max()
M1_aux={}
for i in range(X_train.shape[1]):
    M1_aux[i]=max(M1_aux1[i],M1_aux2[i])+1e-6
M2_aux1=abs(X_train).min()
M2_aux2=abs(X_test).min()
M2_aux={}
for i in range(X_train.shape[1]):
    M2_aux[i]=min(M2_aux1[i],M2_aux2[i])+1e-6

###creo que solo necesito calcular el max



###MODELO

tipo_clasificador='RF'



if tipo_clasificador=='DT':
    leaves, values, restricciones_right, restricciones_left, model, datos_arbol =classifier.decision_tree(X_train,y_train,X_test,y_test)


elif tipo_clasificador=='RF':
    n_arboles=200
    max_depth=3

    leaves, values, restricciones_right, restricciones_left, model, datos_arbol =classifier.random_forest(X_train,y_train,X_test,y_test,n_arboles,max_depth)


##class

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




individuo=0
#n_curvas=[1,2,3,4]
n_curvas=[1,2,3,4]

n_curvas=[2]

for n in n_curvas:

    if tipo_clasificador=='DT':
        modelo_opt=modelo_opt2_multi.modelo_opt(leaves, values, restricciones_right, restricciones_left, x, y, y_pred, model, datos_arbol)
        x0, curvas, indices,  x_sol, distancia =run2_multi.optimizacion(individuo,n,modelo_opt,leaves, values, restricciones_right, restricciones_left, x, J, y, y_pred,model,datos_arbol)
        

    elif tipo_clasificador=='RF':
        modelo_opt=modelo_opt3_multi.modelo_opt(leaves, values, restricciones_right, restricciones_left, x, y, y_pred, model, datos_arbol)
        x0, curvas, indices,  x_sol, distancia, xi_g =run3_multi.optimizacion(individuo,n,modelo_opt,leaves, values, restricciones_right, restricciones_left, x,J, y, y_pred,model,datos_arbol,M1_aux,M2_aux)


    distancia="{:.3f}".format(distancia)
    dim=x0.shape[0] 
    long_t=int(x0.shape[0]/J)
    #print(x0)
    #print(x_sol)
    #print(curvas)
        
        
    #with open('prueba.txt', 'w') as f:
    #    f.writelines(str(x_sol))

    with open('sol_i='+str(individuo)+'n_curvas'+str(n)+'_'+str(clase1_n)+str(clase2_n)+'pruebaa2.pkl','wb') as f:
        pickle.dump([x0, curvas, indices,  x_sol, distancia,xi_g],f)
        
    plt.clf()
    #axes = plt.gca()
    #axes.set_ylim([-1.7,2]) #-1.7, 2.1
    f = plt.figure(figsize=(20,25))
    #f = plt.figure(figsize=(20,6))
    for i in range(J):
        ax=f.add_subplot(6,4,i+1)
        #ax=f.add_subplot(1,3,i+1)
        ax.plot(x0[i*long_t:(i+1)*long_t-1],'k',label='instance x0')
        for j in range(curvas.shape[0]):
            plt.plot(curvas[j][i*long_t:(i+1)*long_t-1],label='instance '+str(indices[j]))
        plt.plot(x_sol[i*long_t:(i+1)*long_t-1],'r--', label='counterfactual')
        plt.legend(loc='upper left')
        #plt.text(15, -1.5,'distance: '+str(distancia))
    f.savefig('prueba_multi_i='+str(individuo)+'n_curvas'+str(n)+'_'+str(clase1_n)+str(clase2_n)+'pruebaa2.png')

        
    


 
