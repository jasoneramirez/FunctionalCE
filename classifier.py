
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np





def decision_tree(X_train,y_train,X_test,y_test):

    model=DecisionTreeClassifier(random_state=0)
    model.fit(X_train, y_train)
    acc=model.score(X_test,y_test)

    parametros_arbol={}
    parametros_arbol['n_nodes']=model.tree_.node_count
    parametros_arbol['children_left'] = model.tree_.children_left
    parametros_arbol['children_right'] = model.tree_.children_right
    parametros_arbol['feature'] = model.tree_.feature
    parametros_arbol['threshold'] = model.tree_.threshold
    
    def nodo_hijo(x):
        if x in datos_arbol['padres_l'].keys():
            nodo_hijo=datos_arbol['padres_l'][x]
            donde='l'
        else:
            nodo_hijo=datos_arbol['padres_r'][x]
            donde='r'
        return([nodo_hijo,donde])

    def lista(x):
        k=x
        dict={'left':[],'right':[]}
        while k!=0:
            nodo=nodo_hijo(k)
            if nodo[1]=='l':
                dict['left'].append(nodo[0])
                k=nodo[0]
            else:
                dict['right'].append(nodo[0])
                k=nodo[0]
        return dict   

    def variable_umbral(leaf,direccion):
        d=datos_arbol['caminos'][leaf]
        res=[]
        if direccion=='left':
            s=d['left']
            for i in s:
                res.append([parametros_arbol['feature'][i],parametros_arbol['threshold'][i]])     
        else:
            s=d['right']
            for i in s:
                res.append([parametros_arbol['feature'][i],parametros_arbol['threshold'][i]]) 
        return res

    datos_arbol={}
    classes=np.array([-1,1])
    is_leaves = np.zeros(shape=parametros_arbol['n_nodes'], dtype=bool)
    is_leaves=parametros_arbol['children_left']==parametros_arbol['children_right']
    datos_arbol['index_leaves']=np.where(is_leaves==True)[0].tolist()
    datos_arbol['value_leaves']=[classes[np.argmax(i)] for i in model.tree_.value[datos_arbol['index_leaves']]]
    datos_arbol['index_splits']=np.where(is_leaves==False)[0].tolist()
    datos_arbol['n_leaves']=len(datos_arbol['index_leaves'])
    padres_l={}
    padres_r={}
    for j in range(parametros_arbol['n_nodes']):
        if len(np.where(parametros_arbol['children_left']==j)[0])!=0:
            padres_l[j]=np.where(parametros_arbol['children_left']==j)[0][0]
    for j in range(parametros_arbol['n_nodes']):
        if len(np.where(parametros_arbol['children_right']==j)[0])!=0:
            padres_r[j]=np.where(parametros_arbol['children_right']==j)[0][0]
    datos_arbol['padres_l']=padres_l
    datos_arbol['padres_r']=padres_r
    datos_arbol['caminos']={}
    for k in datos_arbol['index_leaves']:
        datos_arbol['caminos'][k]=lista(k)
        
    restricciones={}
    restricciones={'left':[],'right':[]}
    for i in datos_arbol['index_leaves']:
        restricciones['left'].append(variable_umbral(i,'left'))
        restricciones['right'].append(variable_umbral(i,'right'))
        
    restricciones_left=[]
    l=restricciones['left']
    for j in range(datos_arbol['n_leaves']):
        l2=l[j]
        for k in range(len(l2)):
            l3=l2[k]
            restricciones_left.append((j,l3[0],l3[1]))
            
    restricciones_right=[]
    l=restricciones['right']
    for j in range(datos_arbol['n_leaves']):
        l2=l[j]
        for k in range(len(l2)):
            l3=l2[k]
            restricciones_right.append((j,l3[0],l3[1]))
            
    values={}
    values=datos_arbol['value_leaves']
    
    leaves=[]
    leaves.append(datos_arbol['n_leaves'])


    return leaves, values, restricciones_right, restricciones_left, model, datos_arbol



def random_forest(X_train,y_train,X_test,y_test,n_arboles,max_depth):

    model=RandomForestClassifier(n_estimators=n_arboles,random_state=0,max_depth=max_depth)
    model.fit(X_train, y_train)

    arboles=[] #lista de todos los arboles
    for t in range(n_arboles):
        arboles.append(model.estimators_[t]) 

    parametros_arbol={}
    for t in range(len(arboles)):
        parametros_arbol[t]={}
        parametros_arbol[t]['n_nodes']=arboles[t].tree_.node_count
        parametros_arbol[t]['children_left'] = arboles[t].tree_.children_left
        parametros_arbol[t]['children_right'] = arboles[t].tree_.children_right
        parametros_arbol[t]['feature'] = arboles[t].tree_.feature
        parametros_arbol[t]['threshold'] = arboles[t].tree_.threshold

    def nodo_hijo(x,t):
        if x in datos_arbol[t]['padres_l'].keys():
            nodo_hijo=datos_arbol[t]['padres_l'][x]
            donde='l'
        else:
            nodo_hijo=datos_arbol[t]['padres_r'][x]
            donde='r'
        return([nodo_hijo,donde])

    def lista(x,t):
        k=x
        dict={'left':[],'right':[]}
        while k!=0:
            nodo=nodo_hijo(k,t)
            if nodo[1]=='l':
                dict['left'].append(nodo[0])
                k=nodo[0]
            else:
                dict['right'].append(nodo[0])
                k=nodo[0]
        return dict   

    def variable_umbral(leaf,direccion,t):
        d=datos_arbol[t]['caminos'][leaf]
        res=[]
        if direccion=='left':
            s=d['left']
            for i in s:
                res.append([parametros_arbol[t]['feature'][i],parametros_arbol[t]['threshold'][i]])     
        else:
            s=d['right']
            for i in s:
                res.append([parametros_arbol[t]['feature'][i],parametros_arbol[t]['threshold'][i]]) 
        return res
    

    datos_arbol={}
    classes=np.array([-1,1])
    for t in range(len(arboles)):
        datos_arbol[t]={}
        is_leaves = np.zeros(shape=parametros_arbol[t]['n_nodes'], dtype=bool)
        is_leaves=parametros_arbol[t]['children_left']==parametros_arbol[t]['children_right']
        datos_arbol[t]['index_leaves']=np.where(is_leaves==True)[0].tolist()
        datos_arbol[t]['value_leaves']=[classes[np.argmax(i)] for i in arboles[t].tree_.value[datos_arbol[t]['index_leaves']]]
        datos_arbol[t]['index_splits']=np.where(is_leaves==False)[0].tolist()
        datos_arbol[t]['n_leaves']=len(datos_arbol[t]['index_leaves'])
        padres_l={}
        padres_r={}
        for j in range(parametros_arbol[t]['n_nodes']):
            if len(np.where(parametros_arbol[t]['children_left']==j)[0])!=0:
                padres_l[j]=np.where(parametros_arbol[t]['children_left']==j)[0][0]
        for j in range(parametros_arbol[t]['n_nodes']):
            if len(np.where(parametros_arbol[t]['children_right']==j)[0])!=0:
                padres_r[j]=np.where(parametros_arbol[t]['children_right']==j)[0][0]
        datos_arbol[t]['padres_l']=padres_l
        datos_arbol[t]['padres_r']=padres_r
        datos_arbol[t]['caminos']={}
        for k in datos_arbol[t]['index_leaves']:
            datos_arbol[t]['caminos'][k]=lista(k,t)

    restricciones={}
    for t in range(len(arboles)):
        restricciones[t]={'left':[],'right':[]}
        for i in datos_arbol[t]['index_leaves']:
            restricciones[t]['left'].append(variable_umbral(i,'left',t))
            restricciones[t]['right'].append(variable_umbral(i,'right',t))

    restricciones_left=[]
    for i in range(n_arboles):
        l=restricciones[i]['left']
        for j in range(datos_arbol[i]['n_leaves']):
            l2=l[j]
            for k in range(len(l2)):
                l3=l2[k]
                restricciones_left.append((i,j,l3[0],l3[1]))
            
    restricciones_right=[]
    for i in range(n_arboles):
        l=restricciones[i]['right']
        for j in range(datos_arbol[i]['n_leaves']):
            l2=l[j]
            for k in range(len(l2)):
                l3=l2[k]
                restricciones_right.append((i,j,l3[0],l3[1]))
    

    values={}
    for i in range(n_arboles):
        values[i]=datos_arbol[i]['value_leaves']

    leaves=[]
    for i in range(n_arboles):
        leaves.append(datos_arbol[i]['n_leaves'])


    return leaves, values, restricciones_right, restricciones_left, model, datos_arbol