from __future__ import division
from pyomo.environ import *



def modelo_opt(leaves, values, restricciones_right, restricciones_left, x, y, y_pred, model,datos_arbol):

   
    modelo_opt = AbstractModel()

    n_arboles=len(leaves)

    index_list = [] #lista de indices (arbol, leaf)
    for t in range(n_arboles):
       for l in range(leaves[t]):
           index_list.append((t,l))


    modelo_opt.N = Param( within=PositiveIntegers ) #len time series  
    modelo_opt.times = RangeSet( 0,modelo_opt.N-1 ) #conjunto de tiempos unidos
    modelo_opt.J = Param(within=PositiveIntegers) #n de features
    modelo_opt.Jaux=RangeSet(0,modelo_opt.J-1) #para controlar los parametros de cada feature
    modelo_opt.tsolo= RangeSet (0, modelo_opt.N/modelo_opt.J -1) #conjunto de tiempos solo

    modelo_opt.N2 = Param (within=PositiveIntegers) #el numero de vecinos
    modelo_opt.vecinos = RangeSet(0, modelo_opt.N2-1) #conjunto de vecinos
    modelo_opt.S =Param(within=PositiveIntegers) #n de curvas de la combinacion lineal

    modelo_opt.trees= Param (within =PositiveIntegers) #numero de arboles
    modelo_opt.t=RangeSet(0,modelo_opt.trees-1)
    modelo_opt.leaves = Param(modelo_opt.t) #numero de leaves de cada arbol
    modelo_opt.index_list = Set(dimen=2,initialize=index_list)
    modelo_opt.values_leaf= Param (modelo_opt.index_list) 

    #las restricciones 
    modelo_opt.nleft=Param(within=PositiveIntegers) #n de restricciones left 
    modelo_opt.nright=Param (within=PositiveIntegers) #n de restricciones right
    modelo_opt.resleft=RangeSet(0,modelo_opt.nleft-1) 
    modelo_opt.resright= RangeSet(0,modelo_opt.nright-1)
    modelo_opt.left=Param(modelo_opt.resleft,within=Any)
    modelo_opt.right=Param(modelo_opt.resright,within=Any)


    modelo_opt.M1=Param(modelo_opt.times,within=Reals) #m1 grande
    modelo_opt.M2=Param(modelo_opt.times,within=Reals) #m2 grande #Positive reals
    modelo_opt.M3=Param(modelo_opt.times,within=Reals) #la m3 para l0
    modelo_opt.M4=Param(within=Reals) #la m4 para la l0 de las curvas
    modelo_opt.epsi=Param(within=PositiveReals) #epsilon


    modelo_opt.y= Param( within=Integers) #valor de y=-y0
    modelo_opt.x0=Param(modelo_opt.times) #la instancia x0 
    modelo_opt.vec=Param (modelo_opt.vecinos,modelo_opt.times) 


    #variables
  
    modelo_opt.x = Var( modelo_opt.times, within=Reals) 
    
    modelo_opt.alphas0=Var(modelo_opt.Jaux,bounds=(0,1))
    modelo_opt.alphas=Var(modelo_opt.Jaux,modelo_opt.vecinos,bounds=(0,1))

    #variable para vez en que rama cae de cada arbol 

    modelo_opt.z=Var(modelo_opt.index_list,within=Binary)
    modelo_opt.D=Var(modelo_opt.t)

    modelo_opt.xi = Var (modelo_opt.vecinos, within=Binary)
    modelo_opt.xi_g = Var(modelo_opt.Jaux, within =Binary) 




    #distancia l2+l0
    def obj_rule(modelo_opt):
         return 0.5*sum((modelo_opt.x0[n]-modelo_opt.x[n])**2 for n in modelo_opt.times)+sum(modelo_opt.xi_g[J] for J in modelo_opt.Jaux)
    modelo_opt.obj = Objective( rule=obj_rule )

    #RangeSet is 1-based RangeSet(5)=[1,2,3,4,5]

    #restricciones

    def path_left(modelo_opt,s):
        return modelo_opt.x[modelo_opt.left[s][2]]-(modelo_opt.M1[modelo_opt.left[s][2]]-modelo_opt.left[s][3])*(1-modelo_opt.z[modelo_opt.left[s][0],modelo_opt.left[s][1]])+modelo_opt.epsi<=modelo_opt.left[s][3]
    modelo_opt.pathleft= Constraint(modelo_opt.resleft,rule=path_left)

    def path_right(modelo_opt,s):
        return modelo_opt.x[modelo_opt.right[s][2]]+(modelo_opt.M1[modelo_opt.right[s][2]]+modelo_opt.right[s][3])*(1-modelo_opt.z[modelo_opt.right[s][0],modelo_opt.right[s][1]])-modelo_opt.epsi>=modelo_opt.right[s][3]
    modelo_opt.pathright= Constraint(modelo_opt.resright,rule=path_right)



    def one_path(modelo_opt,t):
        return sum(modelo_opt.z[t,l] for l in RangeSet(0,modelo_opt.leaves[t]-1))==1.0
    modelo_opt.path=Constraint(modelo_opt.t,rule=one_path)

    def def_salida(modelo_opt,t):
        return modelo_opt.D[t]==sum(modelo_opt.values_leaf[t,l]*modelo_opt.z[t,l] for l in RangeSet(0,modelo_opt.leaves[t]-1))
    modelo_opt.salida=Constraint(modelo_opt.t,rule=def_salida)


    def def_clase(modelo_opt):
        return modelo_opt.y*(sum(modelo_opt.D[t] for t in modelo_opt.t))>=0
    modelo_opt.clase=Constraint(rule=def_clase)


    def def_counterfs(modelo_opt,n,J):
        return modelo_opt.x[n+J*modelo_opt.N/modelo_opt.J]==modelo_opt.alphas0[J]*modelo_opt.x0[n+J*modelo_opt.N/modelo_opt.J]+sum(modelo_opt.alphas[J,j]*modelo_opt.vec[j,n+J*modelo_opt.N/modelo_opt.J] for j in modelo_opt.vecinos)
    modelo_opt.counterfs=Constraint(modelo_opt.tsolo, modelo_opt.Jaux, rule=def_counterfs)


    def aux_coefs(modelo_opt,J):
        return modelo_opt.alphas0[J]+sum(modelo_opt.alphas[J,j] for j in modelo_opt.vecinos)==1
    modelo_opt.coefs=Constraint(modelo_opt.Jaux,rule=aux_coefs)


    def aux_l0_1(modelo_opt,J,n):
        return -modelo_opt.M4*modelo_opt.xi[n]<=(modelo_opt.alphas[J,n])
    modelo_opt.auxl0_1=Constraint(modelo_opt.Jaux,modelo_opt.vecinos,rule=aux_l0_1)


    def aux_l0_2(modelo_opt,J,n):
        return modelo_opt.alphas[J,n]<=modelo_opt.xi[n]*modelo_opt.M4
    modelo_opt.auxl0_2=Constraint(modelo_opt.Jaux,modelo_opt.vecinos,rule=aux_l0_2)

    def n_coef(modelo_opt):
        return sum(modelo_opt.xi[n] for n in modelo_opt.vecinos)==modelo_opt.S
    modelo_opt.ncoef=Constraint(rule=n_coef)

    def aux_l0g1(modelo_opt,n,J):
        return -modelo_opt.M3[n+J*modelo_opt.N/modelo_opt.J]*modelo_opt.xi_g[J]<=(modelo_opt.x[n+J*modelo_opt.N/modelo_opt.J]-modelo_opt.x0[n+J*modelo_opt.N/modelo_opt.J])
    modelo_opt.auxl0g1=Constraint(modelo_opt.tsolo,modelo_opt.Jaux,rule=aux_l0g1)

    def aux_l0g2(modelo_opt,n,J):
        return (modelo_opt.x[n+J*modelo_opt.N/modelo_opt.J]-modelo_opt.x0[n+J*modelo_opt.N/modelo_opt.J])<=modelo_opt.xi_g[J]*modelo_opt.M3[n+J*modelo_opt.N/modelo_opt.J]
    modelo_opt.auxl0g2=Constraint(modelo_opt.tsolo, modelo_opt.Jaux,rule=aux_l0g2)

    return modelo_opt
