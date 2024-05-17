from __future__ import division
from pyomo.environ import *



def modelo_opt(leaves, values, restricciones_right, restricciones_left, x, y, y_pred, model,datos_arbol):

   
    modelo_opt = AbstractModel()

    n_arboles=len(leaves)

    index_list = [] #lista de indices (arbol, leaf)
    for t in range(n_arboles):
       for l in range(leaves[t]):
           index_list.append((t,l))


    modelo_opt.N = Param( within=PositiveIntegers ) 
    modelo_opt.times = RangeSet( 0,modelo_opt.N-1 ) 
    modelo_opt.N2 = Param (within=PositiveIntegers) 
    modelo_opt.vecinos = RangeSet(0, modelo_opt.N2-1) 
    modelo_opt.S =Param(within=PositiveIntegers) 

    modelo_opt.trees= Param (within =PositiveIntegers) 
    modelo_opt.t=RangeSet(0,modelo_opt.trees-1)
    modelo_opt.leaves = Param(modelo_opt.t) 
    modelo_opt.index_list = Set(dimen=2,initialize=index_list)
    modelo_opt.values_leaf= Param (modelo_opt.index_list) 

    #las restricciones 
    modelo_opt.nleft=Param(within=PositiveIntegers) 
    modelo_opt.nright=Param (within=PositiveIntegers) 
    modelo_opt.resleft=RangeSet(0,modelo_opt.nleft-1) 
    modelo_opt.resright= RangeSet(0,modelo_opt.nright-1)
    modelo_opt.left=Param(modelo_opt.resleft,within=Any)
    modelo_opt.right=Param(modelo_opt.resright,within=Any)


    modelo_opt.M1=Param(modelo_opt.times,within=PositiveReals) 
    modelo_opt.M2=Param(modelo_opt.times,within=PositiveReals) 
    modelo_opt.M3=Param(within=PositiveReals) 
    modelo_opt.epsi=Param(within=PositiveReals) 


    modelo_opt.y= Param( within=Integers) # y=-y0
    modelo_opt.x0=Param(modelo_opt.times) 
    modelo_opt.vec=Param (modelo_opt.vecinos,modelo_opt.times) 


    #variables
  
    modelo_opt.x = Var( modelo_opt.times, within=Reals) 
    modelo_opt.alpha0=Var(within=PositiveReals, bounds=(0,1))
    modelo_opt.alpha=Var(modelo_opt.vecinos, within=NonNegativeReals, bounds=(0,1)) 

    #variable para vez en que rama cae de cada arbol 

    modelo_opt.z=Var(modelo_opt.index_list,within=Binary)
    modelo_opt.D=Var(modelo_opt.t)
    modelo_opt.xi = Var (modelo_opt.vecinos, within=Binary) #auxiliar l0 para cada alpha_j



    #distancia l2
    def obj_rule(modelo_opt):
         return sum((modelo_opt.x0[n]-modelo_opt.x[n])**2 for n in modelo_opt.times)
    modelo_opt.obj = Objective( rule=obj_rule )

    #RangeSet is 1-based RangeSet(5)=[1,2,3,4,5]

    #restricciones

    def path_left(modelo_opt,s):
        return modelo_opt.x[modelo_opt.left[s][2]]-(modelo_opt.M1[modelo_opt.left[s][2]])*(1-modelo_opt.z[modelo_opt.left[s][0],modelo_opt.left[s][1]])+modelo_opt.epsi<=modelo_opt.left[s][3]
    modelo_opt.pathleft= Constraint(modelo_opt.resleft,rule=path_left)

    def path_right(modelo_opt,s):
        return modelo_opt.x[modelo_opt.right[s][2]]+(modelo_opt.M2[modelo_opt.right[s][2]])*(1-modelo_opt.z[modelo_opt.right[s][0],modelo_opt.right[s][1]])-modelo_opt.epsi>=modelo_opt.right[s][3]
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


    def def_counterf(modelo_opt,n):
        return modelo_opt.x[n]==modelo_opt.alpha0*modelo_opt.x0[n]+sum(modelo_opt.alpha[j]*modelo_opt.vec[j,n] for j in modelo_opt.vecinos)
    modelo_opt.counterf=Constraint(modelo_opt.times,rule=def_counterf)


    def aux_coef(modelo_opt):
        return modelo_opt.alpha0+sum(modelo_opt.alpha[j] for j in modelo_opt.vecinos)==1 #suma de los coef=1
    modelo_opt.coefs=Constraint(rule=aux_coef)


    def aux_l01(modelo_opt,n):
           return -modelo_opt.M3*modelo_opt.xi[n]<=(modelo_opt.alpha[n])
    modelo_opt.auxl01=Constraint(modelo_opt.vecinos,rule=aux_l01)

    def aux_l02(modelo_opt,n):
        return modelo_opt.alpha[n]<=modelo_opt.xi[n]*modelo_opt.M3
    modelo_opt.auxl02=Constraint(modelo_opt.vecinos,rule=aux_l02)

    def n_coef(modelo_opt):
        return sum(modelo_opt.xi[n] for n in modelo_opt.vecinos)<=modelo_opt.S
    modelo_opt.ncoef=Constraint(rule=n_coef)

    return modelo_opt
