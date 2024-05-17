from __future__ import division
from pyomo.environ import *





def modelo_opt(leaves, values, restricciones_right, restricciones_left, x, y, y_pred, model,datos_arbol):

   
    modelo_opt = AbstractModel()

    modelo_opt.N = Param( within=PositiveIntegers ) #len de cada time series
    modelo_opt.t = RangeSet( 0,modelo_opt.N-1 ) #conjunto de tiempos unidos
    modelo_opt.J = Param(within=PositiveIntegers) #n de features
    modelo_opt.Jaux=RangeSet(0,modelo_opt.J-1) #para controlar los parametros de cada feature
    modelo_opt.tsolo= RangeSet (0, modelo_opt.N/modelo_opt.J -1)



    #modelo_opt.t_d=RangeSet (0, modelo_opt.N-2) #conjunto de tiempos de las derivadas
    #modelo_opt.t_d2=RangeSet(0, modelo_opt.N-3) #tiempos segunda derivada (sumar integral primera derivada)
    #modelo_opt.t_d3=RangeSet(0,modelo_opt.N-4) #para sumar la integral segunda derivada
    modelo_opt.N2 = Param (within=PositiveIntegers) #el numero de vecinos
    modelo_opt.vecinos = RangeSet(0, modelo_opt.N2-1) #conjunto de vecinos
    modelo_opt.S =Param(within=PositiveIntegers) #n de curvas de la combinacion lineal
    #modelo_opt.coef =RangeSet(0, modelo_opt.N2)
    modelo_opt.leaves = Param(within=PositiveIntegers) #numero de leaves del arbol
    modelo_opt.l=RangeSet(0,modelo_opt.leaves-1)
    modelo_opt.values_leaf= Param (modelo_opt.l) 

    #las restricciones 
    modelo_opt.nleft=Param(within=PositiveIntegers) #n de restricciones left 
    modelo_opt.nright=Param (within=PositiveIntegers) #n de restricciones right
    modelo_opt.resleft=RangeSet(0,modelo_opt.nleft-1) 
    modelo_opt.resright= RangeSet(0,modelo_opt.nright-1)
    modelo_opt.left=Param(modelo_opt.resleft,within=Any)
    modelo_opt.right=Param(modelo_opt.resright,within=Any)


    modelo_opt.M1=Param(within=PositiveReals) #m1 grande
    modelo_opt.M2=Param(within=Reals) #m2 grande #Positive reals
    modelo_opt.M3=Param(within=PositiveReals) #la m3 para l0
    modelo_opt.epsi=Param(within=PositiveReals) #epsilon


    modelo_opt.y= Param( within=Integers) #valor de y=-y0
    modelo_opt.x0=Param(modelo_opt.t) #la instancia x0 
    #modelo_opt.x0_d=Param(modelo_opt.t_d) #la derivada de la instancia x0
    #modelo_opt.x0_d2=Param(modelo_opt.t_d2) #segunda derivada
    modelo_opt.vec=Param (modelo_opt.vecinos, modelo_opt.t) #el vecino de la otra clase




    #variables
  
    modelo_opt.x = Var( modelo_opt.t, within=Reals) 


    modelo_opt.alphas0=Var(modelo_opt.Jaux,bounds=(0,1))
    modelo_opt.alphas=Var(modelo_opt.Jaux,modelo_opt.vecinos,bounds=(0,1))


    


    modelo_opt.z=Var(modelo_opt.l,within=Binary)


    #para linealizar l0
    modelo_opt.xi = Var (modelo_opt.vecinos, within=Binary) #auxiliar l0 para cada alpha (los vecinos escogidos son los mismos)
    modelo_opt.xi_g = Var(modelo_opt.Jaux, within =Binary) #auxiliar l0 para controlar feautures que se mueven



    #distancia l2
    #def obj_rule(modelo_opt):
    #     return sum((modelo_opt.x0[n]-modelo_opt.x[n])**2 for n in modelo_opt.t)
    #modelo_opt.obj = Objective( rule=obj_rule )

    #distancia l2+l0
    def obj_rule(modelo_opt):
         return 0.1*sum((modelo_opt.x0[n]-modelo_opt.x[n])**2 for n in modelo_opt.t)+sum(modelo_opt.xi_g[J] for J in modelo_opt.Jaux)
    modelo_opt.obj = Objective( rule=obj_rule )


    #distancia con path definido
    #def obj_rule(modelo_opt):
   #     return sum([(modelo_opt.x0[n]-modelo_opt.x[m])**2 for (n,m) in modelo_opt.path_index])
   # modelo_opt.obj=Objective(rule=obj_rule)

    #RangeSet is 1-based RangeSet(5)=[1,2,3,4,5]

    #restricciones

    def path_left(modelo_opt,s):
        return modelo_opt.x[modelo_opt.left[s][1]]-(modelo_opt.M1)*(1-modelo_opt.z[modelo_opt.left[s][0]])+modelo_opt.epsi<=modelo_opt.left[s][2]
    modelo_opt.pathleft= Constraint(modelo_opt.resleft,rule=path_left)

    def path_right(modelo_opt,s):
        return modelo_opt.x[modelo_opt.right[s][1]]+(modelo_opt.M2)*(1-modelo_opt.z[modelo_opt.right[s][0]])-modelo_opt.epsi>=modelo_opt.right[s][2]
    modelo_opt.pathright= Constraint(modelo_opt.resright,rule=path_right)



    def one_path(modelo_opt):
        return sum(modelo_opt.z[l] for l in modelo_opt.l)==1.0
    modelo_opt.path=Constraint(rule=one_path)


    def def_clase(modelo_opt):
        return modelo_opt.y*(sum(modelo_opt.values_leaf[l]*modelo_opt.z[l] for l in modelo_opt.l))>=0
    modelo_opt.clase=Constraint(rule=def_clase)



    def def_counterfs(modelo_opt,n,J):
        return modelo_opt.x[n+J*modelo_opt.N/modelo_opt.J]==modelo_opt.alphas0[J]*modelo_opt.x0[n+J*modelo_opt.N/modelo_opt.J]+sum(modelo_opt.alphas[J,j]*modelo_opt.vec[j,n+J*modelo_opt.N/modelo_opt.J] for j in modelo_opt.vecinos)
    modelo_opt.counterfs=Constraint(modelo_opt.tsolo, modelo_opt.Jaux, rule=def_counterfs)


    def aux_coefs(modelo_opt,J):
        return modelo_opt.alphas0[J]+sum(modelo_opt.alphas[J,j] for j in modelo_opt.vecinos)==1
    modelo_opt.coefs=Constraint(modelo_opt.Jaux,rule=aux_coefs)

 
    def aux_l0_1(modelo_opt,J,n):
        return -modelo_opt.M3*modelo_opt.xi[n]<=(modelo_opt.alphas[J,n])
    modelo_opt.auxl0_1=Constraint(modelo_opt.Jaux,modelo_opt.vecinos,rule=aux_l0_1)


    def aux_l0_2(modelo_opt,J,n):
        return modelo_opt.alphas[J,n]<=modelo_opt.xi[n]*modelo_opt.M3
    modelo_opt.auxl0_2=Constraint(modelo_opt.Jaux,modelo_opt.vecinos,rule=aux_l0_2)

  

    def n_coef(modelo_opt):
        return sum(modelo_opt.xi[n] for n in modelo_opt.vecinos)==modelo_opt.S
    modelo_opt.ncoef=Constraint(rule=n_coef)



    def aux_l0g1(modelo_opt,n,J):
        return -modelo_opt.M3*modelo_opt.xi_g[J]<=(modelo_opt.x[n+J*modelo_opt.N/modelo_opt.J]-modelo_opt.x0[n+J*modelo_opt.N/modelo_opt.J])
    modelo_opt.auxl0g1=Constraint(modelo_opt.tsolo,modelo_opt.Jaux,rule=aux_l0g1)

    def aux_l0g2(modelo_opt,n,J):
        return (modelo_opt.x[n+J*modelo_opt.N/modelo_opt.J]-modelo_opt.x0[n+J*modelo_opt.N/modelo_opt.J])<=modelo_opt.xi_g[J]*modelo_opt.M3
    modelo_opt.auxl0g2=Constraint(modelo_opt.tsolo, modelo_opt.Jaux,rule=aux_l0g2)



    return modelo_opt


