# FunctionalCE (Counterfactual analysis for functional data)

Code to generate counterfactual explanations when data are functions based on the paper "A new model for counterfactual analysis for functional data" by Emilio Carrizosa, Jasone Ramírez-Ayerbe and Dolores Romero Morales. The paper can be found here: https://link.springer.com/article/10.1007/s11634-023-00563-5 

### Requirements

To run the model, the gurobi solver is required. Free academics licenses are available. 


### Files

To run the experiments for univariate functional data:

* 'classifier.py': to train the classifier
* 'modelo_opt2.py': define the optimization model for a DT using pyomo
* 'run2.py2: solve the optimization problem for a DT
* 'modelo_opt3.py': define the optimization model for a RF using pyomo
* 'run3.py2: solve the optimization problem for a RF
* 'algoritmo2.py': to define x0, and generate all the plots

To run the same experiments but with the DTW distance, the analogous codes are inside the dtw folder. Same for multivariate data. 
