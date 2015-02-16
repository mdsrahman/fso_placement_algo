from pulp import *
total_nodes = 10

'''steps--------
i) 


'''
prob = LpProblem("FSO Placement Problem", LpMinimize)
xval = [i for i in range(total_nodes)]
x = LpVariable.dict("x",xval,0,1,LpContinuous)
for i,v in x.iteritems(): 
  print i,":",v