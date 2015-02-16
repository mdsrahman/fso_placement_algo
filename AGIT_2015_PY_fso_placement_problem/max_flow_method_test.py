import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path
G = nx.Graph()
G.add_edge('x','a', capacity=5)
G.add_edge('x','b', capacity=2.5)
G.add_edge('a','c', capacity=2.5)
G.add_edge('b','c', capacity=2.5)
G.add_edge('b','d', capacity=2.5)
G.add_edge('d','e', capacity=2.5)
G.add_edge('c','y', capacity=2.5)
G.add_edge('e','y', capacity=2.5)
K=G
R = shortest_augmenting_path(K,'x','y')
flow_value = R.graph['flow_value']
print flow_value
'''
flow_dict = R.graph['flow_dict']

for i in flow_dict:
  for j in flow_dict[i]:
    print "(",i,",",j,"): ",flow_dict[i][j]
'''
print "residual graph------"
for e in R.edges(data=True):
  print e
#---------SECOND PASS----------------


R = shortest_augmenting_path(R,'e','a')
flow_value = R.graph['flow_value']
print flow_value
'''
flow_dict = R.graph['flow_dict']

for i in flow_dict:
  for j in flow_dict[i]:
    print "(",i,",",j,"): ",flow_dict[i][j]
'''
print "residual graph------"
for e in R.edges(data=True):
  print e

#--------THIRD PASS----------------
R = shortest_augmenting_path(R,'b','e')
flow_value = R.graph['flow_value']
print flow_value
'''
flow_dict = R.graph['flow_dict']

for i in flow_dict:
  for j in flow_dict[i]:
    print "(",i,",",j,"): ",flow_dict[i][j]
'''
print "residual graph------"
for e in R.edges(data=True):
  print e


