import pulp
import networkx as nx
import numpy as np
import random

'''steps--------
pre: build random nx grap instance- targets, adj graphs
i) build variable dicts
ii) build linear equations
iii) call the solver 

'''
class ILP_Relaxed():
  def __init__(self):
    print "ILP_Relaxed initialized...."
    self.adj = None
    self.T = None
    self.sinks =None
    
  def make_problem_instance(self,
                            num_node, 
                            num_edge, 
                            short_edge_fraction, 
                            num_sink,
                            num_target,
                            max_node_to_target_association
                            ):
    #-----first make a dense connected graph-----------
    while True:
      self.adj = nx.dense_gnm_random_graph(num_node,num_edge)
      if nx.is_connected(self.adj):
        break
      
    self.adj.graph["name"]="adj_graph"
    nx.set_edge_attributes(self.adj, 'con_type', 'long') #by default all edges are long edges..

    #---now randomly choose short edges (<100m)------
    total_short_edges = int(round(short_edge_fraction * self.adj.number_of_edges(), 0))
    short_edges = random.sample(self.adj.edges(), total_short_edges )
    for u,v in short_edges:
      self.adj[u][v]['con_type'] = 'short'
    
    for edge in self.adj.edges(data = True):
      print edge

    #------now randomly choose sinks-----------------------
    self.sinks = random.sample(self.adj.nodes(), num_sink)

    #--------now randomly hooks targets to source nodes-------
    #T is the matrix of size num_tarets x num_nodes
    self.T =  np.zeros(shape = (num_target, self.adj.number_of_nodes()), dtype = int)
    for t in range(num_target):
      num_of_assoc_sources = random.randint(1,max_node_to_target_association)
      assoc_sources = random.sample(self.adj.nodes(),num_of_assoc_sources)
      for n in assoc_sources:
        self.T[t][n] = 1
    print self.T
  
  def build_lp_instance(self):
    
    return

   
   
if __name__ == '__main__':
  ilp = ILP_Relaxed()
  ilp.make_problem_instance(num_node = 10, 
                            num_edge = 25, 
                            short_edge_fraction = 0.4, 
                            num_sink = 2,
                            num_target = 13,
                            max_node_to_target_association = 3
                            )
  '''
  prob = LpProblem("FSO Placement Problem", LpMinimize)
  xval = [i for i in range(total_nodes)]
  x = LpVariable.dict("x",xval,0,1,LpContinuous)
  for i,v in x.iteritems(): 
    print i,":",v
    '''