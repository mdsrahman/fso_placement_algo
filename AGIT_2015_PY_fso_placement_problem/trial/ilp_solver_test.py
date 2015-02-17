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
    #first make the problem instance
    fso_problem = pulp.LpProblem(name = "FSO_PLACEMENT_PROBLEM", sense = pulp.LpMaximize)
    #----task: make x_i variables and set equation (1)-----
    x_index =  [i for i in range(self.adj.number_of_nodes())]
    #----task: make e_ij variables and set equation (2)----
    
    #---task: make y_i variables and set equation (3)----
    
    #---task: make b_ij variables and set equation (4)----
    
    #---task: enforce symmetry by setting equation (5) and (6)-----
    
    #---task: enforce edge-incidence by setting equation (7) and (8)----
    
    #---task: set equation (9) involving y_i and target array T_ij----
    
    #---task: make variables e_si and set equation (10) involving e_si and x_i ----
     
    #---task: make variables b_si and set equation (11) involving b_si and y_i ---
    
    #---task: make variables e_wt and set equation (12)----
    
    #---task: make variables f_si and set equation (13) involving f_si, y_i, N ----
    
    #---task: make variables f_ij and set equation (14) involving f_ij and b_ij ----
    
    #---task: make variables f_jt and set equation (15) involving f_si and f_jt ----
    
    #---task: set equation (16) involving f_ij, f_si, f_jt ------
    
    #---task: set equation (17) involving x_i and n_max ----
    
    #---task: set equation (18) involving set of sinks, b_ij, d_max ----
    
    #---task: set variables g_si and g_jt and set equation (19) -----
    
    #---task: set variables g_ij and set equation (20) involving g_ij, g_si, g_jt ----
    
    #---task: set equation (21) involving g_ij and e_ij ----
    
    #---task: set equation (22) involving N, f_si and x_i ----
    
    #---task: set equation (23) involving N, f_it and x_i ----
    
    #---task: set objective equation (24) involving g_si
    
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