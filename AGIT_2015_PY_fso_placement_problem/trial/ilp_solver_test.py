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
    fso_placement_prob = pulp.LpProblem(name = "FSO_PLACEMENT_PROBLEM", sense = pulp.LpMaximize)
    fso_placement_prob += 0, "Arbitrary Objective Function" #<_--!!!!!comeback with equation (24)-----!!!!
    
    
    #----task: make x_i variables and set equation (1)-----
    x = pulp.LpVariable.dicts(name='x', 
                              indexs = self.adj.nodes(), 
                              lowBound=0, 
                              upBound =1, 
                              cat = pulp.LpContinuous)
    print "DEBUG: x_i variables created"
    for i,v in x.iteritems(): 
      print "DEBUG: ",i,":",v
    #***********end_of_task**************************************
    
    #----task: make e_ij variables and set equation (2)----
    e = pulp.LpVariable.dicts(name='e', 
                              indexs = (self.adj.nodes(),self.adj.nodes()), 
                              lowBound=0, 
                              upBound =1, 
                              cat = pulp.LpContinuous)
    print "DEBUG: e_ij variables created"
    for i,e_item in e.iteritems():
      print "DEBUG: ",i,":",e_item
    #***********end_of_task**************************************
    
    #---task: make y_i variables and set equation (3)----
    y = pulp.LpVariable.dicts(name='y', 
                              indexs = self.adj.nodes(), 
                              lowBound=0, 
                              #upBound =1, 
                              cat = pulp.LpContinuous)
    print "DEBUG: y_i variables created"
    for i,v in y.iteritems(): 
      print "DEBUG: ",i,":",v
      
    for i,v in y.iteritems():
      fso_placement_prob += y[i] <= x[i], "eqn_3_("+str(i)+")"
    #***********end_of_task**************************************
    
    #---task: make b_ij variables and set equation (4)----
    b = pulp.LpVariable.dicts(name='b', 
                              indexs = (self.adj.nodes(),self.adj.nodes()), 
                              lowBound=0, 
                              upBound =1, 
                              cat = pulp.LpContinuous)
    print "DEBUG: b_ij variables created"
    for i,b_item in b.iteritems():
      print "DEBUG: ",i,":",b_item
    for i in self.adj.nodes():
      for j in self.adj.nodes():
        fso_placement_prob += b[i][j] <= e[i][j], "eqn_4_("+str(i)+","+str(j)+")"
    #***********end_of_task**************************************
    
    #---task: enforce symmetry by setting equation (5) and (6)-----
    for i in self.adj.nodes():
      for j in self.adj.nodes():
        fso_placement_prob += e[i][j] == e[j][i], "eqn_5_("+str(i)+","+str(j)+")"
        fso_placement_prob += b[i][j] == b[j][i], "eqn_6_("+str(i)+","+str(j)+")"
    #***********end_of_task**************************************
    #---task: enforce edge-incidence by setting equation (7) and (8)----
    for i in self.adj.nodes():
      for j in self.adj.nodes():
        fso_placement_prob += e[i][j] <= 0.5 * (x[i] + x[j] ), "eqn_7_("+str(i)+","+str(j)+")"
        fso_placement_prob += b[i][j] <= 0.5 * (y[i] + y[j] ), "eqn_8_("+str(i)+","+str(j)+")"
    #***********end_of_task**************************************
    #---task: set equation (9) involving y_i and target array T_ij----
    num_targets, num_nodes = self.T.shape
    for t in range(num_targets):
      fso_placement_prob += pulp.lpSum( y[i] * self.T[t][i] for i in self.adj.nodes() ) >= 1,\
                           "eqn_9_("+str(t)+")"
    #***********end_of_task**************************************
    #---task: make variables e_si and set equation (10) involving e_si and x_i ----
    
    #***********end_of_task**************************************
    #---task: make variables b_si and set equation (11) involving b_si and y_i ---
    
    #***********end_of_task**************************************
    #---task: make variables e_wt and set equation (12)----
    
    #***********end_of_task**************************************
    #---task: make variables f_si and set equation (13) involving f_si, y_i, N ----
    
    #***********end_of_task**************************************
    #---task: make variables f_ij and set equation (14) involving f_ij and b_ij ----
    
    #***********end_of_task**************************************
    #---task: make variables f_jt and set equation (15) involving f_si and f_jt ----
    
    #***********end_of_task**************************************
    #---task: set equation (16) involving f_ij, f_si, f_jt ------
    
    #***********end_of_task**************************************
    #---task: set equation (17) involving x_i and n_max ----
    
    #***********end_of_task**************************************
    #---task: set equation (18) involving set of sinks, b_ij, d_max ----
    
    #***********end_of_task**************************************
    #---task: set variables g_si and g_jt and set equation (19) -----
    
    #***********end_of_task**************************************
    #---task: set variables g_ij and set equation (20) involving g_ij, g_si, g_jt ----
    
    #***********end_of_task**************************************
    #---task: set equation (21) involving g_ij and e_ij ----
    
    #***********end_of_task**************************************
    #---task: set equation (22) involving N, f_si and x_i ----
    
    #***********end_of_task**************************************
    #---task: set equation (23) involving N, f_it and x_i ----
    
    #***********end_of_task**************************************
    #---task: set objective equation (24) involving g_si
    
    #***********end_of_task**************************************
    fso_placement_prob.writeLP("fso_placement_ILP_relaxed.lp")
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
  ilp.build_lp_instance()
  '''
  prob = LpProblem("FSO Placement Problem", LpMinimize)
  xval = [i for i in range(total_nodes)]
  x = LpVariable.dict("x",xval,0,1,LpContinuous)
  for i,v in x.iteritems(): 
    print i,":",v
    '''