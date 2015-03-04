import pulp
import networkx as nx
import numpy as np
import random
import sys
from collections import defaultdict
from copy import deepcopy
'''steps--------
pre: build random nx grap instance- targets, adj graphs
i) build variable dicts
ii) build linear equations
iii) call the solver 

'''
class ILP_Relaxed():
  def __init__(self,
               nmax,
               dmax,
               adj,
               T_N,
               sinks
               ):
    #print "ILP_Relaxed initialized...."
    self.adj = deepcopy(adj)
    self.T_N = deepcopy(T_N)
    self.sinks  = deepcopy(sinks)
    self.nmax = nmax
    self.dmax = dmax
    
  def make_problem_instance(self,
                            num_node, 
                            num_edge, 
                            short_edge_fraction, 
                            num_sink,
                            num_target,
                            max_node_to_target_association,
                            ):
    #self.T =  np.zeros(shape = (num_target, num_node), dtype = int)
    #-----first make a dense connected graph-----------
    self.num_node = num_node
    self.num_target = num_target
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
    '''
    for edge in self.adj.edges(data = True):
      print edge
    '''
    #------now randomly choose sinks-----------------------
    self.sinks = random.sample(self.adj.nodes(), num_sink)
        #-------- randomly hooks targets to source nodes-------
    #T is the matrix of size num_tarets x num_nodes
    #self.T =  np.zeros(shape = (num_target, num_node), dtype = bool)
    self.T_N=defaultdict(list)
    for t in range(num_target):
      num_of_assoc_sources = random.randint(1,max_node_to_target_association)
      assoc_sources = random.sample(self.adj.nodes(),num_of_assoc_sources)
      for n in assoc_sources:
        self.T_N[t].append(n)
    #print self.T

  
  def solve(self):
    #first make the problem instance
    N = 1.0 * self.adj.number_of_nodes()
    fso_placement_prob = pulp.LpProblem(name = "FSO_PLACEMENT_PROBLEM", sense = pulp.LpMaximize)
    #obj func. g_s must be defined first as required by the PuLP....
    g_s = pulp.LpVariable.dicts(name='g_s', 
                              indexs = self.adj.nodes(), 
                              lowBound=0,  
                              cat = pulp.LpContinuous)
    '''
    print "DEBUG: objective func. variable g_si created"
    for i,v in g_s.iteritems(): 
      print "DEBUG: ",i,":",v
    '''  
    fso_placement_prob += pulp.lpSum(g_s[i] for i in self.adj.nodes()), "objective_func_eqn_29" #<---objective func...
    
    #----task: make x_i variables and set equation (1) [default in var def] ---------
    x = pulp.LpVariable.dicts(name='x', 
                              indexs = self.adj.nodes(), 
                              lowBound=0, 
                              upBound =1, 
                              cat = pulp.LpContinuous)
    '''
    print "DEBUG: x_i variables created"
    for i,v in x.iteritems(): 
      print "DEBUG: ",i,":",v
    '''
    #***********end_of_task**************************************
    
    #----task: make e_ij variables set equation (2) [default in var def] ---------
    e = pulp.LpVariable.dicts(name='e', 
                              indexs = (self.adj.nodes(),self.adj.nodes()), 
                              lowBound=0, 
                              upBound =1, 
                              cat = pulp.LpContinuous)
    '''
    print "DEBUG: e_ij variables created"
    for i,e_item in e.iteritems():
      print "DEBUG: ",i,":",e_item
    ''' 
    #***********end_of_task**************************************
    
    #----task: for all non-LOS edges, set equation (3)  ---------
    for i in self.adj.nodes():
      for j in self.adj.nodes():
        if not self.adj.has_edge(i, j):
          fso_placement_prob += e[i][j] == 0, "eqn_3_("+str(i)+","+str(j)+")"
    #***********end_of_task**************************************
    
    #----task: make y_i variables and set equation (4)  ---------
    y = pulp.LpVariable.dicts(name='y', 
                              indexs = self.adj.nodes(), 
                              lowBound=0, 
                              #upBound =1, 
                              cat = pulp.LpContinuous)
    '''
    print "DEBUG: y_i variables created"
    for i,v in y.iteritems():  
      print "DEBUG: ",i,":",v
    '''  
    for i,v in y.iteritems():
      fso_placement_prob += y[i] <= x[i], "eqn_4_("+str(i)+")"
    #***********end_of_task**************************************
    
    #----task: make b_ij variables set equation (5)---------
    b = pulp.LpVariable.dicts(name='b', 
                              indexs = (self.adj.nodes(),self.adj.nodes()), 
                              lowBound=0, 
                              upBound =1, 
                              cat = pulp.LpContinuous)
    '''
    print "DEBUG: b_ij variables created"
    for i,v in b.iteritems():
      print "DEBUG: ",i,":",v
    '''  
    for i in self.adj.nodes():
      for j in self.adj.nodes():
        fso_placement_prob += b[i][j] <= e[i][j], "eqn_5_("+str(i)+","+str(j)+")"
    #***********end_of_task**************************************
    
    #----task: set equation (6) and (7) involving e_ij and b_ij ---------
    for i in self.adj.nodes():
      for j in self.adj.nodes():
        fso_placement_prob += e[i][j] == e[j][i], "eqn_6_("+str(i)+","+str(j)+")"
        fso_placement_prob += b[i][j] == b[j][i], "eqn_7_("+str(i)+","+str(j)+")"
    #***********end_of_task**************************************
    
    #----task: set equation (8) and (9)  involving x_i, x_j, e_ij, y_i, y_j, b_ij ---------
    for i in self.adj.nodes():
      for j in self.adj.nodes():
        if self.adj.has_edge(i, j):
          fso_placement_prob += e[i][j] <= 0.5 * (x[i] + x[j] ), "eqn_8_("+str(i)+","+str(j)+")"
        else:
          fso_placement_prob += e[i][j] == 0.0, "eqn_8_("+str(i)+","+str(j)+")"
        fso_placement_prob += b[i][j] <= 0.5 * (y[i] + y[j] ), "eqn_9_("+str(i)+","+str(j)+")"
    #***********end_of_task**************************************
    
    #----task: set equation (10) for each sink w, involving x_i, y_i ---------
    for i in self.sinks:
        fso_placement_prob += x[i] == 1, "eqn_10_a_("+str(i)+")"
        fso_placement_prob += y[i] == 1, "eqn_10_b_("+str(i)+")"
    #***********end_of_task**************************************
    
    #----task: set equation (11) aka (12) involving T_ij, y_i  ---------
    #num_targets = len(self.T_N)
    #num_nodes = self.num_node
    #for t in range(num_targets):
    for t in self.T_N.keys():
      #print "DEBUG @relaxed_ilp:T_N[t]",t,":",self.T_N[t]
      #print "DEBUG @relaxed_ilp:total-ys:",y
      fso_placement_prob += pulp.lpSum( y[i] for i in self.T_N[t] if i in y.keys()) >= 1,\
                           "eqn_11_("+str(t)+")"
    #***********end_of_task**************************************
    
    #----task: make variables f_ij, set equation (13) involving f_ij, b_ij ---------
    f = pulp.LpVariable.dicts(name='f', 
                              indexs = (self.adj.nodes(),self.adj.nodes()), 
                              lowBound=0, 
                              #upBound =1, 
                              cat = pulp.LpContinuous)
    '''
    print "DEBUG: f_ij variables created"
    for i,v in f.iteritems(): 
      print "DEBUG: ",i,":",v
    ''' 
    for i in self.adj.nodes():
      for j in self.adj.nodes():
        fso_placement_prob += f[i][j] <= b[i][j],"eqn_13_("+str(i)+","+str(j)+")"
    #***********end_of_task**************************************
        
    #----task: make f_si variables, set equation (14) involving f_si, y_i, N for non-sink node i ---------
    f_s = pulp.LpVariable.dicts(name='f_s', 
                              indexs = self.adj.nodes(), 
                              lowBound=0,  #should it be so ??!!
                              cat = pulp.LpContinuous)
    '''
    print "DEBUG: f_si variables created"
    
    for i,v in f_s.iteritems(): 
      print "DEBUG: ",i,":",v
    ''' 
    for i in self.adj.nodes():
      if i not in self.sinks:
        fso_placement_prob += f_s[i] == y[i] / N , "eqn_14_("+str(i)+")"
        #fso_placement_prob += f_s[i] <= y[i] * N , "eqn_14_U_("+str(i)+")"
    #***********end_of_task**************************************
    
        
    #----task: set equation (15) involving f_si for each sink i  ---------
    for i in self.sinks:
      fso_placement_prob += f_s[i] == 0,"eqn_15_("+str(i)+")"
    #***********end_of_task**************************************
    
        
    #----task: make variables f_ti, set equation (16) for each sink i ---------
    f_t = pulp.LpVariable.dicts(name='f_t', 
                              indexs = self.adj.nodes(), 
                              lowBound=0, 
                              #upBound = 1, #!!! is it really needed ?? 
                              cat = pulp.LpContinuous)
    '''
    print "DEBUG: f_jt variables created"
    for i,v in f_t.iteritems(): 
      print "DEBUG: ",i,":",v
    '''
    for i in self.sinks: 
      fso_placement_prob += f_t[i] <= N, "eqn_16_("+str(i)+")" 
    #***********end_of_task**************************************
    
        
    #----task: set equation (17) involving f_ti for non-sink node i ---------
    for i in self.adj.nodes():
      if i not in self.sinks:
        fso_placement_prob += f_t[i] == 0, "eqn_17_("+str(i)+")"
    #***********end_of_task**************************************
        
    #----task: make variables f_ij, set equation (18) involving f_ij, f_ti, f_si ---------
    for j in self.adj.nodes():
      fso_placement_prob += pulp.lpSum(f[i][j] for i in self.adj.nodes())+ f_s[j] == \
                            pulp.lpSum(f[j][i] for i in self.adj.nodes())+ f_t[j], \
                            "eqn_18_("+str(j)+")"
    #***********end_of_task**************************************
    
    #----task: set equation (19) involving f_ti, f_si ---------
    fso_placement_prob += pulp.lpSum(f_s[i] for i in self.adj.nodes()) == \
                          pulp.lpSum(f_t[j] for j in self.adj.nodes()), "eqn_19" 
    #***********end_of_task**************************************
    
    #----task: set equation (20) involving x_i, n_max for each non-sink node i ---------
    fso_placement_prob += pulp.lpSum(x[i] for i in self.adj.nodes() if i not in self.sinks)\
                                                                      <= self.nmax,"eqn_20"
    #***********end_of_task**************************************
    
    #----task: set equation (21) involving b_ij, d_max for each non-sink node i ---------
    for i in self.adj.nodes():
      if i not in self.sinks:
        fso_placement_prob += pulp.lpSum(b[i][j] for j in self.adj.nodes()) <= self.dmax,\
                                "eqn_21_("+str(i)+")"
    #***********end_of_task**************************************
    
    #----task: make variables g_ij, set equation (22) involving g_ij, e_ij ---------
    g = pulp.LpVariable.dicts(name='g', 
                              indexs = (self.adj.nodes(),self.adj.nodes()), 
                              lowBound=0, 
                              #upBound =1, 
                              cat = pulp.LpContinuous)
    '''
    print "DEBUG: g_ij variables created"
    for i,v in g.iteritems(): 
      print "DEBUG: ",i,":",v
    '''  
    for i in self.adj.nodes():
      for j in self.adj.nodes():
        fso_placement_prob += g[i][j] <= e[i][j],"eqn_22_("+str(i)+","+str(j)+")"
    #***********end_of_task**************************************
    
    #----task: make variables g_si(made at the top), set equation (23) involving g_si, N, x_i for non-sink node i ---------
    for i in self.adj.nodes():
      if i not in self.sinks:
        fso_placement_prob += g_s[i] <= N *x[i], "eqn_23_("+str(i)+")"
    #***********end_of_task**************************************
    #----task: set equation (24) involving g_si involving sink node i ---------
    for i in self.sinks:
      fso_placement_prob += g_s[i] == 0, "eqn_24_("+str(i)+")"
    #***********end_of_task**************************************
    
    #----task: make variables g_ti, set equation (25) involving g_ti, N for each sink-node i ---------
    g_t = pulp.LpVariable.dicts(name='g_t', 
                              indexs = self.adj.nodes(), 
                              lowBound=0,  
                              cat = pulp.LpContinuous)
    '''
    print "DEBUG: g_jt variables created"
    for i,v in g_t.iteritems(): 
      print "DEBUG: ",i,":",v 
    '''  
    for i in self.sinks:
      fso_placement_prob += g_t[i] <= N, "eqn_25_("+str(i)+")" 
    
    #***********end_of_task**************************************
    
    #----task: set equation (26) involving g_ti for each non-sink node i ---------
    for i in self.adj.nodes():
      if i not in self.sinks:
        fso_placement_prob += g_t[i] == 0, "eqn_26_("+str(i)+")"
    #***********end_of_task**************************************
    
    #----task: set equation (27) involving g_ij, g_ti, g_si ---------
    for j in self.adj.nodes(): 
      fso_placement_prob += pulp.lpSum(g[i][j] for i in self.adj.nodes())+ g_s[j] == \
                            pulp.lpSum(g[j][i] for i in self.adj.nodes())+ g_t[j], \
                            "eqn_27_("+str(j)+")"
    #***********end_of_task**************************************
    
    #----task: set equation (28) involving g_ti, g_si ---------
    fso_placement_prob += pulp.lpSum(g_s[i] for i in self.adj.nodes()) == \
                          pulp.lpSum(g_t[j] for j in self.adj.nodes()), "eqn_28"
    #***********end_of_task**************************************
    
    #----task: set equation (29/30) objective equation involving g_si ---------
      #done at the top
    #***********end_of_task**************************************
    
    #fso_placement_prob.writeLP("./lp_log/fso_placement_ILP_relaxed_"+str(self.adj.number_of_nodes())+"_.lp")
    fso_placement_prob.solve(pulp.CPLEX())  
    #fso_placement_prob.solve() 
    self.max_flow = pulp.value(fso_placement_prob.objective)
    #print "MAX FLOW:",  pulp.value(fso_placement_prob.objective)
    return

   