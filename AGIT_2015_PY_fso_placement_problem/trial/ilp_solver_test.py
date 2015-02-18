import pulp
import networkx as nx
import numpy as np
import random
import sys

'''steps--------
pre: build random nx grap instance- targets, adj graphs
i) build variable dicts
ii) build linear equations
iii) call the solver 

'''
class ILP_Relaxed():
  def __init__(self,nmax,dmax=3):
    print "ILP_Relaxed initialized...."
    self.adj = None
    self.T = None
    self.sinks =None
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
    #obj func. g_s must be defined first as required by the PuLP....
    g_s = pulp.LpVariable.dicts(name='g_s', 
                              indexs = self.adj.nodes(), 
                              lowBound=0,  
                              cat = pulp.LpContinuous)
    print "DEBUG: objective func. variable g_si created"
    for i,v in g_s.iteritems(): 
      print "DEBUG: ",i,":",v
    fso_placement_prob += pulp.lpSum(g_s[i] for i in self.adj.nodes()), "objective_func_eqn_24" #<---objective func...
    
    
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
    for i in self.adj.nodes():
      for j in self.adj.nodes():
        if not self.adj.has_edge(i, j):
          fso_placement_prob += e[i][j] == 0, "eqn_1_non_edge_("+str(i)+","+str(j)+")"
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
    e_s = pulp.LpVariable.dicts(name='e_s', 
                              indexs = self.adj.nodes(), 
                              lowBound=0,  
                              cat = pulp.LpContinuous)
    print "DEBUG: e_si variables created"
    for i,v in e_s.iteritems(): 
      print "DEBUG: ",i,":",v
    for i in self.adj.nodes():
      fso_placement_prob += e_s[i] <= x[i], "eqn_10_("+str(i)+")"
    #***********end_of_task**************************************
    
    #---task: make variables b_si and set equation (11) involving b_si and y_i ---
    b_s = pulp.LpVariable.dicts(name='b_s', 
                              indexs = self.adj.nodes(), 
                              lowBound=0,  
                              cat = pulp.LpContinuous)
    print "DEBUG: b_si variables created"
    for i,v in b_s.iteritems(): 
      print "DEBUG: ",i,":",v
    for i in self.adj.nodes():
      fso_placement_prob += b_s[i] <= y[i], "eqn_11_("+str(i)+")"
    #print sys.maxint
    #***********end_of_task**************************************
    #---task: make variables e_wt and set equation (12)----
    e_wt = pulp.LpVariable.dicts(name='e_wt', 
                              indexs = self.adj.nodes(), 
                              #lowBound=0,  
                              cat = pulp.LpContinuous)
    print "DEBUG: e_wt variables created"
    for i,v in e_wt.iteritems(): 
      print "DEBUG: ",i,":",v
    for i in self.adj.nodes():
      fso_placement_prob += e_wt[i] == 1, "eqn_12_("+str(i)+")"
    #***********end_of_task**************************************
    
    #---task: make variables f_si and set equation (13) involving f_si, y_i, N ----
    N = sys.maxint * 1.0
    f_s = pulp.LpVariable.dicts(name='f_s', 
                              indexs = self.adj.nodes(), 
                              #lowBound=0,  
                              cat = pulp.LpContinuous)
    print "DEBUG: f_si variables created"
    for i,v in f_s.iteritems(): 
      print "DEBUG: ",i,":",v
    for i in self.adj.nodes():
      fso_placement_prob += f_s[i] >= y[i] / N , "eqn_13_L_("+str(i)+")"
      fso_placement_prob += f_s[i] <= y[i] * N , "eqn_13_U_("+str(i)+")"
    #***********end_of_task**************************************
    
    #---task: make variables f_ij and set equation (14) involving f_ij and b_ij ----
    f = pulp.LpVariable.dicts(name='f', 
                              indexs = (self.adj.nodes(),self.adj.nodes()), 
                              lowBound=0, 
                              #upBound =1, 
                              cat = pulp.LpContinuous)
    print "DEBUG: f_ij variables created"
    for i,v in f.iteritems(): 
      print "DEBUG: ",i,":",v
    for i in self.adj.nodes():
      for j in self.adj.nodes():
        fso_placement_prob += f[i][j] <= b[i][j],"eqn_14_("+str(i)+","+str(j)+")"
        
    #***********end_of_task**************************************
    #---task: make variables f_jt and set equation (15) involving f_si and f_jt ----
    f_t = pulp.LpVariable.dicts(name='f_t', 
                              indexs = self.adj.nodes(), 
                              lowBound=0, 
                              upBound = 1, #!!! is it really needed ?? 
                              cat = pulp.LpContinuous)
    print "DEBUG: f_jt variables created"
    for i,v in f_t.iteritems(): 
      print "DEBUG: ",i,":",v
    fso_placement_prob += pulp.lpSum(f_s[i] for i in self.adj.nodes()) == \
                          pulp.lpSum(f_t[j] for j in self.adj.nodes()), "eqn_15" 
    #***********end_of_task**************************************
    #---task: set equation (16) involving f_ij, f_si, f_jt ------
    for j in self.adj.nodes():
      fso_placement_prob += pulp.lpSum(f[i][j] for i in self.adj.nodes())+ f_s[j] <= \
                            pulp.lpSum(f[j][i] for i in self.adj.nodes())+ f_t[j], \
                            "eqn_16_("+str(j)+")"
    #***********end_of_task**************************************
    #---task: set equation (17) involving x_i and n_max ----
    fso_placement_prob += pulp.lpSum(x[i] for i in self.adj.nodes()) <= self.nmax,\
                          "eqn_17"
    #***********end_of_task**************************************
    #---task: set equation (18) involving set of sinks, b_ij, d_max ----
    for i in self.adj.nodes():
      if i not in self.sinks:
        fso_placement_prob += pulp.lpSum(b[i][j] for j in self.adj.nodes()) <= self.dmax,\
        "eqn_18_("+str(i)+")"
    #***********end_of_task**************************************
    #---task: set variables g_si and g_jt and set equation (19) -----
    #g_s is declared at the top, due to the requirement that obj must appear first in PuLP
    '''
    g_s = pulp.LpVariable.dicts(name='g_s', 
                              indexs = self.adj.nodes(), 
                              lowBound=0,  
                              cat = pulp.LpContinuous)
    print "DEBUG: g_si variables created"
    for i,v in g_s.iteritems(): 
      print "DEBUG: ",i,":",v
    '''
    g_t = pulp.LpVariable.dicts(name='g_t', 
                              indexs = self.adj.nodes(), 
                              lowBound=0,  
                              cat = pulp.LpContinuous)
    print "DEBUG: g_jt variables created"
    for i,v in g_t.iteritems(): 
      print "DEBUG: ",i,":",v   
    fso_placement_prob += pulp.lpSum(g_s[i] for i in self.adj.nodes()) == \
                          pulp.lpSum(g_t[j] for j in self.adj.nodes()), "eqn_19"
      
    #***********end_of_task**************************************
    
    #---task: set variables g_ij and set equation (20) involving g_ij, g_si, g_jt ----
    g = pulp.LpVariable.dicts(name='g', 
                              indexs = (self.adj.nodes(),self.adj.nodes()), 
                              lowBound=0, 
                              #upBound =1, 
                              cat = pulp.LpContinuous)
    print "DEBUG: g_ij variables created"
    for i,v in g.iteritems(): 
      print "DEBUG: ",i,":",v
      
    for j in self.adj.nodes(): 
      fso_placement_prob += pulp.lpSum(g[i][j] for i in self.adj.nodes())+ g_s[j] <= \
                            pulp.lpSum(g[j][i] for i in self.adj.nodes())+ g_t[j], \
                            "eqn_20_("+str(j)+")"
    #***********end_of_task**************************************
    
    #---task: set equation (21) involving g_ij and e_ij ----
    for i in self.adj.nodes():
      for j in self.adj.nodes():
        fso_placement_prob += g[i][j] <= e[i][j],"eqn_21_("+str(i)+","+str(j)+")"
    #***********end_of_task**************************************
    #---task: set equation (22) involving N, f_si and x_i ----
    for i in self.adj.nodes():
      fso_placement_prob += f_s[i] <= N*x[i],"eqn_22_("+str(i)+")"
    #***********end_of_task**************************************
    #---task: set equation (23) involving N, f_it and x_i ----
    for i in self.adj.nodes():
      fso_placement_prob += f_t[i] <= N*x[i],"eqn_23_("+str(i)+")"
    #***********end_of_task**************************************
    #---task: set objective equation (24) involving g_si
    #done in the first step ----due to the structure of PuLP, where obj must appear first
    #***********end_of_task**************************************
    fso_placement_prob.writeLP("fso_placement_ILP_relaxed_"+str(self.adj.number_of_nodes())+"_.lp")
    fso_placement_prob.solve(pulp.CPLEX()) 
    #fso_placement_prob.solve() 
    return

   
   
if __name__ == '__main__':
  ilp = ILP_Relaxed(nmax = 200, dmax = 8)
  ilp.make_problem_instance(num_node = 10, 
                            num_edge = 30, 
                            short_edge_fraction = 1.0, 
                            num_sink = 6,
                            num_target = 4,
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