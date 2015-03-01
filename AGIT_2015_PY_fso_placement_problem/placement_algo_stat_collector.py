
"""runs the algo in heurisitc_placement_algo and relaxed_ilp_sovler_for_placement_algo
 collects the following stat in the specified files:
 for input graph:
  a) for adj_graph
    1) number_of_nodes 
    2) number_of_edges 
    3) max_node_degree
    4) min_node_degree
    5) avg_node_degree
    6) number_of_sink
    7) number_of_targets
  b) for output:backbone_graph (G_p)
    1) number_of_nodes 
    2) number_of_edges 
    3) max_node_degree
    4) min_node_degree
    5) avg_node_degree
  c) for output:dynamic_graph (g_1)
    1) number_of_nodes 
    2) number_of_edges 
    3) max_node_degree
    4) min_node_degree
    5) avg_node_degree
    6) number of long_edges
    7) number of long_edges as % of short edges
  d) for flow values:heuristic and RILP
    1) flow_val heuristic
    2) flow_val RILP
    3) flow_val_heuristic/flow_val_RILP as %
  also plots the cdf for the full run
"""
import heuristic_placement_algo as hpa
import relaxed_ilp_solver_for_placement_algo as rilp
import networkx as nx
from collections import defaultdict
import random

class Placement_Algo_Stat_Collector():
  def __init__(self,seed):
    self.seed = seed
    self.sample_no = 0
    print "Initializing Placement_Algo_Stat_Collector"
    
  def make_problem_instance(self,
                            num_node,
                            n_max,
                            num_edge, 
                            short_edge_fraction, 
                            num_sink,
                            num_target,
                            max_node_to_target_association,
                            ):
    #self.T =  np.zeros(shape = (num_target, num_node), dtype = int)
    
    #-----first make a dense connected graph-----------
    self.n_max= n_max
    self.sample_no += 1
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
    self.T=defaultdict(set)
    self.T_N=defaultdict(list)
    for t in range(num_target):
      num_of_assoc_sources = random.randint(1,max_node_to_target_association)
      assoc_sources = random.sample(self.adj.nodes(),num_of_assoc_sources)
      for n in assoc_sources:
        self.T[n].add(t) #node n covers target t
        self.T_N[t].append(n)
  def solve(self):
    hp = hpa.Heuristic_Placement_Algo(    capacity =1.0, 
                                          adj = self.adj, 
                                          sinks = self.sinks, 
                                          T = self.T, 
                                          T_N = self.T_N,
                                          n_max = self.n_max)
                                          #seed = self.seed)
    hp.solve()
    print "Max_Capacity:",hp.max_capacity
    print "Backbone Network max degree (d_max):",hp.d_max
    print "Total Nodes in the network:", hp.g1.number_of_nodes()
    print 'building Relaxed ILP....'
    ilp = rilp.ILP_Relaxed(nmax = self.n_max,
                            dmax = hp.d_max,
                            adj = hp.adj,
                            T_N = hp.T_N,
                            sinks = hp.sinks)
    ilp.solve()
    print "ILP Max FLow:",ilp.max_flow
    percent_flow = 100.0 * hp.max_capacity / ilp.max_flow
    print "PERCENT OF RILP:",percent_flow,"%"
    with open('dyn_stat.txt', 'a') as f:
      fstr = str(self.sample_no)+','\
              +str(self.n_max)+','\
              +str(hp.d_max)+','\
              +str(hp.max_capacity)+','\
              +str(ilp.max_flow)+','\
              +str(percent_flow)+'\n'
      f.write(fstr)
      #self.sample_no += 1
    return

if __name__ == '__main__':
  seed = 101937
  s = Placement_Algo_Stat_Collector(seed = seed)
  num_samples = 20
  for i in range(num_samples):
    s.make_problem_instance(num_node = 100, 
                            n_max =  50,
                            num_edge = 200, 
                            short_edge_fraction = 0.5, 
                            num_sink = 4, 
                            num_target = 20, 
                            max_node_to_target_association = 3)
    s.solve()
    
  
  


