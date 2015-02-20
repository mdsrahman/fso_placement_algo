import networkx as nx
#import numpy as np
import random 
from collections import defaultdict
from babel._compat import iteritems
from docutils.parsers.rst.directives import path
from copy import deepcopy
from operator import itemgetter
import networkx.algorithms.flow as flow

import relaxed_ilp_solver_for_placement_algo as rilp
'''
S = set of nodes added.
Initially, S = set of gateways.
T = set of targets.
G = graph (all possible links)
G' = current graph.

while T is non-null {
    (t, P) = FindClosest(); /**Find a target t in T which is closest to some node s in S. **/
    Add the path P to G'.
    S = S union {t}
    T = T - t
    }


FindClosest() {
  S' = S
  for each s in S' {
    S'' = new nodes (i.e., not in S') reachable from s.
    Does S'' contain a target t? If yes  {return t and P from T to a node in S}
    S' = S' union S''.
    Stop with error if no more new nodes. /* Means there is an unreachable target */
    }
}

task 1: make a random dense graph with random target nodes, short and long links
task 2: run round-robin BFS, till you hit a new 

'''
class Heuristic_Placement_Algo:
  def __init__(self, capacity =1.0, seed = 1012915):
    print "initializing Step_1_2_3_4...."
    random.seed(seed)  
    self.capacity = capacity
    self.adj = None
    self.G_p = None
    self.g1 = self.G_p # the backbone graph g=(y_i, b_ij)
    self.g =  self.adj # the full graph g1=(x_i, e_ij)
    
  def print_graph(self, g):
    if 'name' in g.graph.keys():
      print "graph_name:",g.graph['name']
    print "---------------------------"
    print "connected:",nx.is_connected(g)
    print "num_edges:",g.number_of_edges()
    
    for n in g.nodes(data=True):
      print n 
    for e in g.edges(data=True):
      print e
      
  def greedy_set_cover_targets(self):
    F = deepcopy(self.T)
    #print "DEBUG:", F
    D = defaultdict(list)
    L=defaultdict(set)
    for y,S in iteritems(F):
        for a in S:
            D[a].append(y)
        # Now place sets into an array that tells us which sets have each size
        L[len(S)].add(y)
        
    
    #self.E=[] # Keep track of selected sets
    # Now loop over each set size
    E=[]
    for sz in range(max(len(S) for S in F.values()),0,-1):
        if sz in L:
            P = L[sz] # set of all sets with size = sz
            while len(P):
                x = P.pop()
                E.append(x)
                for a in F[x]:
                    for y in D[a]:
                        if y!=x:
                            S2 = F[y]
                            L[len(S2)].remove(y)
                            S2.remove(a)
                            L[len(S2)].add(y)
    self.N = deepcopy(E)
    
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
    
  def find_path(self,p,t):
    n = t
    path=[n]
    while n in p.keys():
      path = [p[n]]+path
      n = p[n]
    return path
  
  def find_closest(self, S_in, T):
    S =list(S_in)
    parent_node ={}
    t=None
    P=[]
    for s in S:
      #print "DEBUG:S",S
      s_nbr = self.adj.neighbors(s)
      for nbr in s_nbr:
        if nbr not in S and self.adj[s][nbr]['con_type']=='short':
          parent_node[nbr] = s
          S.append(nbr)
          if nbr in T:
            t=nbr
            P=self.find_path(parent_node, t)
            #print  'DEBUG:P',P
            return t,P            
    return t,P

  def build_backbone_network(self):
    self.uT=[]
    S=set(self.sinks)
    #T=self.T.keys()
    T= set(self.N)
    T =  T - S.intersection(T)
    self.G_p = nx.Graph()
    self.G_p.graph['name'] = 'Backbone Graph'
    #G= self.adj # do not modify G
    self.sources = []
    while T:
      t, P = self.find_closest(S, T)
      if t==None:
        if len(T)>0:
          self.uT=deepcopy(T) 
        return
      else:
        self.G_p.add_path(P)
        #print "DEBUG:S:",S
        T.remove(t)
        self.sources.append(t)
        S.add(t)
        #print "DEBUG:S:",S
    return
  def get_max_degree_nodes(self,g):
    n_list=[]
    non_sink_nodes = set(g.nodes()) - set(self.sinks).intersection(g.nodes())
    node_degree = nx.degree(g, non_sink_nodes)
    sorted_node_degree = sorted(node_degree.items(), key=itemgetter(1),reverse=True)
    #x, max_degree =  sorted_node_degree[0]
    max_degree = -1
    for n,degree in sorted_node_degree:
      if degree >= max_degree:
        max_degree = degree
        n_list.append(n)
      else:
        break
    #also send back a candidate list of nodes in sorted order of suitiabiilty
    candidate_nodes = []
    candidate_nodes.extend(self.sinks)
    for n,degree in reversed(sorted_node_degree):
      if degree <= max_degree - 2:
        candidate_nodes.append(n)
      else:
        break
    return max_degree, n_list,candidate_nodes
  def check_node_for_degree_reduction(self, n, cnodes):
    #print "DEBUG@check_for_node_degree_reduction:",n,cnodes
    for s in self.sinks:
      bfs_successors = nx.bfs_successors(self.G_p, s)
      if n in bfs_successors.keys():
        break
    else:
      return False
    nbr_s = bfs_successors[n]
    for c in cnodes:
      for nbr in nbr_s:
        if self.adj.has_edge(c, nbr):
          self.G_p.remove_edge(n, nbr)
          self.G_p.add_edge(c, nbr)
          #print "prev_edge:",n,"-",nbr," new edge:",c,"-",nbr
          #self.print_graph(self.G_p)
          return True
    return False
  def reduce_node_degree(self):
    #add all sinks if not added:
    for s in self.sinks:
      self.G_p.add_node(s)
    
    #for u,v in iteritems(nx.degree(self.G_p)):
    #  print u,v
    
    
    while True:
      max_degree, max_deg_nodes, cnodes = self.get_max_degree_nodes(self.G_p)
      self.d_max = max_degree
      #print max_degree,max_deg_nodes,cnodes
      if max_degree <= 1 or len(max_deg_nodes)==0 or len(cnodes)==0:
        break
      degree_reduced = False
      for n in max_deg_nodes:
        degree_reduced = self.check_node_for_degree_reduction(n, cnodes)
        if degree_reduced:
          break
      if not degree_reduced:
        break
      
    
    
    #bfs_successors = nx.bfs_successors(self.G_p, s)

    return
    #-----------------related to step 4--------------------
  def add_capacity(self, g, capacity=1.0, src_list=None,  snk_list = None):
    if src_list == None: src_list= self.sources
    if snk_list == None: snk_list=self.sinks
    
    g.add_node('src')
    for i in self.sources:
      g.add_edge('src',i)
    
    g.add_node('snk')
    for i in self.sinks:
      g.add_edge(i,'snk')
      
    edge_set =  g.edges()
    for x,y in edge_set:
      g.edge[x][y]['capacity'] = capacity
      
    #now override the supersrc-->src capacities and sink-->supersink capacities
    #self.print_graph(g)
    for n in src_list:
      g.edge['src'][n]['capacity'] = float('inf')
    
    for n in snk_list:
      g.edge[n]['snk']['capacity'] = float('inf')
    return g
  
  def compute_residual_graph(self,g, capacity='capacity', flow='flow'):
    for i,j in g.edges():
      g[i][j]['capacity'] -= g[i][j]['flow']
      g[i][j]['flow'] = 0.0
    return g
    
  
  def run_step_4(self):
    self.g = self.adj
    self.g1 = self.G_p
    '''#<-- 
    Given a dynamic graph D, "residual graph" G and super-source S and super-sink T.
    while (nodes available to add) {
             For every ORDERED pair of nodes i,j in G
                Benefit(i,j) =   min(source-potential(i), sink-potential(j)) / newNodesPathLength(i,j)
             Pick the pair with highest benefit and add nodes on path(i,j) to D (dynamic graph).
             Also, add any links between nodes already added to D
             Find the residual graph G again. 
    
    newLinksPathLength(i,j)
    This is the shortest length of a path from i to j using only NEW intermediate nodes.
    
    Source-potential(i) -- This is the maximum flow from S to i in G.
    Sink-potential(j) -- This is maximum flow from i to T in G.
    '''#<--
    #find set of nodes still available to add to dynamic graph g1 to from input graph g
    backbone_nodes = self.g1.nodes()
    #print "DEBUG:backbone_nodes:",backbone_nodes
    available_nodes = list(set(self.g.nodes()) - set(backbone_nodes))
    #print available_nodes
    
    
    while(available_nodes):
      d = nx.Graph(self.g1)
      #print "DEBUG:type(d):",type(d)
      d = self.add_capacity(g = d, capacity = self.capacity)
      #print "DEBUG:type(d):",type(d)
      #print d.has_node('src')
      max_benefit = - float('inf')
      max_i = max_j = -1
      new_node_list = None
      
      for i in backbone_nodes:
        for j in backbone_nodes:
          #print "DEBUG:(i,j):",i,j
          if i !=j:
            #step i) run max-flow and get the dynamic graph g1 and compute the residual graph
            r = flow.shortest_augmenting_path(G=d, s='src', t='snk', capacity = 'capacity')
            r = self.compute_residual_graph(g=r, capacity = 'capacity', flow='flow')
            #step ii) find source-potential for i on r
            r1 = flow.shortest_augmenting_path(G=r, s='src',t=i, capacity = 'capacity')
            i_potential = r1.graph['flow_value']
            #step iii) find sink-potential for j on r
            r1 = flow.shortest_augmenting_path(G=r, s=j,t='snk', capacity = 'capacity')
            j_potential = r1.graph['flow_value']
            #step iv) find shortest path from i to j on g
            all_shortest_paths_i_j = nx.all_shortest_paths(G=self.g,source=i,target=j)
            
            minimum_node_from_availabe_set = 0
            min_path = []
            for p in all_shortest_paths_i_j:
              #print "DEBUG:p:",p
              #print "DEBUG:available_set:",available_nodes
              available_node_count = len(p) - len(set(p) - set(available_nodes))
              if available_node_count > minimum_node_from_availabe_set:
                minimum_node_from_availabe_set = available_node_count
                min_path = p
              #find the minimum number of external-node shortest paths
              
            #i_j_path = nx.shortest_path(self.g, i, j)
            
            i_j_path = min_path
            #print "DEBUG: i_j_path: ",i_j_path
            if len(i_j_path)<=2: 
              continue
            new_nodes_on_i_j_path = []
            for n in i_j_path:
              if n in available_nodes:
                new_nodes_on_i_j_path.append(n)
            
            new_node_count = len(new_nodes_on_i_j_path )
            if  new_node_count < 1:
              continue
            
            #step v) calculate benefit, if the benefit>max, save i,j,benefit,path(i->j)
            i_j_path_benefit = min(i_potential, j_potential)/(1.0 * new_node_count)
            #print "DEBUG:new_nodes_i_j_path:",new_nodes_on_i_j_path
            #print "DEBUG:i_potential:",i_potential
            #print "DEBUG:j_potential:",j_potential
            #print "DEBUG: (i,j,path_benefit):",i,",",j,",",i_j_path_benefit
            if i_j_path_benefit > max_benefit:
              max_benefit = i_j_path_benefit
              max_i = i
              max_j = j
              new_node_list = new_nodes_on_i_j_path
            #t= raw_input("press enter:")
      #step vi) update graph g1 with new nodes,edges from i,j 
      #and remove the nodes from list available_nodes
      if max_i ==-1:
        break #no i,j found
      #otherwise add the paths:
      #print "DEBUG: new_nodes_added:",new_node_list
      #print "DEBUG: available_nodes:",available_nodes
      for n in new_node_list:
        self.g1.add_node(n)
        available_nodes.remove(n)

      for n in new_node_list: 
        for nbr in self.g[n]:
          if nbr in self.g1.nodes():
            self.g1.add_edge(n,nbr)
    return
  def solve(self):
    self.greedy_set_cover_targets()
    self.build_backbone_network()
    self.reduce_node_degree()
    
    self.run_step_4()
    #self.print_graph(self.g1)
    d = nx.Graph(self.g1)
      #print "DEBUG:type(d):",type(d)
    d = self.add_capacity(g = d, capacity = self.capacity)
    r = flow.shortest_augmenting_path(G=d, s='src', t='snk', capacity = 'capacity')
    self.max_capacity = r.graph['flow_value']
    return
  

if __name__ == '__main__':
  print "starting run..."
  hp = Heuristic_Placement_Algo()
  hp.make_problem_instance(num_node = 50, 
                              num_edge = 200, 
                              short_edge_fraction = 0.5, 
                              num_sink = 10, 
                              num_target = 30, 
                              max_node_to_target_association = 4)
  #for i,v in iteritems(s_2_3.T):
  #  print i,v
  hp.solve()
  #hp.print_graph(hp.g1)
  print "Max_Capacity:",hp.max_capacity
  print "Backbone Network max degree (d_max):",hp.d_max
  n_max = hp.g1.number_of_nodes()
  print "Total Nodes in the network:", n_max 
  print 'building Relaxed ILP....'
  ilp = rilp.ILP_Relaxed(nmax = n_max,
                          dmax = hp.d_max,
                          adj = hp.adj,
                          T_N = hp.T_N,
                          sinks = hp.sinks)
  ilp.solve()
 
  
  
  
  
