import networkx as nx
import networkx.algorithms.flow as flow

import random 
from collections import defaultdict

from numpy import mean,ceil
import numpy as np
from copy import deepcopy
from operator import itemgetter

import operator as op
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


#import pylab
#import relaxed_ilp_solver_for_placement_algo as rilp
#from xlwt.Row import adj
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
  def __init__(self, capacity, adj, sinks, T , T_N, n_max,  static_d_max = 4): #, seed = 101239):
    #print "initializing Step_1_2_3_4...."
    #random.seed(seed)  
    self.capacity = capacity
    self.adj = deepcopy(adj)
    self.sinks = deepcopy(sinks)
    #self.node= deepcopy(node)
    self.T = deepcopy(T)
    self.T_N = deepcopy(T_N)
    self.G_p = None
    self.g1 = self.G_p # the backbone graph g=(y_i, b_ij)
    self.g =  self.adj # the full graph g1=(x_i, e_ij)
    self.n_max =  n_max
    self.static_d_max = static_d_max
    #self.full_adj_max_flow = None
    self.avg_max_flow_val =  None
    self.avg_upper_bound_flow_val = None
    
    self.shortest_path_cache={}
    self.last_time = time.time()
    
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
    for y,S in F.iteritems():
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
        if nbr not in S and str(self.adj[s][nbr]['con_type'])=='short':
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
        if self.adj.has_edge(c, nbr) and str(self.adj[c][nbr]['con_type']) =='short':
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
    #self.bg = deepcopy(self.G_p)
    return
    #-----------------related to step 4--------------------
  def add_capacity(self, g, capacity=1.0, src_list=None,  snk_list = None):
    if src_list == None: src_list= self.sources
    if snk_list == None: snk_list=self.sinks
    
    g.add_node('src')
    for i in src_list:
      g.add_edge('src',i)
    
    g.add_node('snk')
    for i in snk_list:
      g.add_edge(i,'snk')
      
    edge_set = g.edges()
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
  
  def find_shortest_path_new_nodes(self, i, j, available_nodes):

            if (i,j) not in self.shortest_path_cache.keys():
              self.shortest_path_cache[(i,j)] = list(nx.all_shortest_paths(G=self.g,source=i,target=j))

              
            all_shortest_paths_i_j = list(self.shortest_path_cache[(i,j)])
            #print "DEBUG:@find_shortest_path_new_nodes:",all_shortest_paths_i_j
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
              return []
            new_nodes_on_i_j_path = []
            for n in i_j_path:
              if n in available_nodes:
                new_nodes_on_i_j_path.append(n)
            return new_nodes_on_i_j_path
  
  def find_path_benefit(self, r, i, j, new_node_count):
    r1 = flow.shortest_augmenting_path(G=r, s='src',t=i, capacity = 'capacity')
    i_potential = r1.graph['flow_value']
    r1 = flow.shortest_augmenting_path(G=r, s=j,t='snk', capacity = 'capacity')
    j_potential = r1.graph['flow_value']
            
    i_j_path_benefit = min(i_potential, j_potential)/(1.0 * new_node_count)
    return i_j_path_benefit
  
  def generate_all_node_potential(self, r, nlist):
    source_potential = {}
    sink_potential = {}
    for n in nlist:
      r1 = flow.shortest_augmenting_path(G=r, s='src',t=n, capacity = 'capacity')
      snk_p = r1.graph['flow_value']
      r1 = flow.shortest_augmenting_path(G=r, s=n ,t='snk', capacity = 'capacity')
      src_p = r1.graph['flow_value']
      source_potential[n] = src_p
      sink_potential[n] = snk_p
    return source_potential, sink_potential
  
  def get_time_diff(self, msg):
    time_t = self.last_time
    self.last_time = time.time()
    print"\t\tDEBUG: time elapsed: ", time.time() - time_t," sec: msg:",msg
  
  def run_step_4(self):
    self.g = self.adj
    self.g1 = deepcopy(self.G_p)
    
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
    
    iter_counter = 0
    self.last_time = time.time()
    while(available_nodes and self.g1.number_of_nodes() <= self.n_max):
      iter_counter += 1
      print "\t\tDEBUG: iteration no:",iter_counter,\
        " available nodes:",len(available_nodes),\
        " allowable nodes:",self.n_max - self.g1.number_of_nodes(),\
        " current nodes:",self.g1.number_of_nodes()
      self.get_time_diff(msg = "at the beginning of while loop") #----!!!time diff
      
      d = nx.Graph(self.g1)
      
      max_benefit = - float('inf')
      max_i = max_j = -1
      
      new_node_list = None
      
      backbone_nodes = self.g1.nodes()
      total_bnodes = len(backbone_nodes)
      
      src_list = list(set(self.g1.nodes()) - set(self.sinks))
      
      d = self.add_capacity(g = d, src_list = src_list, capacity = self.capacity)
      r = flow.shortest_augmenting_path(G=d, s='src', t='snk', capacity = 'capacity')
      r = self.compute_residual_graph(g=r, capacity = 'capacity', flow='flow')
      source_potential, sink_potential = self.generate_all_node_potential(r = r, 
                                                                         nlist = backbone_nodes) 
      self.get_time_diff(msg = "computed all potential, trapping in i,j loop") #----!!!time diff
      for c1 in range(total_bnodes-1):
        self.get_time_diff(msg = "current c1:"+str(c1)) #----!!!time diff
        i = backbone_nodes[c1]
        i_path_length, i_path = nx.single_source_dijkstra(G = self.adj, source = i, target=None, cutoff=None)
        for c2 in range(c1+1, total_bnodes):
          self.get_time_diff(msg = "current c2:"+str(c2)) #----!!!time diff
          
          j = backbone_nodes[c2]
          #new_nodes_on_i_j_path = self.find_shortest_path_new_nodes(i, j, available_nodes)
          if j not in i_path_length.keys():
            continue
          #i_j_path = nx.dijkstra_path(G = self.adj, source = i, target = j)
          i_j_path = i_path[j]
          #j_i_path = list(reversed(i_path[j]))
          new_nodes_on_i_j_path = [i for i in i_j_path if i in available_nodes]
          new_node_count = len(new_nodes_on_i_j_path)
   
          if  new_node_count < 1:
            continue
          #print "DEBUG:(i,j):",i,j
          if i !=j and not self.g1.has_edge(i, j):
            #step i) run max-flow and get the dynamic graph g1 and compute the residual graph
            #r = flow.shortest_augmenting_path(G=d, s='src', t='snk', src_list = src_list, capacity = 'capacity')
            #r = self.compute_residual_graph(g=r, capacity = 'capacity', flow='flow')
            
            #i_j_path_benefit = self.find_path_benefit(r, i, j, new_node_count)
            #j_i_path_benefit = self.find_path_benefit(r, j, i, new_node_count)
            i_j_path_benefit = min(source_potential[i], sink_potential[j])/(1.0 * new_node_count)
            j_i_path_benefit = min(source_potential[j], sink_potential[i])/(1.0 * new_node_count)
            if i_j_path_benefit > max_benefit:
              max_benefit = i_j_path_benefit
              max_i = i
              max_j = j
              new_node_list = list(new_nodes_on_i_j_path)
            elif j_i_path_benefit > max_benefit:
              max_benefit =j_i_path_benefit
              max_i = j
              max_j = i
              new_node_list = list(new_nodes_on_i_j_path)
            #t= raw_input("press enter:")
      #step vi) update graph g1 with new nodes,edges from i,j 
      #and remove the nodes from list available_nodes
      if max_i == -1:
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
  
  def find_static_shortest_path(self, i, j, available_nodes):
            if (i,j) not in self.shortest_path_cache.keys():
              self.shortest_path_cache[(i,j)] = list(nx.all_shortest_paths(G=self.g,source=i,target=j))

            all_shortest_paths_i_j = list(self.shortest_path_cache[(i,j)])
            #print "DEBUG:@find_static_shortest_path:",all_shortest_paths_i_j
            minimum_node_from_availabe_set = 0
            min_path = []
            for p in all_shortest_paths_i_j:
              #print "DEBUG:p:",p
              #print "DEBUG:available_set:",available_nodes
              #first check degree constraint
              invalid_path = False
              for nindx, n in enumerate(p):
                if n in self.static.nodes():
                  req_degree = 2
                  if nindx > 0 and self.static.has_edge(p[nindx - 1], p[nindx]):
                    req_degree -= 1
                  if nindx < len(p)-1 and self.static.has_edge(p[nindx], p[nindx+1]):
                    req_degree -= 1
                  if self.static.degree(n)+req_degree > self.static_d_max:
                    invalid_path = True
                    break
              if invalid_path:
                continue
              available_node_count = len(p) - len(set(p) - set(available_nodes))
              if available_node_count > minimum_node_from_availabe_set:
                minimum_node_from_availabe_set = available_node_count
                min_path = p
              #find the minimum number of external-node shortest paths
              
            #i_j_path = nx.shortest_path(self.g, i, j)
            
            i_j_path = min_path
            #print "DEBUG: i_j_path: ",i_j_path
            if len(i_j_path)<=2: 
              return [], 0
            new_nodes_on_i_j_path = []
            for n in i_j_path:
              if n in available_nodes:
                new_nodes_on_i_j_path.append(n)
            return i_j_path, len(new_nodes_on_i_j_path)
  
  def run_static_step_4(self):
    #print "DEBUG: running run_static_step_4()....."
    self.g = self.adj
    self.static = deepcopy(self.G_p)
    #find set of nodes still available to add to dynamic graph static to from input graph g
    backbone_nodes = self.static.nodes()
    #print "DEBUG:backbone_nodes:",backbone_nodes
    available_nodes = list(set(self.g.nodes()) - set(backbone_nodes))
    #print available_nodes
    
    iter_counter = 0
    while(available_nodes and self.static.number_of_nodes() <= self.n_max):
      iter_counter += 1
      print "\t\tDEBUG: iteration no:",iter_counter,\
        " available nodes:",len(available_nodes),\
        " allowable nodes:",self.n_max - self.static.number_of_nodes(),\
        " current nodes:",self.static.number_of_nodes() 
      #print "DEBUG: Available nodes:",len(available_nodes),\
      #      " current nodes:",self.static.number_of_nodes(),"------"
      d = nx.Graph(self.static)
      #print "DEBUG:type(d):",type(d)
      
      #print "DEBUG:type(d):",type(d)
      #print d.has_node('src')
      max_benefit = - float('inf')
      max_i = max_j = -1
      #new_node_list = None
      backbone_nodes = self.static.nodes()
      total_bnodes = len(backbone_nodes)
      
      src_list = list(set(self.static.nodes()) - set(self.sinks))
      
      #d = self.add_capacity(g = d,  src_list= src_list, capacity = self.capacity)
      #r = flow.shortest_augmenting_path(G=d, s='src', t='snk', capacity = 'capacity')
      #r = self.compute_residual_graph(g=r, capacity = 'capacity', flow='flow')
      
      d = self.add_capacity(g = d, src_list = src_list, capacity = self.capacity)
      r = flow.shortest_augmenting_path(G=d, s='src', t='snk', capacity = 'capacity')
      r = self.compute_residual_graph(g=r, capacity = 'capacity', flow='flow')
      source_potential, sink_potential = self.generate_all_node_potential(r = r, 
                                                                         nlist = backbone_nodes)
      
      for c1 in range(total_bnodes-1):
        for c2 in range(c1+1, total_bnodes):
          #print "DEBUG:", c1,c2
          i = backbone_nodes[c1]
          j = backbone_nodes[c2]
          i_j_path, new_node_count = self.find_static_shortest_path(i, j, available_nodes)
          #print "DEBUG:i_j_path:",i_j_path," new_node_count:",new_node_count
          #print "DEBUG:", c1,c2,i_j_path,new_node_count
          if  new_node_count < 1:
            continue
          #print "DEBUG:(i,j):",i,j
          if i !=j:
            #step i) run max-flow and get the dynamic graph static and compute the residual graph
            #r = flow.shortest_augmenting_path(G=d, s='src', t='snk', src_list= src_list, capacity = 'capacity')
            #r = self.compute_residual_graph(g=r, capacity = 'capacity', flow='flow')
            
            #i_j_path_benefit = self.find_path_benefit(r, i, j, new_node_count)
            #j_i_path_benefit = self.find_path_benefit(r, j, i, new_node_count)
            i_j_path_benefit = min(source_potential[i], sink_potential[j])/(1.0 * new_node_count)
            j_i_path_benefit = min(source_potential[j], sink_potential[i])/(1.0 * new_node_count)
            
            if i_j_path_benefit > max_benefit:
              max_benefit = i_j_path_benefit
              max_i = i
              max_j = j
              max_i_j_path = list(i_j_path)
            elif j_i_path_benefit > max_benefit:
              max_benefit =j_i_path_benefit
              max_i = j
              max_j = i
              #print  "\t\tDEBUG: i_j_path", list(reversed(i_j_path))
              max_i_j_path  = list(reversed(i_j_path))
            #t= raw_input("press enter:")
      #step vi) update graph static with new nodes,edges from i,j 
      #and remove the nodes from list available_nodes
      if max_i == -1:
        break #no i,j found
      #otherwise add the paths:
      #print "DEBUG: new_nodes_added:",new_node_list
      #print "DEBUG: available_nodes:",available_nodes
      #else add the path to the static graph
      for n in max_i_j_path:
        if n in available_nodes:
          available_nodes.remove(n)
      self.static.add_path(max_i_j_path)

    #now check for the less-degree nodes 
    static_nodes = self.static.nodes()
    len_static_nodes = len(static_nodes)
    #print "DEBUG: processing single edge addition..."
    for uindx in range(len_static_nodes - 1):
      #print "DEBUG: uindx:",uindx
      u = static_nodes[uindx]
      if self.static.degree(u)+1 > self.static_d_max:
        continue
      #edge_created = True 
      for vindx in range(uindx+1, len_static_nodes):
        v= static_nodes[vindx]
        if self.static.degree(v)+1 > self.static_d_max:
          continue
        if self.adj.has_edge(u,v):
          self.static.add_edge(u,v)
        if self.static.degree(u)+1 > self.static_d_max:
          break 
    return
  
  def ncr(self,n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom

  def find_static_average_flow(self, num_iterations = 100, source_ratio = 0.05):
    
    if   self.avg_max_flow_val and  self.avg_upper_bound_flow_val:
      return self.avg_max_flow_val, self.avg_upper_bound_flow_val
    print "\t\t@heurisitc placement algo: step: finding static graph avg. flows"
    self.static_avg_flow = 0.0
    self.static_upper_bound_flow = 0.0
    source_sample_size = int(ceil(source_ratio * self.static.number_of_nodes()))
    
    sample_flow_vals = []
    sample_upper_bound_flow_vals = []
    
    allowed_sources = []
    for n in self.static.nodes():
      if n not in self.sinks:
        allowed_sources.append(n)
        
    sink_list = []
    for n in self.sinks:
      if self.static.has_node(n):
        sink_list.append(n)
        
    pattern_count = self.ncr(self.static.number_of_nodes(), source_sample_size)
    real_iteration_count = min(pattern_count, num_iterations)
    
    for i in range(real_iteration_count):
      #choose source_ratio nodes randomly
      sources = random.sample(allowed_sources,source_sample_size)
      

      #run flow algo and find max flow
      static_d = nx.Graph(self.static)
      #print "DEBUG:@find_static_average_flow(): static_d nodes:",static_d.nodes()
      #print "DEBUG:@find_static_average_flow(): static_d src_list:",sources

      static_d = self.add_capacity(g = static_d, 
                                   capacity = self.capacity, 
                                   src_list = sources,
                                   snk_list= sink_list)
      #print "DEBUG:s-t path:", [p for p in nx.all_shortest_paths(G=static_d,source='src',target='snk')]
      static_r = flow.shortest_augmenting_path(G=static_d, s='src', t='snk', capacity = 'capacity')
      #set avg_flow and upper_bound_flows
      sample_flow_vals.append(static_r.graph['flow_value'])
      sample_upper_bound_flow_vals.append( sum(self.static.degree(sources).values()) )
    avg_max_flow_val = mean(sample_flow_vals)
    avg_upper_bound_flow_val = mean(sample_upper_bound_flow_vals)
    self.avg_max_flow_val =  avg_max_flow_val
    self.avg_upper_bound_flow_val = avg_upper_bound_flow_val
    return avg_max_flow_val, avg_upper_bound_flow_val

  def solve(self):
    print "\t@heurisitc placement algo: step i: running greedy target cover...."
    self.greedy_set_cover_targets()
    self.get_time_diff(msg = "greedy_set_cover_complete") #----!!!time diff
    print "\t@heurisitc placement algo: step ii: building backbone network...."
    self.build_backbone_network()
    self.get_time_diff(msg = "backbone_network_complete") #----!!!time diff
    print "\t@heurisitc placement algo: step iii: reducing node degree...."
    self.reduce_node_degree()
    self.get_time_diff(msg = "node_reduction_complete") #----!!!time diff
    print "\t@heurisitc placement algo: step iv-a: builidng dynamic graph...."
    self.run_step_4()
    #self.print_graph(self.g1)
    d = nx.Graph(self.g1)
      #print "DEBUG:type(d):",type(d)
    d = self.add_capacity(g = d, capacity = self.capacity)
    r = flow.shortest_augmenting_path(G=d, s='src', t='snk', capacity = 'capacity')
    self.max_flow = r.graph['flow_value']
    
    print "\t@heurisitc placement algo: step iv-b: builidng static graph...."
    #print "DEBUG: path cache:",self.shortest_path_cache
    self.run_static_step_4()
    #self.print_graph(self.g1)
    static_d = nx.Graph(self.static)
      #print "DEBUG:type(d):",type(d)
    static_d = self.add_capacity(g = static_d, capacity = self.capacity)
    static_r = flow.shortest_augmenting_path(G=static_d, s='src', t='snk', capacity = 'capacity')
    self.static_max_flow = static_r.graph['flow_value']
    '''
    print "\t@heurisitc placement algo: step: running max_flow on full adj graph...."
    full_d = nx.Graph(self.adj)
    full_d  = self.add_capacity(g = full_d , capacity = self.capacity)
    full_r = flow.shortest_augmenting_path(G=full_d , s='src', t='snk', capacity = 'capacity')
    self.full_adj_max_flow = full_r.graph['flow_value']'''
    return

 
 
  
  
  
  
