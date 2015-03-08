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
  def __init__(self, adj, sinks, T , T_N, static_d_max = 4): #, seed = 101239):
    #print "initializing Step_1_2_3_4...."
    #random.seed(seed)  
    self.capacity = 1.0
    self.adj = nx.Graph(adj)
    self.sinks = list(sinks)
    #self.node= deepcopy(node)
    self.T = deepcopy(T)
    self.T_N = deepcopy(T_N)
    self.G_p = None
    self.g1 = self.G_p # the backbone graph g=(y_i, b_ij)
    self.g =  self.adj # the full graph g1=(x_i, e_ij)
    self.n_max =  None
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
    #print "connected:",nx.is_connected(g)
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
      if s not in self.adj.nodes():
        for k in S:
          if k not in self.adj.nodes():  
            print "DEBUG:?? s not in adj:",k
        print "DEBUG: S:",S 
        print "DEBUG: T:", T 
        print "DEBUG: self.sinks:",self.sinks
        
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
    self.n_max = 2*self.G_p.number_of_nodes()
    return
    #-----------------related to step 4--------------------
  def add_super_source(self, g, src_list = None):
    if src_list == None: src_list= self.sources
    for n in src_list:
      g.add_edge('src',n,capacity = float('inf'))
    return g
  
  def add_super_sink(self, g, snk_list = None):
    if snk_list == None: snk_list= self.sinks
    for n in snk_list:
      g.add_edge(n,'snk',capacity = float('inf'))
    return g
  
  def add_capacity(self, g, capacity=1.0):
    edge_set = g.edges()
    for x,y in edge_set:
      g.edge[x][y]['capacity'] = capacity
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
  
  def generate_all_node_source_potential(self, r, nlist):
    source_potential ={}
    for n in nlist:
      source_potential[n] = 0
    for n in nlist:
      r1 = nx.DiGraph(r)
      
      if r1.has_node('src'):
        r1.remove_node('src')
        
      if not r1.has_node('snk'):
        r1 = self.add_super_sink(r1, nlist)
      if r1.has_edge(n,'snk'):
        r1.remove_edge(n, 'snk')
        
      r1 = flow.shortest_augmenting_path(G=r1, s=n ,t='snk', capacity = 'capacity')
      src_p = r1.graph['flow_value']
      '''
      if src_p == 0:
        print "src_p ==0!,n:",n
        self.print_graph(g = r)
        self.print_graph(g = r1)
        paths = nx.all_simple_paths(r1,n,'snk')
        for p in paths:
          print "\t\tp:",p
          nx.draw_networkx(r1, with_labels = True)
          #nx.draw_networkx_edge_labels(r1, nx.spring_layout(r1))
          plt.show()
        t_input = raw_input("press enter:")'''
      if src_p >0:
        source_potential[n] = src_p 
    return source_potential
  
  def generate_all_node_sink_potential(self, r, nlist):
    sink_potential ={}
    for n in nlist:
      sink_potential[n] = 0
    for n in nlist:
      r1 = nx.DiGraph(r)
      
      if r1.has_node('snk'):
        r1.remove_node('snk')
        
      if not r1.has_node('src'):
        r1 = self.add_super_source(r1, nlist)
      if r1.has_edge('src',n):
        r1.remove_edge('src',n)
        
      r1 = flow.shortest_augmenting_path(G=r1, s='src' ,t=n, capacity = 'capacity')
      snk_p = r1.graph['flow_value']
      '''
      if snk_p == 0:
        print "snk_p ==0!: n:",n
        self.print_graph(g = r)
        self.print_graph(g = r1)
        paths = nx.all_simple_paths(r1,'src',n)
        for p in paths:
          print "\t\tp:",p
          nx.draw_networkx(r1, with_labels = True)
          #nx.draw_networkx_edge_labels(r1, nx.spring_layout(r1))
          plt.show()
        t_input = raw_input("press enter:")'''
      if snk_p >0:
        sink_potential[n] = snk_p
      
    return sink_potential
  
  def get_time_diff(self, msg = None):
    time_t = self.last_time
    self.last_time = time.time()
    print"\t\tDEBUG: time elapsed: ", time.time() - time_t," ",msg
  
  def run_step_4(self):
    self.g = self.adj
    self.g1 = nx.Graph(self.G_p)
    #make it dynamic---##
    g1_node_list = self.g1.nodes()
    for u in g1_node_list:
      for v in g1_node_list:
        if u!=v and self.adj.has_edge(u, v):
          self.g1.add_edge(u,v)
    
    print"DEBUG: at the beginning of step_4_static:adj"
    #nx.draw_networkx(self.adj, with_labels = True)
    #plt.show()
    print"DEBUG: at the beginning of step_4"
    #nx.draw_networkx(self.g1, with_labels = True)
    #plt.show()
    
      
    backbone_nodes = list(self.g1.nodes())
    available_nodes = list(set(self.g.nodes()) - set(backbone_nodes))
    iter_counter = 0
    self.last_time = time.time()
    
    while(available_nodes and self.g1.number_of_nodes() <= self.n_max):
      iter_counter += 1
      self.get_time_diff()
      print "\t\tDEBUG: iteration no:",iter_counter,\
        " available nodes:",len(available_nodes),\
        " allowable nodes:",self.n_max - self.g1.number_of_nodes(),\
        " current nodes:",self.g1.number_of_nodes()
      #t_input = raw_input("press enter:")
      #self.get_time_diff(msg = "at the beginning of while loop") #----!!!time diff
      
      d = nx.Graph(self.g1)
      
      max_benefit = 0
      max_i = max_j = -1
      
      new_node_list = None
      
      backbone_nodes = list(self.g1.nodes())
      total_bnodes = len(backbone_nodes)
      
      src_list = list(set(self.g1.nodes()) - set(self.sinks))
      
      d = self.add_capacity(g = d, capacity = self.capacity)
      d= self.add_super_source(d, src_list = src_list)
      d = self.add_super_sink(d)
      
      r = flow.shortest_augmenting_path(G=d, s='src', t='snk', capacity = 'capacity')
      
      r = self.compute_residual_graph(g=r, capacity = 'capacity', flow='flow')
      
      source_potential = self.generate_all_node_source_potential(r = r, 
                                               nlist = backbone_nodes) 
      sink_potential = self.generate_all_node_sink_potential(r = r, 
                                             nlist = backbone_nodes) 
      for c1 in range(total_bnodes-1):
        i = backbone_nodes[c1]
        dg = nx.Graph(self.adj)
        i_path_length, i_path = nx.single_source_dijkstra(G = dg, source = i, target=None, cutoff=None)
        for c2 in range(c1+1, total_bnodes):
          j = backbone_nodes[c2]
          if j not in i_path_length.keys():
            print "\t\tDEBUG: i-j-disconnected in the adj graph: (i,j):",i,j
            #print "DEBUG:",i_path_length
            continue
          
          i_j_path = list(i_path[j])
          new_nodes_on_i_j_path = [u for u in i_j_path if u in available_nodes]
          
          #print "i,j, is_edge",i,j,self.g1.has_edge(i, j)
          #print "i-j-path:",i_j_path
          #print "i_src_p, i_snk_p:",source_potential[i], sink_potential[i]
          #print "j_src_p, j_snk_p:",source_potential[j], sink_potential[j]
          #print "DEBUG: new_nodes_on_i_j_path:",new_nodes_on_i_j_path
          
          #raw_t = raw_input("press enter (inside run-4 i,j loop):")
          
          new_node_count = len(new_nodes_on_i_j_path)
   
          if  new_node_count >= 1:
            i_j_path_benefit = min(source_potential[i], sink_potential[j])/(1.0 * new_node_count)
            j_i_path_benefit = min(source_potential[j], sink_potential[i])/(1.0 * new_node_count)
            #print "DEBUG:i_j_path_benefit, j_i_path_benefit ",i_j_path_benefit,j_i_path_benefit
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
      if max_i == -1:
        break 
      for n in new_node_list:
        self.g1.add_node(n)
        print "\t\tDEBUG@heurisitc_placement_algo..(): adding new node step iv: node",n
        available_nodes.remove(n)
      
      for n in new_node_list: 
        for nbr in self.g[n]:
          if nbr in self.g1.nodes():
            self.g1.add_edge(n,nbr)
      print "DEBUG: added node:",n
      #node_color = ['w' if u not in new_node_list else 'g' for u in self.g1.nodes()]
      #nx.draw_networkx(self.g1, with_labels = True, node_color = node_color)
      #plt.show()
    return
  def is_path_valid_for_static_graph(self, path):
    if len(path)<2:
      return False

    i = path[0]
    j = path[-1]
    
    i_deg = self.static.degree(i)
    j_deg = self.static.degree(j)
    if (self.static_d_max - i_deg) < 1 or (self.static_d_max - j_deg) < 1:
      return False
    for n in path[1:-1]:
      if self.static.has_node(n) and ( self.static_d_max - self.static.degree(n)) < 2:
        return False
    return True
  
  def run_static_modified_step_4(self):
    self.g = self.adj
    self.static = nx.Graph(self.G_p)
    
      
    backbone_nodes = list(self.static.nodes())
    available_nodes = list(set(self.g.nodes()) - set(backbone_nodes))
    iter_counter = 0
    self.last_time = time.time()
    
    while(available_nodes and self.static.number_of_nodes() <= self.n_max):
      iter_counter += 1
      self.get_time_diff()
      print "\t\tDEBUG: iteration no:",iter_counter,\
        " available nodes:",len(available_nodes),\
        " allowable nodes:",self.n_max - self.static.number_of_nodes(),\
        " current nodes:",self.static.number_of_nodes()
      d = nx.Graph(self.static)
      
      max_benefit = 0
      max_i = max_j = -1
      
      new_node_list = None
      
      backbone_nodes = list(self.static.nodes())
      total_bnodes = len(backbone_nodes)
      
      src_list = list(set(self.static.nodes()) - set(self.sinks))
      
      d = self.add_capacity(g = d, capacity = self.capacity)
      d= self.add_super_source(d, src_list = src_list)
      d = self.add_super_sink(d)
      
      r = flow.shortest_augmenting_path(G=d, s='src', t='snk', capacity = 'capacity')
      
      r = self.compute_residual_graph(g=r, capacity = 'capacity', flow='flow')
      
      source_potential = self.generate_all_node_source_potential(r = r, 
                                               nlist = backbone_nodes) 
      sink_potential = self.generate_all_node_sink_potential(r = r, 
                                             nlist = backbone_nodes) 
      for c1 in range(total_bnodes-1):
        i = backbone_nodes[c1]
        dg = nx.Graph(self.adj)
        i_path_length, i_path = nx.single_source_dijkstra(G = dg, source = i, target=None, cutoff=None)
        for c2 in range(c1+1, total_bnodes):
          j = backbone_nodes[c2]
          if j not in i_path_length.keys():
            print "\t\tDEBUG: i-j-disconnected in the adj graph: (i,j):",i,j
            #print "DEBUG:",i_path_length
            continue
          
          i_j_path = list(i_path[j])
          if not self.is_path_valid_for_static_graph(i_j_path):
            continue
          new_nodes_on_i_j_path = [u for u in i_j_path if u in available_nodes]
          
          new_node_count = len(new_nodes_on_i_j_path)
   
          if  new_node_count >= 1:
            i_j_path_benefit = min(source_potential[i], sink_potential[j])/(1.0 * new_node_count)
            j_i_path_benefit = min(source_potential[j], sink_potential[i])/(1.0 * new_node_count)
            #print "DEBUG:i_j_path_benefit, j_i_path_benefit ",i_j_path_benefit,j_i_path_benefit
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
      if max_i == -1:
        break 
      for n in new_node_list:
        available_nodes.remove(n)
      self.static.add_path(i_j_path)
      
      
      #print "DEBUG: added node:",n
      #node_color = ['w' if u not in new_node_list else 'g' for u in self.static.nodes()]
      #nx.draw_networkx(self.static, with_labels = True, node_color = node_color)
      #plt.show()
      
    static_nodes = self.static.nodes()
    len_static_nodes = len(static_nodes)
    for uindx in range(len_static_nodes - 1):
      u = static_nodes[uindx]
      if self.static.degree(u)+1 > self.static_d_max:
        continue
      for vindx in range(uindx+1, len_static_nodes):
        v= static_nodes[vindx]
        if self.static.degree(v)+1 > self.static_d_max:
          continue
        if self.adj.has_edge(u,v):
          self.static.add_edge(u,v)
        if self.static.degree(u)+1 > self.static_d_max:
          break 
    return

  
  def find_static_shortest_path(self, i, j, available_nodes):
            if (i,j) not in self.shortest_path_cache.keys():
              self.shortest_path_cache[(i,j)] = list(nx.all_shortest_paths(G=self.g,source=i,target=j))

            all_shortest_paths_i_j = list(self.shortest_path_cache[(i,j)])
            #print "DEBUG:@find_static_shortest_path:",all_shortest_paths_i_j
            minimum_node_from_availabe_set = 0
            min_path = []
            for p in all_shortest_paths_i_j:
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
            
            i_j_path = min_path
            if len(i_j_path)<=2: 
              return [], 0
            new_nodes_on_i_j_path = []
            for n in i_j_path:
              if n in available_nodes:
                new_nodes_on_i_j_path.append(n)
            return i_j_path, len(new_nodes_on_i_j_path)
  
  def run_static_step_4(self):
    
    self.g = self.adj
    self.static = nx.Graph(self.G_p)
    
    backbone_nodes = list(self.static.nodes())

    available_nodes = list(set(self.g.nodes()) - set(backbone_nodes))

    iter_counter = 0
    while(available_nodes and self.static.number_of_nodes() <= self.n_max):
      iter_counter += 1
      self.get_time_diff()
      print "\t\tDEBUG: iteration no:",iter_counter,\
        " available nodes:",len(available_nodes),\
        " allowable nodes:",self.n_max - self.static.number_of_nodes(),\
        " current nodes:",self.static.number_of_nodes() 
      d = nx.Graph(self.static)
      max_benefit = 0.0
      max_i = max_j = -1
 
      backbone_nodes = self.static.nodes()
      total_bnodes = len(backbone_nodes)
      
      src_list = list(set(self.static.nodes()) - set(self.sinks))
      d = self.add_capacity(g = d,capacity = self.capacity)
      d = self.add_super_source(d, src_list = src_list)
      d= self.add_super_sink(d)
      r = flow.shortest_augmenting_path(G=d, s='src', t='snk', capacity = 'capacity')
      r = self.compute_residual_graph(g=r, capacity = 'capacity', flow='flow')
      source_potential = self.generate_all_node_source_potential(r = r, 
                                                                nlist = backbone_nodes)
      sink_potential = self.generate_all_node_sink_potential(r = r, 
                                                          nlist = backbone_nodes)
      print "DEBUG:@step-4:src_pot:",source_potential
      print "DEBUG:@step-4:snk_pot:",sink_potential

      for c1 in range(total_bnodes-1):
        i = backbone_nodes[c1]
        for c2 in range(c1+1, total_bnodes):
          j = backbone_nodes[c2]
          i_j_path, new_node_count = self.find_static_shortest_path(i, j, available_nodes)
          if  new_node_count < 1:
            continue
          if i !=j:
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
              max_i_j_path  = list(reversed(i_j_path))
      if max_i == -1:
        break 
      for n in max_i_j_path:
        if n in available_nodes:
          available_nodes.remove(n)
      self.static.add_path(max_i_j_path)
 
    static_nodes = self.static.nodes()
    len_static_nodes = len(static_nodes)
    for uindx in range(len_static_nodes - 1):
      u = static_nodes[uindx]
      if self.static.degree(u)+1 > self.static_d_max:
        continue
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
                                   capacity = self.capacity)
      static_d = self.add_super_source(static_d, sources)
      static_d = self.add_super_sink(static_d, sink_list)
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
    d = nx.Graph(self.g1)
    d = self.add_capacity(g = d, capacity = self.capacity)
    d = self.add_super_source(d)
    d= self.add_super_sink(d)
    r = flow.shortest_augmenting_path(G=d, s='src', t='snk', capacity = 'capacity')
    self.max_flow = r.graph['flow_value']
    
    
    
    print "\t@heurisitc placement algo: step iv-b: builidng static graph...."
    self.run_static_modified_step_4()
    static_d = nx.Graph(self.static)
    static_d = self.add_capacity(g = static_d, capacity = self.capacity)
    static_d = self.add_super_source(static_d)
    static_d= self.add_super_sink(static_d)
    static_r = flow.shortest_augmenting_path(G=static_d, s='src', t='snk', capacity = 'capacity')
    self.static_max_flow = static_r.graph['flow_value']
    
    
    #nx.draw_networkx(self.g1, with_labels = True, node_color = 'w', label ="Dyanmic Graph")
    #plt.show()
    
    #nx.draw_networkx(self.static, with_labels = True, node_color = 'w', label ="Static Graph")
    #plt.show()
    return

 
 
  
  
  
  
