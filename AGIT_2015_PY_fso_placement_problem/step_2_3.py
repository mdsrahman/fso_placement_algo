import networkx as nx
#import numpy as np
import random 
from collections import defaultdict
from babel._compat import iteritems
from docutils.parsers.rst.directives import path
from copy import deepcopy
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
class Step_2_3:
  def __init__(self):
    print "initializing Step_2_3...."
    
  def print_graph(self, g):
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
    for t in range(num_target):
      num_of_assoc_sources = random.randint(1,max_node_to_target_association)
      assoc_sources = random.sample(self.adj.nodes(),num_of_assoc_sources)
      for n in assoc_sources:
        self.T[n].add(t) #node n covers target t
    
  def find_path(self,p,t):
    n = t
    path=[n]
    while n in p.keys():
      path = [p[n]]+path
      n = p[n]
    return path
  
  def find_closest(self, S, T):
    parent_node ={}
    t=None
    P=[]
    for s in S:
      s_nbr = self.adj.neighbors(s)
      for nbr in s_nbr:
        if nbr not in S and self.adj[s][nbr]['con_type']=='short':
          parent_node[nbr] = s
          S.append(nbr)
          if nbr in T:
            t=nbr
            P=self.find_path(parent_node, t)
            return t,P            
    return t,P
  def add_path_to_graph_G_p(self,P):
    if len(P) < 2:
      self.G_p.add_node(P.pop(0))
    else:
      u=P.pop(0)
      while P:
        v=P.pop(0)
        self.G_p.add_edge(u, v)
        u=v
    return
  def build_backbone_network(self):
    S=list(self.sinks)
    #T=self.T.keys()
    T= deepcopy(self.N)
    self.G_p = nx.Graph()
    self.G_p.graph['name'] = 'Backbone Graph'
    #G= self.adj # do not modify G
    while T:
      t, P = self.find_closest(S, T)
      if t==None:
        return
      else:
        self.add_path_to_graph_G_p(P)
        T.remove(t)
        S.append(t)
    return
if __name__ == '__main__':
  print "starting run..."
  s_2_3 = Step_2_3()
  s_2_3.make_problem_instance(num_node = 10, 
                              num_edge = 25, 
                              short_edge_fraction = 0.3, 
                              num_sink = 3, 
                              num_target = 10, 
                              max_node_to_target_association =3)
  for i,v in iteritems(s_2_3.T):
    print i,v
  s_2_3.greedy_set_cover_targets()
  s_2_3.build_backbone_network()
  print "sinks:", s_2_3.sinks
  print "target_connected nodes:",s_2_3.N
  s_2_3.print_graph(s_2_3.adj)
  s_2_3.print_graph(s_2_3.G_p)
  
  
  
  
