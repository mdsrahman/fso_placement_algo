'''
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
'''
import networkx as nx
import random 
  
class Step_4:
  def __init__(self, seed = 101349):
    print "Initializing Step_4....."
    random.seed(seed)  
    
  def print_graph(self, g):
    print "graph_name:",g.graph['name']
    print "---------------------------"
    print "connected:",nx.is_connected(g)
    print "num_edges:",g.number_of_edges()
    
    for n in g.nodes():
      print n 
    for e in g.edges():
      print e
      
  def make_problem_instance(self,
                            num_nodes, 
                            num_edges, 
                            num_source, 
                            num_sink,
                            max_node_time):
    #-----first makes the starting subgraph g1=(v1,e1)------------------------------
    while True:
      self.g1 = nx.dense_gnm_random_graph(num_nodes,num_edges)
      if nx.is_connected(self.g1):
        break
    self.g1.graph["name"]="subgraph"
    #self.print_graph(self.g1)
    #-------------end of generating connected graph g1---------------------------------
    
    #--------------now generate the original graph based on g1-------------------------
    self.g=nx.Graph(self.g1)
    self.g.graph["name"]="input_graph"
    
    g1_node_list = self.g1.nodes()
    g1_max_degree = max(self.g1.degree(g1_node_list).values())
    
    for i in range(self.g1.number_of_nodes(),self.g1.number_of_nodes()*max_node_time):
      self.g.add_node(i)
      #make sure it hooks up to at least one node
      node_no = random.randint(0,self.g1.number_of_nodes())
      self.g.add_edge(node_no, i)
    
    #now add more edges between the newly added nodes:
    for i in range(self.g1.number_of_nodes()+1, self.g.number_of_nodes()):
      g_node_list = self.g.nodes()
      g_node_list.remove(i)
      
      num_edges =  random.randint(0,g1_max_degree)
      node_list = random.sample(g_node_list, num_edges)
      for j in node_list:
        u = min(i,j)
        v = max(i,j)
        self.g.add_edge(u,v)

    
    #-----end of generation of input graph g-------------------------------------------------
    
    #---------now randomly choose source and sink nodes-------------------------------------
    g1_node_list = self.g1.nodes()
    self.sources = random.sample(g1_node_list,num_source)
    g1_node_list =  list(set(g1_node_list) - set(self.sources))
    self.sinks = random.sample(g1_node_list,num_sink)
    #now add edges to these super sources and super sinks
    self.g1.add_node('src')
    for i in self.sources:
      self.g1.add_edge('src',i)
    
    self.g1.add_node('snk')
    for i in self.sinks:
      self.g1.add_edge(i,'snk')
    
    print "sources:",self.sources
    print "sinks:",self.sinks
    
    self.print_graph(self.g1)
    self.print_graph(self.g)
    
if __name__ == '__main__':
  print "starting run..."
  
  s_4 = Step_4()
  
  num_nodes_in_flow_graph = 10
  num_edges_in_flow_graph = 25
  input_graph_node_time = 3
  num_source = 2
  num_sink = 2
  
  s_4.make_problem_instance(num_nodes_in_flow_graph, 
                            num_edges_in_flow_graph, 
                            num_source, 
                            num_sink,
                            input_graph_node_time)
  
  
  