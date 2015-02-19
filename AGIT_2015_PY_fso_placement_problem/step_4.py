import networkx as nx
import random 
import networkx.algorithms.flow as flow
  
class Step_4:
  def __init__(self, capacity = 1.0, seed = 101349):
    print "Initializing Step_4....."
    random.seed(seed)  
    self.capacity = capacity
    self.g1 = None # the backbone graph g=(y_i, b_ij)
    self.g =  None # the full graph g1=(x_i, e_ij)
    
  def print_graph(self, g):
    print "graph_name:",g.graph['name']
    print "---------------------------"
    print "connected:",nx.is_connected(g)
    print "num_edges:",g.number_of_edges()
    
    for n in g.nodes(data=True):
      print n 
    for e in g.edges(data=True):
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

    
    print "sources:",self.sources
    print "sinks:",self.sinks
    
    #self.print_graph(self.g1)
    #self.print_graph(self.g)
    
  def add_capacity(self, g, capacity=1.0, src_list=None,  snk_list = None,):
    if src_list == None: src_list=self.sources
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
    self.print_graph(g)
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
    print "DEBUG:backbone_nodes:",backbone_nodes
    available_nodes = list(set(self.g.nodes()) - set(backbone_nodes))
    print available_nodes
    
    
    while(available_nodes):
      d = nx.Graph(self.g1)
      print "DEBUG:type(d):",type(d)
      d = self.add_capacity(g = d, capacity = self.capacity)
      print "DEBUG:type(d):",type(d)
      print d.has_node('src')
      max_benefit = - float('inf')
      max_i = max_j = -1
      new_node_list = None
      
      for i in backbone_nodes:
        for j in backbone_nodes:
          print "DEBUG:(i,j):",i,j
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
              print "DEBUG:p:",p
              #print "DEBUG:available_set:",available_nodes
              available_node_count = len(p) - len(set(p) - set(available_nodes))
              if available_node_count > minimum_node_from_availabe_set:
                minimum_node_from_availabe_set = available_node_count
                min_path = p
              #find the minimum number of external-node shortest paths
              
            #i_j_path = nx.shortest_path(self.g, i, j)
            
            i_j_path = min_path
            print "DEBUG: i_j_path: ",i_j_path
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
            print "DEBUG:new_nodes_i_j_path:",new_nodes_on_i_j_path
            print "DEBUG:i_potential:",i_potential
            print "DEBUG:j_potential:",j_potential
            print "DEBUG: (i,j,path_benefit):",i,",",j,",",i_j_path_benefit
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
      print "DEBUG: new_nodes_added:",new_node_list
      print "DEBUG: available_nodes:",available_nodes
      for n in new_node_list:
        self.g1.add_node(n)
        available_nodes.remove(n)

      for n in new_node_list: 
        for nbr in self.g[n]:
          if nbr in self.g1.nodes():
            self.g1.add_edge(n,nbr)
    return
  
if __name__ == '__main__':
  print "starting run..."
  
  s_4 = Step_4( capacity = 1.0)
  
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
  
  capacity = 1.0
  #s_4.add_capacity(g = s_4.g1, capacity = capacity)
  s_4.run_step_4()

  
  
  
  