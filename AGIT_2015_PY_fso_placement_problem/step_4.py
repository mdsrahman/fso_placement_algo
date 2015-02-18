import networkx as nx
import random 
import networkx.algorithms.flow as flow
  
class Step_4:
  def __init__(self, seed = 101349):
    print "Initializing Step_4....."
    random.seed(seed)  
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
    
    self.g1.add_node('src')
    for i in self.sources:
      self.g1.add_edge('src',i)
    
    self.g1.add_node('snk')
    for i in self.sinks:
      self.g1.add_edge(i,'snk')
      
    edge_set =  g.edges()
    for x,y in edge_set:
      g.edge[x][y]['capacity'] = capacity
      
    #now override the supersrc-->src capacities and sink-->supersink capacities
    for n in src_list:
      g.edge['src'][n]['capacity'] = float('inf')
    
    for n in snk_list:
      g.edge[n]['snk']['capacity'] = float('inf')
      
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
    available_nodes = list(set(self.g.nodes()) - set(self.g1.nodes()))
    #print available_nodes
    backbone_nodes = self.g1.nodes()
    
    while(available_nodes):
      max_benefit = - float('inf')
      max_i = max_j = -1
      new_node_list = None
      for i in backbone_nodes:
        for j in backbone_nodes:
          if i !=j:
            #step i) run max-flow and get the dynamic graph g1 and compute the residual graph
            r = flow.shortest_augmenting_path(G=s_4.g1, s='src', t='snk', capacity = 'capacity')
            r = self.compute_residual_graph(g=r, capacity = 'capacity', flow='flow')
            #step ii) find source-potential for i on r
            r1 = flow.shortest_augmenting_path(G=r, s='src',t=i, capacity = 'capacity')
            i_potential = r1.graph['flow_value']
            #step iii) find sink-potential for j on r
            r1 = flow.shortest_augmenting_path(G=r, s=j,t='snk', capacity = 'capacity')
            j_potential = r1.graph['flow_value']
            #step iv) find shortest path from i to j on g
            i_j_path = nx.shortest_path(self.g, i, j)
            
            if len(i_j_path)<=2: 
              continue
            new_nodes_on_i_j_path =  set(self.g1.nodes()) - set(i_j_path)
            new_node_count = len(new_nodes_on_i_j_path )
            if  new_node_count < 1:
              continue
            
            #step v) calculate benefit, if the benefit>max, save i,j,benefit,path(i->j)
            i_j_path_benefit = min(i_potential, j_potential)/(1.0 * new_node_count)
            print "DEBUG: (i,j,path_benefit):",i,",",j,",",i_j_path_benefit
            if i_j_path_benefit > max_benefit:
              max_benefit = i_j_path_benefit
              max_i = i
              max_j = j
              new_node_list = new_nodes_on_i_j_path
            #t= raw_input("press enter:")
      #step vi) update graph g1 with new nodes,edges from i,j
      if max_i ==-1:
        break #no i,j found
      #otherwise add the paths:
      
    return
  
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
  
  capacity = 1.0
  s_4.add_capacity(g = s_4.g1, capacity = capacity)
  #s_4.run_step_4()
  #'''
  #now run the max_flow
  #K= s_4.g1.to_directed()
  #ets =  K.edges(K.nodes(),data=True)
  
  #for e in ets:
    #print e
  
  #flow_value, flow_dict = nx.maximum_flow(K, 'src', 'snk')
  r = flow.shortest_augmenting_path(G=s_4.g1, s='src', t='snk', capacity = 'capacity')
  flow_value = r.graph['flow_value']
  print flow_value

  #now print the edges:
  for e in r.edges(data=True):
    print e
  r = s_4.compute_residual_graph(r)
  print "residual graph"
  for e in r.edges(data=True):
    print e
  #call shortest_augmenting_path again
  r1 = flow.shortest_augmenting_path(G=r, s=8, t='snk', capacity = 'capacity')
  print "r1 flow value:",r1.graph['flow_value']
  
  print "is r modified?-------"
  for e in r.edges(data=True):
    print e
  print "r flow value:",r.graph['flow_value']
  
  print "is r1 modified?-------"
  for e in r1.edges(data=True):
    print e
  #now print the edges:
  #for e in r1.edges(data=True):
  #  print e
  #for u in R.nodes():
  #  for v in R.nodes():
  #    if u not in ['src','snk',v] and v not in ['src','snk',u]:
  #      print nx.shortest_path(s_4.g, u, v)
  
  #print nx.shortest_path(s_4.g, 'src', v)   
'''
  flow_dict = R.graph['flow_dict']
  print flow_value
  for i in flow_dict:
    for j in flow_dict[i]:
      print "(",i,",",j,"): ",flow_dict[i][j]
  '''
  #'''
  #s_4.print_graph(s_4.g1)
  
  
  
  