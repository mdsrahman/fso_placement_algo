import map_to_graph 
import heuristic_placement_algo as hpa
import relaxed_ilp_solver_for_placement_algo as rilp
import random
import ntpath

def get_filename(mfile):
    ntpath.basename(mfile)
    head, tail = ntpath.split(mfile)
    return tail or ntpath.basename(head)

def create_dynamic_graph_spec(filename, g, sources, sinks, fso_count = 6, capacity_gbps =  10):
  f=open(filename,"w")
  #-------trafficSources-------:
  f.write("trafficSources:\n")
  for n in g.nodes():
    f.write(str(n)+"\n")
  f.write('\n')
  
  
  #---FSONodes:
  f.write("FSONodes:\n")
  nodes = g.nodes()
  for n in nodes:
    for fso in range(1, fso_count+1):
      f.write(str(n)+"_fso"+str(fso)+"\n")
  f.write('\n')   
  
  #---FSOLinks:---
  f.write("FSOLinks:\n")   
  edges = g.edges()
  for u,v in edges:
    for fso1 in range(1, fso_count+1):
      for fso2 in range(1, fso_count+1):  
        #"0_0_fso2To2_0_fso1 10Gbps"
        f_text = str(u)+"_fso"+str(fso1)+"To"\
                +str(v)+"_fso"+str(fso2)+" "+str(capacity_gbps)+"Gbps\n" 
        f.write(f_text)
        f_text = str(v)+"_fso"+str(fso2)+"To"\
                +str(u)+"_fso"+str(fso1)+" "+str(capacity_gbps)+"Gbps\n" 
        f.write(f_text)
  f.write('\n')   
  #------gateways--:
  f.write('gateways:\n')
  for n in sinks:
    f.write(str(n)+"\n")
  f.write('\n')
  
  f.close()
  return 
if __name__ == '__main__':
  #*************************************
  #                  params            
  #*************************************
  seed = 101973  
  max_allowed_node = 400
  max_allowed_edge = 3000
  mapfiles = ['./map/dallas_v1.osm',
           './map/dallas_v3.osm'] 

  target_granularity = 10.0 #meter
  sink_to_node_ratio = 0.05
  n_max_ratio = 0.5
  capacity = 10.0
  
  stat_filename = './stat/city_fragment.txt'
  #dyn_spec_filepath = './dyn_spec/'
  #*************************************
  #                  end of params            
  #*************************************
  random.seed(seed)
  f= open(stat_filename, 'a')
  for mfile in mapfiles:
    configname = get_filename(mfile)
    #print configname
    #*************************************
    #     map to graph processing          
    #*************************************
    mtg = map_to_graph.MapToGraph() 
    adj, sinks, T, T_N, node, tnode \
    =\
    mtg.generate_graph(fileName= mfile,  
                     target_granularity = target_granularity, 
                     sink_to_node_ratio = sink_to_node_ratio)
    #-----------------------------------------------------------------------
    #*************************************
    #     heuristic solver          
    #************************************* 
    #check for impossibilty of number of nodes/edges:
    print "Current Sample: #node:",adj.number_of_nodes()," #edges:",adj.number_of_edges()
    if adj.number_of_nodes() > max_allowed_node or adj.number_of_edges() > max_allowed_edge:
      print "Skipping sample....too many nodes/edges generated"
      continue
    print"Running Heurisitc Placement Algo...."
    n_max = int(n_max_ratio * adj.number_of_nodes())
    #print  "DEBUG: n_max: ", n_max
    hp = hpa.Heuristic_Placement_Algo(    capacity = 1.0, #capacity normalized
                                          adj = adj, 
                                          sinks = sinks, 
                                          T = T, 
                                          T_N = T_N,
                                          n_max = n_max)
                                          #seed = self.seed)
    hp.solve()
    #print hp.max_flow
    #print hp.static_max_flow
    #-----------------------------------------------------------------------
    #*************************************
    #    relaxed ILP solver          
    #************************************* 
    print "Running Relaxed ILP...."
    ilp = rilp.ILP_Relaxed(nmax = n_max,
                            dmax = hp.d_max,
                            adj = adj,
                            T_N = T_N,
                            sinks = sinks)
    ilp.solve()
    #print ilp.max_flow
    d_i_ratio = 100.0*hp.max_flow/ilp.max_flow
    s_d_ratio = 100.0*hp.static_max_flow/ilp.max_flow
    static_avg_max_flow, static_avg_upbound_flow = \
            hp.find_static_average_flow()
    print "------------------>",s_d_ratio, d_i_ratio
    mtg.generate_visual_map(in_adj = hp.G_p)
    #current tasks:-----
    '''
    i) create traffic pattern
    ii)create dynamci_graph_spec .txt file (file name same as osm file name)
    iii)save the stat in file: data fields with city_map_fragments
      a) osmfilename, 
        dyn_rilp_max_flow_ratio,
        stati_dyn_max_flow_ratio,
        dyn_max_flow,
        rilp_max_flow,
        static_max_flow,
        static_avg_flow_for_patterns,
        total_adj_nodes, 
        total_bbone_nodes,
        total_static_nodes   
    '''
    #create traffic_pattern
    
    create_dynamic_graph_spec(filename = './spec/'+str(configname)+'.txt', 
                              g = hp.g1, 
                              sources = hp.sources, 
                              sinks = hp.sinks, 
                              fso_count = 6, 
                              capacity_gbps = 10) 
    stat_string = str(configname)+\
                  ','+str(d_i_ratio)+\
                  ','+str(s_d_ratio)+\
                  ','+str(capacity * hp.max_flow)+\
                  ','+str(capacity * ilp.max_flow)+\
                  ','+str(capacity * hp.static_max_flow)+\
                  ','+str(capacity * static_avg_max_flow)+\
                  ','+str(capacity * static_avg_upbound_flow)
    f.write(stat_string+'\n')
    #print hp.sources
  f.close()
  
   
    
    
    
    