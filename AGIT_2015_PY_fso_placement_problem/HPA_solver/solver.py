import map_to_graph 
import heuristic_placement_algo as hpa
import relaxed_ilp_solver_for_placement_algo as rilp
import random
if __name__ == '__main__':
  #*************************************
  #                  params            
  #*************************************
  seed = 101973  
  max_allowed_node = 200
  max_allowed_edge = 400
  mapfiles = ['./map/dallas_v1.osm',\
           './map/dallas_v2.osm',\
           './map/dallas_v3.osm']
           #'./map/nyc_hells_kitchen.osm']
  target_to_node_ratio = 0.4
  sink_to_node_ratio = 0.05
  n_max_ratio = 0.5
  #*************************************
  #                  end of params            
  #*************************************
  random.seed(seed)
  for mfile in mapfiles:
    #*************************************
    #     map to graph processing          
    #*************************************
    mtg = map_to_graph.MapToGraph() 
    adj, sinks, T, T_N, node, tnode \
    =\
    mtg.generate_graph(fileName= mfile,  
                     target_to_node_ratio = target_to_node_ratio, 
                     sink_to_node_ratio = sink_to_node_ratio)
    #-----------------------------------------------------------------------
    #*************************************
    #     heuristic solver          
    #************************************* 
    #check for impossibilty of number of nodes/edges:
    print "DEBUG: #node:",adj.number_of_nodes()," #edges:",adj.number_of_edges()
    if adj.number_of_nodes() > max_allowed_node or adj.number_of_edges() > max_allowed_edge:
      print "Skipping sample....too many nodes/edges generated"
      continue
    n_max = int(n_max_ratio * adj.number_of_nodes())
    #print  "DEBUG: n_max: ", n_max
    hp = hpa.Heuristic_Placement_Algo(    capacity =1.0, 
                                          adj = adj, 
                                          sinks = sinks, 
                                          T = T, 
                                          T_N = T_N,
                                          n_max = n_max)
                                          #seed = self.seed)
    hp.solve()
    print hp.max_flow
    #-----------------------------------------------------------------------
    #*************************************
    #    relaxed ILP solver          
    #************************************* 
    ilp = rilp.ILP_Relaxed(nmax = n_max,
                            dmax = hp.d_max,
                            adj = adj,
                            T_N = T_N,
                            sinks = sinks)
    ilp.solve()
    print ilp.max_flow
    percentage = 100.0*hp.max_flow/ilp.max_flow
    print "------------------>",percentage
    #mtg.generate_visual_map()
    
    
    
    