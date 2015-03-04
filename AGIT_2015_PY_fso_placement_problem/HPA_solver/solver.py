import map_to_graph 
import heuristic_placement_algo as hpa
import relaxed_ilp_solver_for_placement_algo as rilp
from numpy import sqrt
import random


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
  max_allowed_edge = 4000
  mapfile_folder_name = './map/'
  mapfiles = [
              #'nyc_8th_av_26th_west.osm'
              #'chicago_michigan_avenue.osm'
              #'nyc_hells_kitchen.osm'
              #'washington_dc_franklin_street.osm'
              'washington_dc_monroe_street_v2.osm'
              ]  
 
  target_granularity = 10.0 #meter
  sink_to_target_ratio = 0.01
  n_max_times = 4 #<--vital
  capacity = 10.0
  
  stat_filename = './stat/full_city_stat.txt'
  #dyn_spec_filepath = './dyn_spec/'
  #*************************************
  #                  end of params            
  #*************************************
  random.seed(seed)
  f= open(stat_filename, 'a')
  for mfile in mapfiles:
    mapfilepath = mapfile_folder_name+mfile
    configname = mfile
    #configname = get_filename(mfile)
    #print configname
    #*************************************
    #     map to graph processing          
    #*************************************
    mtg = map_to_graph.MapToGraph() 
    adj, sinks, T, T_N, node, tnode \
    =\
    mtg.generate_graph(fileName= mapfilepath,  
                     target_granularity = target_granularity, 
                     sink_to_node_ratio = sink_to_target_ratio,
                     bounding_box_nodes_only= True)

    #check for impossibilty of number of nodes/edges:
    print "Current Sample: #node:",adj.number_of_nodes()," #edges:",adj.number_of_edges()
    #*************************************
    #     ILP decider          
    #************************************* 
    #if adj.number_of_nodes() > max_allowed_node or adj.number_of_edges() > max_allowed_edge:
    #  print "Skipping sample....too many nodes/edges generated"
    #  continue
    #mtg.generate_visual_map()
    #-----------------------------------------------------------------------
    #*************************************
    #     heuristic solver           
    #************************************* 
    print"Running Heurisitc Placement Algo...."
    n_max = n_max_times * int(sqrt(len(mtg.tnode)))
    #print "DEBUG: number of tnodes:",len(mtg.tnode)
    #print "DEBUG: n_max: ",n_max

    
    #print  "DEBUG: n_max: ", n_max
    hp = hpa.Heuristic_Placement_Algo(    capacity = 1.0, #capacity normalized
                                          adj = adj, 
                                          sinks = sinks, 
                                          T = T, 
                                          T_N = T_N,
                                          n_max = n_max)
                                          #seed = self.seed)
    hp.solve()
    mtg.generate_visual_map(in_adj = hp.G_p)
    #print hp.max_flow
    #print hp.static_max_flow
    #-----------------------------------------------------------------------
    #*************************************
    #    relaxed ILP solver          
    #************************************* 
    '''
    print "Running Relaxed ILP...."
    ilp = rilp.ILP_Relaxed(nmax = n_max,
                            dmax = hp.d_max,
                            adj = adj,
                            T_N = T_N,
                            sinks = sinks)
    ilp.solve()
    
    
    #*************************************
    #    stat WITH ilp         
    #************************************* 
    #print ilp.max_flow
    d_i_ratio = 100.0*hp.max_flow/ilp.max_flow
    s_d_ratio = 100.0*hp.static_max_flow/ilp.max_flow
    #d_f_ratio = 100.0*hp.max_flow/hp.full_adj_max_flow
    static_avg_max_flow, static_avg_upbound_flow = \
            hp.find_static_average_flow()
    print "------------------>s_d_ratio:",s_d_ratio," d_i_ratio:",d_i_ratio# ," d_f_ratio: ",d_f_ratio
 
    stat_string = str(configname)+\
                  ','+str(d_i_ratio)+\
                  ','+str(s_d_ratio)+\
                  ','+str(capacity * hp.max_flow)+\
                  ','+str(capacity * ilp.max_flow)+\
                  ','+str(capacity * hp.full_adj_max_flow)+\
                  ','+str(capacity * hp.static_max_flow)+\
                  ','+str(capacity * static_avg_max_flow)+\
                  ','+str(capacity * static_avg_upbound_flow)
    '''
    #-------------------------------------------------------------------------
    #*************************************
    #    stat without ilp         
    #************************************* 
    static_avg_max_flow, static_avg_upbound_flow = \
            hp.find_static_average_flow()
    s_d_ratio = 100.0*hp.static_max_flow/hp.max_flow
    print "------------------>s_d_ratio:",s_d_ratio
    
    stat_string = str(configname)+\
                  ','+str(s_d_ratio)+\
                  ','+str(capacity * hp.max_flow)+\
                  ','+str(capacity * hp.full_adj_max_flow)+\
                  ','+str(capacity * hp.static_max_flow)+\
                  ','+str(capacity * static_avg_max_flow)+\
                  ','+str(capacity * static_avg_upbound_flow)
    
    #------------------------------------------------------------------
    
    f.write(stat_string+'\n')
    #print hp.sources
    print "\t\twriting dynamic graph spec to file..."
    create_dynamic_graph_spec(filename = './spec/'+str(configname)+'.txt', 
                              g = hp.g1, 
                              sources = hp.sources, 
                              sinks = hp.sinks, 
                              fso_count = 6, 
                              capacity_gbps = 10) 
  f.close()
  
   
    
    
    
    