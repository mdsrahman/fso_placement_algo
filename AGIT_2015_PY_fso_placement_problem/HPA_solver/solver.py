import map_to_graph 
import heuristic_placement_algo as hpa
import relaxed_ilp_solver_for_placement_algo as rilp
from numpy import sqrt
import random
import cPickle as pkl
import networkx as nx

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

def save_graph_data(configpath, adj, T, T_N, sinks, node, tnode):
    #nx.write_gml(self.adj, configname+'.gml')
  pkl_data={}
  pkl_data['adj'] = adj
  pkl_data['T']= T
  pkl_data['T_N']=T_N
  pkl_data['sinks'] = sinks
  pkl_data['node'] =node 
  pkl_data['tnode'] = tnode
  pkl.dump(obj = pkl_data, file = open(configpath,'wb'))
  return


def load_graph_data(configpath):
  pkl_data = pkl.load( open(configpath, "rb" ) )
  return pkl_data['adj'], pkl_data['T'], pkl_data['T_N'],\
         pkl_data['sinks'], pkl_data['node'], pkl_data['tnode']

if __name__ == '__main__':
  #*************************************
  #                  params            
  #*************************************
  seed = 101973  
  max_allowed_node = 400
  max_allowed_edge = 4000
  mapfile_folder_name = './map/'
  mapfiles = [
              'nyc_8th_av_26th_west.osm'
              #'chicago_michigan_avenue.osm'
              #'nyc_hells_kitchen.osm'
              #'washington_dc_franklin_street.osm'
              #'washington_dc_monroe_street_v2.osm'
              ]  
 
  target_granularity = 10.0 #meter
  sink_to_target_ratio = 0.1
  n_max_times = 2 #<--vital
  capacity = 10.0
  fso_per_node = 4
  
  stat_filename = './stat/frag_city_stat.txt'
  graph_data_dump_dir = './graph_data/'
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
    #mtg.save_graph(configname = graph_data_dump_dir+mfile)
    #check for impossibilty of number of nodes/edges:
    #save graph data
    save_graph_data(configpath = graph_data_dump_dir+mfile+'.pkl', 
                    adj = adj, 
                    T = T, 
                    T_N = T_N, 
                    sinks = sinks,
                    node = node,
                    tnode = tnode)
    mtg.generate_visual_map()
    
    #--***************************************end of graph generation and saving***********
    dadj,  dT, dT_N, dsinks, dnode, dtnode = \
      load_graph_data(configpath = graph_data_dump_dir+mfile+'.pkl')
    #*************************************
    #    run-breaker, in too many nodes/edges          
    #************************************* 
    print "Current Sample: #node:",adj.number_of_nodes()," #edges:",adj.number_of_edges()
    if adj.number_of_nodes() > max_allowed_node or adj.number_of_edges() > max_allowed_edge:
      print "Skipping sample....too many nodes/edges generated"
      continue
 
    #-----------------------------------------------------------------------
    #*************************************
    #     heuristic solver           
    #************************************* 
    print"Running Heurisitc Placement Algo...."
    n_max = n_max_times * int(sqrt(len(dtnode)))

    
    hp = hpa.Heuristic_Placement_Algo(    capacity = 1.0, #capacity normalized
                                          adj = dadj, 
                                          sinks = dsinks, 
                                          T = dT, 
                                          T_N = dT_N,
                                          n_max = n_max,
                                          static_d_max = fso_per_node)
                                          #seed = self.seed)
    hp.solve()
    print "-----DEBUG--------------------"
    #hp.print_graph(g = adj)
    #hp.print_graph(g = hp.bg)
    #hp.print_graph(g = hp.G_p)
    
    #hp.print_graph(g = hp.adj)
    #hp.print_graph(g = hp.bg)
    #hp.print_graph(g = hp.G_p)
    print "-----END DEBUG--------------------"
    
    mtg.generate_visual_map(adj = hp.G_p)
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
    
    
    #*************************************
    #    stat WITH ilp         
    #************************************* 
    #print ilp.max_flow
    d_i_ratio = 100.0*hp.max_flow/ilp.max_flow
    s_d_ratio = 100.0*hp.static_max_flow/hp.max_flow
    #d_f_ratio = 100.0*hp.max_flow/hp.full_adj_max_flow
    static_avg_max_flow, static_avg_upbound_flow = \
            hp.find_static_average_flow()
    print "------------------>s_d_ratio:",s_d_ratio," d_i_ratio:",d_i_ratio# ," d_f_ratio: ",d_f_ratio
 
    stat_string = str(configname)+\
                  ','+str(d_i_ratio)+\
                  ','+str(s_d_ratio)+\
                  ','+str(capacity * hp.max_flow)+\
                  ','+str(capacity * ilp.max_flow)+\
                  ','+str(capacity * hp.static_max_flow)+\
                  ','+str(capacity * static_avg_max_flow)+\
                  ','+str(capacity * static_avg_upbound_flow)
    
    #-------------------------------------------------------------------------
    f.write(stat_string+'\n')
    #print hp.sources
    print "\t\twriting dynamic graph spec to file..."
    create_dynamic_graph_spec(filename = './spec/'+str(configname)+'.txt', 
                              g = hp.g1, 
                              sources = hp.sources, 
                              sinks = hp.sinks, 
                              fso_count = fso_per_node, 
                              capacity_gbps = 10) 
  f.close()
  print "*****************solved for map:",configname
  
  
  
   
    
    
    
    