from xml.etree.cElementTree import iterparse
#import xml.etree.cElementTree as et
import geopy.distance
import numpy as np
from collections import defaultdict
from shapely import geometry as shgm
import time
import networkx as nx
import itertools

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def parse(xml):
  ways = {}
  tags = {}
  refs = []
  way_id = None
  lat= {}
  lon = {}
  building_lat = {}
  building_lon = {}
  max_lat = max_lon = -float('inf')
  min_lat = min_lon = float('inf')
  building_counter = 0
  
  
  context = iterparse(xml, events=("start", "end"))

  context = iter(context)

  # get the root element
  event, root = context.next()
  
  for event, elem in context:
      if event == 'start': continue
      if elem.tag == 'tag':
          tags[elem.attrib['k']] = elem.attrib['v']
      #-----------------------------------------------------#
      #              node processing
      #-----------------------------------------------------#
      elif elem.tag == 'node':
          osmid = int(elem.attrib['id'])
          lat[osmid] = float(elem.attrib['lat'])
          lon[osmid] = float(elem.attrib['lon'])
          tags = {}
      #-----------------------------------------------------#
      #              node ref i.e nd processing
      #-----------------------------------------------------#          
      elif elem.tag == 'nd':
          refs.append(int(elem.attrib['ref']))
      #-----------------------------------------------------#
      #              way_id  processing
      #-----------------------------------------------------# 
      elif elem.tag == 'way_id':
        if elem.attrib['role'] == 'outer':
          way_id = int(elem.attrib['ref'])
          #members.append((int(elem.attrib['ref']), elem.attrib['type'], elem.attrib['role']))
      #-----------------------------------------------------#
      #              way processing
      #-----------------------------------------------------# 
      elif elem.tag == 'way':
          osm_id = int(elem.attrib['id'])
          ways[osm_id] = refs
          if 'building' in tags.keys():
            blat_list = [lat[nid] for nid in refs]
            del blat_list[-1]
            blon_list = [lon[nid] for nid in refs]
            del blon_list[-1]
            building_lat[osm_id] = blat_list
            building_lon[osm_id] = blon_list
            max_lat = max(blat_list+[max_lat])
            max_lon = max(blon_list+[max_lon])
            min_lat = min(blat_list+[min_lat])
            min_lon = min(blon_list+[min_lon])
            building_counter +=1
            #print "DEBUG:building# ",building_counter
            #ways.append((osm_id, tags, refs)) 
          refs = []
          tags = {}
      #-----------------------------------------------------#
      #              relation processing
      #-----------------------------------------------------# 
      elif elem.tag == 'relation':
          osm_id = int(elem.attrib['id'])
          if 'building' in tags.keys() and way_id:
            #<-----process the ways right here
            blat_list = [lat[nid] for nid in ways[way_id]]
            del blat_list[-1]
            blon_list = [lon[nid] for nid in ways[way_id]]
            del blon_list[-1]
            building_lat[osm_id] = blat_list
            building_lon[osm_id] = blon_list
            max_lat = max(blat_list+[max_lat])
            max_lon = max(blon_list+[max_lon])
            min_lat = min(blat_list+[min_lat])
            min_lon = min(blon_list+[min_lon])
            building_counter +=1
            #print "DEBUG:building# ",building_counter
            #relations.append((osm_id, tags, members))
          way_id = None
          tags = {}
      root.clear()

  return building_lat, building_lon, min_lat, min_lon, max_lat, max_lon

def get_cartesian_coord(lat , lon, ref_lat, ref_lon):
  x_ref = geopy.Point(ref_lat, lon)
  y_ref = geopy.Point(lat, ref_lon)
  xy = geopy.Point(lat,lon)
  x = geopy.distance.distance(xy, x_ref).m
  y = geopy.distance.distance(xy, y_ref).m
  return x,y
def transform_to_cartesian_coord(building_lat, building_lon, ref_lat, ref_lon):
  building_x = {}
  building_y = {}
  max_x = max_y = -float('inf')
  for bindx, blats in building_lat.iteritems():
    building_x[bindx] = []
    building_y[bindx] = []
    blons = building_lon[bindx]
    for blat, blon in zip(blats, blons):
      xy = geopy.Point(blat,blon)
      x_ref = geopy.Point(ref_lat, blon)
      y_ref = geopy.Point(blat, ref_lon)
      x = geopy.distance.distance(xy, x_ref).m
      y = geopy.distance.distance(xy, y_ref).m
      building_x[bindx].append(x)
      building_y[bindx].append(y)
      max_x = max(max_x, x)
      max_y = max(max_y, y)
  return building_x, building_y, max_x, max_y

def select_nodes(building_x, building_y, interval_dist = 10):
  node_x =  []
  node_y =  []
  for bid in building_x.keys():
    node_x.extend(building_x[bid])
    node_y.extend(building_y[bid])
  
  #total_nodes = len(node_x)
  total_nodes = len(node_x)
  total_building = len(building_x)
  max_x = max(node_x)
  max_y = max(node_y)
  
  max_x_index = int(np.ceil(max_x / interval_dist))
  max_y_index = int(np.ceil(max_y / interval_dist))
  
  print "DEBUG:starting binning...interval_dist = ",interval_dist
  print "DEBUG: total_nodes:",total_nodes," total buildings:",total_building
  print" DEBUG: max_x:",max_x,"  max_y:",max_y
  #print" DEBUG: max_x_index:",max_x_index," max_y_index:",max_y_index

  bin_node = defaultdict(dict)
  for i in xrange(max_x_index):
    for j in xrange(max_y_index):
      bin_node[i][j] = set()
      

  for nindx in range(total_nodes):
    xbin = int(node_x[nindx]//interval_dist)
    ybin = int(node_y[nindx]//interval_dist)
    bin_node[xbin][ybin].add(nindx)

  chosen_nodes = set()

  for i in xrange(max_x_index):
    #print "DEBUG: i:",i
    for j in xrange(max_y_index):
      node_pool = set(bin_node[i][j])
      
      for k in xrange(max(0,i-1), min( max_x_index , i+2) ):
        for l in xrange(max(0,j-1), min( max_y_index, j+2) ):
          node_pool.update(bin_node[k][l])
          
      node_set = list(set(node_pool))
      
      node_set_size = len(node_pool)
      #if node_set_size > 0:
        #print " Node set size:",node_set_size
      for u in xrange(node_set_size):  
        n1 = node_set[u]
        if n1 in chosen_nodes:
          continue
        is_chosen = True
        for v in xrange(node_set_size):
          n2 = node_set[v]
          if n1==n2 or n2 not in chosen_nodes:
            continue
          x1 = node_x[n1]
          x2 = node_x[n2]
          y1 = node_y[n1]
          y2 = node_y[n2]

          dist = np.sqrt(sum( (a - b)**2 for a, b in zip([x1, y1], [x2, y2])))
          if  dist <= interval_dist:
            is_chosen = False
            break
        if is_chosen:
          chosen_nodes.add(n1)
      
  print "Number of nodes chosen:",len(chosen_nodes)
  del bin_node
  cnode_x = []
  cnode_y = []
  
  while chosen_nodes:
    node_id =  chosen_nodes.pop()
    cnode_x.append(node_x[node_id])
    cnode_y.append(node_y[node_id])
  return cnode_x, cnode_y

def get_linear_index(i,j, max_y_index):
  return  i*(max_y_index + 1) + j

def get_coord(n, max_y_index):
  x = n // (max_y_index + 1)
  y = n % (max_y_index + 1)
  return x, y

def hash_builidings( building_x, building_y, max_x, max_y, box_x_dist, box_y_dist ):
  #total_buildings = len(building_x)

  max_x_index = int(max_x // box_x_dist)
  max_y_index = int(max_y // box_y_dist)
  #print "DEBUG:max_x, max_y, box_x_dist, box_y_dist, max_x_index, max_y_index:",\
  #    max_x, max_y, box_x_dist, box_y_dist, max_x_index, max_y_index
  #t = raw_input("press enter")
  
  bin_building = {}
  for i in xrange(max_x_index+1):
    for j in xrange(max_y_index+1):
      bin_indx = get_linear_index(i, j, max_y_index)
      #print "DEBUG:i,j,bin_indx",i,j,bin_indx
      bin_building[bin_indx] = set()
      
  bids =  building_x.keys()
 # print "DEBUG:total bids",len(bids)
  for bid in bids:
    bxs = building_x[bid]
    bys = building_y[bid]
    #print "DEBUG:bid,bxs,bys",bid,bxs,bys
    max_bx = max(bxs)
    min_bx = min(bxs)
    max_by = max(bys)
    min_by = min(bys)
    max_i =  int( max_bx  // box_x_dist)
    min_i =  int( min_bx  // box_x_dist)
    max_j =  int( max_by  // box_y_dist)
    min_j =  int( min_by  // box_y_dist)
    #print "DEBUG: max_i, min_i, max_j, min_j:",max_i,min_i,max_j,min_j
    for i in range(min_i, max_i+1):
      for j in range(min_j,  max_j+1):
        bin_indx = get_linear_index(i, j, max_y_index) 
        bin_building[bin_indx].add(bid)
        #print "DEBUG::bid,i,j,max_bx, min_bx, max_by, min_by:",\
        #        bid,i,j,bin_indx,max_i, min_i, max_j, min_j 
  
  empty_grid = 0
  #print "DEBUG: ALL BUILDING IDS:",building_x.keys()
  #print "DEBUG: TOTAL BUILDINGS:",len(building_x)
  #for i in xrange(max_x_index+1):
  #  for j in xrange(max_y_index+1):
  #    bin_indx = get_linear_index(i, j, max_y_index)
  #    print "DEBUG:i,j",i,j,bin_building[bin_indx]
      
  #print "DEBUG: TOTAL EMPTY GRID:",1.0*empty_grid/(max_x_index*max_y_index)
  return bin_building, max_x_index, max_y_index, box_x_dist, box_y_dist


def is_intersecting(x1,y1,x2,y2,xs,ys, max_fso_dist = 2000, min_fso_dist = 100):
  line = shgm.LineString([(x1, y1), (x2, y2)])
  polygon = shgm.Polygon(zip(xs,ys))
  if line.crosses(polygon): # and not line.touches(polygon):
    return 0
  else: 
    return 1 

#@def is_intersecting(x1,y1,x2,y2,xs,ys, max_fso_dist = 2000, min_fso_dist = 100):
  
  
  
 
def grid_ray_trace(x0, y0, x1, y1, max_x_index, max_y_index, bbin):
  dx = abs(x0 - x1)
  dy = abs(y0 - y1)
  x = x0
  y = y0
  n = 1 + dx + dy
  x_inc = 1 if x1 > x0 else -1
  y_inc = 1 if y1 > y0 else -1
  error = dx - dy
  dx *= 2
  dy *= 2
  los_bin_set = set()
  while n>0:
    g_num = get_linear_index(x, y, max_y_index)
    # x*(max_x_index-1) + y
    los_bin_set.add(g_num)
    if error >=0:
      x += x_inc
      error -= dy
    else:
      y += y_inc
      error += dx
    n -= 1
  #print "DEBUG:",los_set
  los_bset=set()
  while los_bin_set:
    bin_id = los_bin_set.pop()
    los_bset.update(bbin[bin_id])

  #print "debug lost_bset", los_bset
   
  return los_bset
  #yield los_set
#def get_bin_distance(i,j,k,l,max_x_index,max_y_index):
def withinDistance(i,j,k,l,constraint_distance, x_interval, y_interval ):
  return ( (i-k)**2*x_interval**2 + (j-l)**2*y_interval**2 <= constraint_distance**2)
 

#@profile
def generate_bin_pair_line_intersection(bin_max_x_indx, 
                                        bin_max_y_indx, 
                                        bin_x_interval,
                                        bin_y_interval,
                                        bbin, 
                                        building_x, 
                                        building_y,
                                        threshold_dist,
                                        max_fso_dist):
  max_fso_dist_sq = max_fso_dist * max_fso_dist
  threshold_dist_sq = threshold_dist * threshold_dist
  max_x_index = bin_max_x_indx 
  max_y_index = bin_max_y_indx
  max_x_reach = max_fso_dist//bin_x_interval
  max_y_reach = max_fso_dist//bin_y_interval
  print "DEBUG:max_x_index, max_y_index:",max_x_index, max_y_index
  
  bin_intersection = defaultdict(dict)
  bin_LOS = defaultdict(dict)
  bin_distance = defaultdict(dict)
  
  bin_pair_counter = 0
  total_los_calc = 0
  total_is_intersection_call = 0
  time_t = time.clock()
  max_index = get_linear_index(max_x_index, max_y_index, max_y_index)
  for bin_1 in xrange(max_index+1):
      for bin_2 in xrange(max_index+1):
          
          if bin_pair_counter == 20000:
              print "processing time for 20000 bin-pairs::",time.clock() - time_t
              time_t = time.clock()
              bin_pair_counter = 0
          if bin_1 > bin_2:
            bin_distance[bin_1][bin_2] = bin_distance[bin_2][bin_1]
            if bin_distance[bin_1][bin_2] == 2:
              bin_LOS[bin_1][bin_2] = bin_LOS[bin_2][bin_1]
            if bin_distance[bin_1][bin_2] != 3:
              bin_intersection[bin_1][bin_2] = bin_intersection[bin_2][bin_1]
            continue
          elif bin_1 == bin_2:
            bin_distance[bin_1][bin_2] = 1
            bin_intersection[bin_1][bin_2] = set(bbin[bin_1])
            continue
          bin_pair_counter  += 1
          i,j = get_coord(bin_1, max_y_index)
          k,l = get_coord(bin_2, max_y_index)
          #------window calculation----
          if not withinDistance(i,j,k,l,max_fso_distance,bin_x_interval,bin_y_interval):
            bin_distance[bin_1][bin_2]=3
            continue
          bin_intersection[bin_1][bin_2] = set(grid_ray_trace(i, j, k, l, max_x_index, max_y_index, bbin))
          #print "binIntersection", bin_intersection[bin_1][bin_2]
          #if bin_intersection[bin_1][bin_2]:
          #  print "DEBUG: NonEMPTY GRID:",bin_1,bin_2
          #  t=raw_input("Enter:")
          if withinDistance(i,j,k,l,threshold_dist,bin_x_interval,bin_y_interval):
            bin_distance[bin_1][bin_2]=1
            continue
          bin_distance[bin_1][bin_2]=2
          intersection_status = 1
         
          #put building intersection code here.....
          total_los_calc += 1
          building_set = None
          if bin_1 != bin_2:
            building_set = set(bin_intersection[bin_1][bin_2])
          while building_set:
            bid = building_set.pop()   
            bxs = building_x[bid] 
            bys = building_y[bid]
            
            bin1_x = (i + 0.5) * bin_x_interval
            bin1_y = (j + 0.5) * bin_y_interval
            bin2_x = (k + 0.5) * bin_x_interval
            bin2_y = (l + 0.5) * bin_y_interval
            #intersection_status = 1 #DEBUG!!!!
            total_is_intersection_call += 1
            intersection_status = is_intersecting(x1 = bin1_x, 
                                     y1 = bin1_y, 
                                     x2 = bin2_x, 
                                     y2 = bin2_y, 
                                     xs = bxs, 
                                     ys = bys) 
            if intersection_status == 0:
              break 
          if intersection_status == 1:
            bin_LOS[bin_1][bin_2] = 1
          else:
            bin_LOS[bin_1][bin_2] = 0
          
          #print "DEBUG:i,j,k,l,bin_1, bin_2:",i,j,k,l,bin_1,bin_2,time.clock() - time_t
          
          
  #print "DEBUG: total binLOSPairs:", total_los_calc
  #print "DEBUG: total intersection calls:",total_is_intersection_call
  #print "whole array of bin_intersection", bin_intersection
  return bin_intersection, bin_distance, bin_LOS #, bin_distance_sq
  
       
def calculate_LOS(buildin_x, 
                  buiding_y, 
                  node_x, 
                  node_y, 
                  bbin, 
                  bbin_max_x, 
                  bbin_max_y,
                  bbin_x_interval,
                  bbin_y_interval,
                  max_fso_distance,
                  threshold_distance,
                  bin_pair_intersection,
                  bin_distance,
                  bin_LOS
                  ):
  '''
  print "inside calcLOS", bin
  for i in bin_pair_intersection.keys():
    for j in bin_pair_intersection[i].keys():
      print "i,j:",i,j, bin_pair_intersection[i][j]
  t=raw_input("enter")'''
  node_adj_counter = defaultdict(int)
  total_los_calc = 0
  total_nodes = len(node_x)
  node_degree = [0 for i in xrange(total_nodes)]
  #adj=defaultdict(dict)
  adj = nx.Graph()

  pair_count = 0
  elapsed_time_t = 0
  last_time_t=time.time()
  
  total_is_intersection_call = 0
  for i in range(total_nodes):
    n1_x = node_x[i]
    n1_y = node_y[i]
    
    n1_bin_x = int(n1_x // bbin_x_interval)
    n1_bin_y = int(n1_y // bbin_y_interval)
    
    n1_bin = get_linear_index(n1_bin_x, n1_bin_y, bbin_max_y)
    
    for j in range(i+1, total_nodes):
      #print "DEBUG: inside for loop:i,j,",i,j,bin_pair_intersection
      #t=raw_input("enter:")
      n2_x = node_x[j]
      n2_y = node_y[j]
      n2_bin_x = int(n2_x // bbin_x_interval)
      n2_bin_y = int(n2_y // bbin_y_interval)
      
      n2_bin = get_linear_index(n2_bin_x, n2_bin_y, bbin_max_y)
      pair_count += 1
      
      #adj[i][j]=0
      if bin_distance[n1_bin][n2_bin] == 3:
        continue
      if bin_distance[n1_bin][n2_bin] == 2:
        if bin_LOS[n1_bin][n2_bin] == 1:
          adj.add_edge(i,j)
        continue

      building_set = None
      total_los_calc += 1
      '''
      print "DEBUG: BEFORE ASSIGNMENT--------------------------------"
      print "n1bin, n2bin, before assignment", n1_bin, n2_bin
      for i in bin_pair_intersection.keys():
        for j in bin_pair_intersection[i].keys():
          print "[i][j]:",i,j,"building set:",bin_pair_intersection[i][j]'''
          
      building_set = set(bin_pair_intersection[n1_bin][n2_bin])
      
      #print "after assignment", bin_pair_intersection
      
      intersection_status = 1 # no intersection.
      #if not building_set:
      #    print n1_bin, n2_bin
      while building_set:
        total_is_intersection_call += 1
        bid = building_set.pop()   
        bxs = buildin_x[bid]
        bys = building_y[bid]
        intersection_status = is_intersecting(x1 = n1_x, 
                                 y1 = n1_y, 
                                 x2 = n2_x, 
                                 y2 = n2_y, 
                                 xs = bxs, 
                                 ys = bys) 
        if intersection_status == 0:
          break 
      if intersection_status > 0:
        node_degree[i] += 1
        node_degree[j]  +=1
        adj.add_edge(i,j) #add edge type later
      
      elapsed_time_t += time.time() - last_time_t 
      last_time_t = time.time()
      
      if pair_count%10000 == 0:
        print "DEBUG: elapsed_time:",elapsed_time_t," for point-pairs:",pair_count
  print "DEBUG: total pointLOSpairs:",total_los_calc
  print "DEBUG: total intersection calls:",total_is_intersection_call
  print "DEBUG: Total time for points:",elapsed_time_t," for point-pairs:",pair_count
  print "DEBUG: Total Edges created among nodes:",sum(node_degree)/2
  return adj
  
def generate_visual_map( builindg_x, building_y, node_x, node_y,  adj):
  patches = [] 
  for bid, bxs in building_x.iteritems():
    #print "bid:",i,"------------------------------" 
    bys = building_y[bid]
    pcoord = np.asarray(zip(bxs, bys), dtype = float)
    polygon = Polygon(pcoord, fc='grey')
    #polygon.set_facecolor('none')
    patches.append(polygon) 
    
  
  fig, ax = plt.subplots()
  
  p = PatchCollection(patches, match_original=True)
  ax.add_collection(p)
  #plt.grid(True)
  '''
  if self.box:
    for i in range(self.nbox_dim):
      for j in range(self.nbox_dim):
        xs,ys = zip(*self.box[i][j].exterior.coords)
        plt.plot(xs,ys,'r')'''
  
  #---plot the sink and other nodes----
  #print "DEBUG:@generate_visual_map", self.sinks
  for indx, x in enumerate(node_x):
    y = node_y[indx]
    plt.plot([x],[y],"bo")      
  #--- plot the target nodes------
  #for u in self.tnode:
  #  plt.plot([u.x],[u.y],"ro")
   
  #adj = in_adj
  #if not adj:
  #  adj = self.adj
  ##--adj disal
  #adj = None #<-------edge view disabled---

  if adj:
    for u,v in adj.edges():
      plt.plot([ node_x[u], node_x[v] ],\
               [ node_y[u], node_y[v] ], color = 'g' ,  ls ='dotted')
  
  
  ax.autoscale(enable=True, axis = 'both', tight= True)
  ax.set_aspect('equal', 'box')
  #fig.set_size_inches(100, 100)
  #fig.savefig(self.mapFileName+".pdf", bbox_inches = 'tight')
  plt.show()

  return 
  #return adj
if __name__ == '__main__':
  #xml = './map/lower_manhattan_full.osm'
  #xml = './map/chicago_michigan_avenue.osm'
  #xml = './map/washington_dc_monroe_street.osm'
  xml = './map/manhattan_1_4th.osm'
  #xml = './map/nyc_small_area.osm'
  bin_dist_threshold = 400
  max_fso_distance = 1000
  bbin_x_interval = 100
  bbin_y_interval = 100
  
  
  
  building_lat, \
  building_lon, \
  min_lat, \
  min_lon, \
  max_lat, \
  max_lon = parse(xml = xml)
  
  building_x, building_y, max_x, max_y =\
       transform_to_cartesian_coord(building_lat =  building_lat, 
                               building_lon= building_lon,
                               ref_lat = min_lat,
                               ref_lon = min_lon)
  
  del building_lat, building_lon #save memory
  node_x, node_y = select_nodes(building_x, building_y, interval_dist = 5) 
  total_nodes = len(node_x)
  print" DEBUG: Binning buildings..."
  bin_building, max_bbin_x, max_bbin_y, bbin_x_interval, bbin_y_interval =\
  hash_builidings(building_x = building_x, 
                  building_y = building_y, 
                  max_x = max_x, 
                  max_y = max_y, 
                  box_x_dist = bbin_x_interval, 
                  box_y_dist = bbin_y_interval) 
  

  print "DEBUG: Calculating bin pair int/dist for #nodes:",total_nodes 
  time_t = time.clock()
  bin_pair_intersection, bin_distance, bin_LOS = generate_bin_pair_line_intersection(
                                             bin_max_x_indx = max_bbin_x, 
                                             bin_max_y_indx = max_bbin_y, 
                                             bin_x_interval = bbin_x_interval, 
                                             bin_y_interval = bbin_y_interval, 
                                             bbin = bin_building, 
                                             building_x= building_x,
                                             building_y= building_y,
                                             threshold_dist = bin_dist_threshold, 
                                             max_fso_dist = max_fso_distance)
  #print "debug returned binIntersect", bin_pair_intersection
  elapsed_time_t = time.clock() - time_t
  print "elapsed_time for bin_pair_generation:",elapsed_time_t
  #for v,l in bin_building.iteritems():
  #  v_x, v_y = get_coord(v, max_bbin_y)

   # print "DEBUG: v_x, v_y:",v_x, v_y, l 
  
  t=raw_input("enter:")
  
  '''
  for i,v1 in bin_pair_intersection.iteritems(): 
    for j,v2 in v1.iteritems(): 
      print "DEBUG: i,j, bin_pair_intersection[i][j]:",i, j, bin_pair_intersection[i][j]
         
  for i,v1 in bin_pair_distance.iteritems():
    for j,v2 in v1.iteritems():
      print "i,j,dist:",i,j,v2'''
  
  #temp_t = raw_input("enter:")                                    
  time_t = time.time()
  print "DEBUG: Calculating LOS for #nodes:",total_nodes    
  adj = calculate_LOS(buildin_x = building_x,
                buiding_y = building_y, 
                node_x = node_x, 
                node_y = node_y, 
                bbin = bin_building, 
                bbin_max_x = max_bbin_x, 
                bbin_max_y = max_bbin_y,
                bbin_x_interval = bbin_x_interval,
                bbin_y_interval = bbin_y_interval,
                max_fso_distance=max_fso_distance,
                threshold_distance = bin_dist_threshold,
                bin_pair_intersection = bin_pair_intersection,
                bin_distance = bin_distance,
                bin_LOS =  bin_LOS)
  #generate_visual_map( building_x, building_y, node_x, node_y,  adj )
  print" DEBUG: Done...."
  
 

  
  
  
  
  
  