from xml.etree.cElementTree import iterparse
#import xml.etree.cElementTree as et
import geopy.distance
import numpy as np
from collections import defaultdict
from shapely import geometry as shgm
import time
import networkx as nx
import itertools

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
  
def hash_builidings( building_x, building_y, max_x, max_y, box_x_dist, box_y_dist ):
  #total_buildings = len(building_x)

  max_x_index = int(max_x // box_x_dist)
  max_y_index = int(max_y // box_y_dist)
  print "DEBUG:max_x, max_y, box_x_dist, box_y_dist, max_x_index, max_y_index:",\
      max_x, max_y, box_x_dist, box_y_dist, max_x_index, max_y_index
  t = raw_input("press enter")
  
  bin_building = {}
  for i in xrange(max_x_index):
    for j in xrange(max_y_index):
      bin_indx = i*(max_x_index -1 )+j
      print "DEBUG:",bin_indx
      bin_building[bin_indx] = set()
      
  bids =  building_x.keys()
  print "DEBUG:total bids",len(bids)
  for bid in bids:
    bxs = building_x[bid]
    bys = building_y[bid]
    max_i =  int( max(bxs)  // box_x_dist)
    min_i =  int( min(bxs)  // box_x_dist)
    max_j =  int( max(bys)  // box_y_dist)
    min_j =  int( min(bys)  // box_y_dist)
    #print "DEBUG: max_i, min_i, max_j, min_j:",max_i,min_i,max_j,min_j
    for i in range(min_i, max_i):
      for j in range(min_j,  max_j):
        bin_indx = i*( max_x_index -1 ) + j
        bin_building[bin_indx].add(bid)
        
  for b in sorted(bin_building.keys()):
    i = b // box_y_dist
    j = b% box_y_dist
    print b, i,j, bin_building[b]
  t=raw_input("enter:")
  
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
  n = 1+ dx + dy
  x_inc = 1 if x1 > x0 else -1
  y_inc = 1 if y1 > y0 else -1
  error = dx - dy
  dx *= 2
  dy *= 2
  
  los_bin_set = set()
  while n>0:
    g_num = x*(max_x_index-1) + y
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
  return los_bset
  #yield los_set
#def get_bin_distance(i,j,k,l,max_x_index,max_y_index):
  
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
  bin_distance = defaultdict(dict)
  bin_pair_counter = 0
  total_los_calc = 0
  total_is_intersection_call = 0
  time_t = time.clock()
  for i in xrange(max_x_index):
    for j in xrange(max_y_index): 
      k_min = max(0, i - max_x_reach)
      k_max = min(i+max_x_reach, max_x_index)    
      l_min = max(0, j - max_y_reach)
      l_max = min(j+max_y_reach, max_y_index)
      
      for k in xrange(k_min, k_max):
        for l in xrange(l_min, l_max):
          bin_1 = i*(max_x_index-1) + j
          bin_2 = k*(max_x_index-1) + l
              
          if bin_1 >= bin_2:
            continue
          bin_pair_counter  += 1
          bin_intersection[bin_1][bin_2] = grid_ray_trace(i, j, k, l, max_x_index, max_y_index, bbin)
          
          bin1_x = (i + 0.5) * bin_x_interval
          bin1_y = (j + 0.5) * bin_y_interval
          bin2_x = (k + 0.5) * bin_x_interval
          bin2_y = (l + 0.5) * bin_y_interval
          
          bin_distance_sq = (bin1_x - bin2_x)**2 + (bin1_y - bin2_y)**2 
          bin_distance[bin_1][bin_2] = 0 # Not an edge YET
          intersection_status = 1 #assuming not intersection
          if bin_distance_sq <= threshold_dist_sq:
            bin_distance[bin_1][bin_2] = 2 # edge by point-method
          elif bin_distance_sq > threshold_dist_sq and bin_distance_sq <= max_fso_dist_sq:
            #put building intersection code here.....
            total_los_calc += 1
            building_set = None
            if bin_1 != bin_2:
              building_set = bin_intersection[bin_1][bin_2]
            while building_set:
              bid = building_set.pop()   
              bxs = building_x[bid] 
              bys = building_y[bid]
              
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
              bin_distance[bin_1][bin_2] =  1 
          
          #print "DEBUG:i,j,k,l,bin_1, bin_2:",i,j,k,l,bin_1,bin_2,time.clock() - time_t
          
          if bin_pair_counter == 10000:
            print "processing time for 10000 binWindow-pairs::",time.clock() - time_t
            time_t = time.clock()
            bin_pair_counter = 0
  print "DEBUG: total binLOSPairs:", total_los_calc
  print "DEBUG: total intersection calls:",total_is_intersection_call
  return bin_intersection, bin_distance #, bin_distance_sq
  
       
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
                  bin_pair_intersection,
                  bin_pair_distance
                  ):
  node_adj_counter = defaultdict(int)
  total_los_calc = 0
  total_nodes = len(node_x)
  max_fso_distance_sq = max_fso_distance*max_fso_distance  
  adj =nx.Graph() 
  adj.graph['name']='Adjacency Graph'
  #--bin all the nodes first-----#
  node_bin_x= []
  node_bin_y= []

  pair_count = 0
  elapsed_time_t = 0
  last_time_t=time.time()
  
  total_is_intersection_call = 0
  max_bin_x_spread = int(max_fso_distance // bbin_x_interval)
  max_bin_y_spread = int(max_fso_distance // bbin_y_interval)
  for i in range(total_nodes-1):
    n1_x = node_x[i]
    n1_y = node_y[i]
    n1_bin_x = int(n1_x // bbin_x_interval)
    n1_bin_y = int(n1_y // bbin_x_interval)
    
    n1_bin = n1_bin_x*(bbin_max_x-1)+n1_bin_y
    for j in range(i+1, total_nodes):
      n2_x = node_x[j]
      n2_y = node_y[j]
      n2_bin_x = int(n2_x // bbin_x_interval)
      n2_bin_y = int(n2_y // bbin_x_interval)
      
      n2_bin = n2_bin_x*(bbin_max_x-1)+n2_bin_y
      pair_count += 1
      bin1_x = (n1_bin_x + 0.5) * bbin_x_interval
      bin1_y = (n1_bin_y + 0.5) * bbin_y_interval
      bin2_x = (n2_bin_x + 0.5) * bbin_x_interval
      bin2_y = (n2_bin_y+  0.5) * bbin_y_interval
          
      bin_distance_sq = (bin1_x - bin2_x)**2 + (bin1_y - bin2_y)**2 
       
      if bin_distance_sq > max_fso_distance * max_fso_distance:
        continue
      #------check whether to calculate intersection----
      calculate_los = True
      
      if n1_bin != n2_bin:
        if n1_bin > n2_bin: #swap
            n1_bin, n2_bin = n2_bin, n1_bin
        #print "DEBUG: n1_bin, n2_bin:", n1_bin, n2_bin
        bin_los_status = bin_pair_distance[n1_bin][n2_bin]
        if  bin_los_status <= 1 :
          calculate_los = False
          print n1_bin, n2_bin,"calculateLosFalse Within Range++++++++++++"
          if bin_los_status == 1:
            intersection_status = 1
      #--------------end of check-----------------------
      if calculate_los:
        #print "DEBUG: n1_bin,n2_bin:", n1_bin, n2_bin
        building_set = None
        total_los_calc += 1
        if n1_bin == n2_bin:
          building_set = bbin[n1_bin]
        else:
          building_set = bin_pair_intersection[n1_bin][n2_bin]
        if not building_set and n1_bin != n2_bin:
            print n1_x, n1_y, n2_x, n2_y, n1_bin, n2_bin, "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
        while building_set:
          total_is_intersection_call += 1
          bid = building_set.pop()   
          bxs = buildin_x[bid]
          bys = building_y[bid]
          #time_t = time.clock()
          #intersection_status = 1 #DEBUG!!!!
          intersection_status = is_intersecting(x1 = n1_x, 
                                   y1 = n1_y, 
                                   x2 = n2_x, 
                                   y2 = n2_y, 
                                   xs = bxs, 
                                   ys = bys) 
          
          #print "elapsed_time:",time.clock() - time_t,"second"
          
          if intersection_status == 0:
            break 
      if intersection_status > 0:
        node_adj_counter[i] += 1
        node_adj_counter[j]+=1
        continue
        #adj.add_edge(i,j) #add edge type later
      
      elapsed_time_t += time.time() - last_time_t 
      last_time_t = time.time()
      
      if pair_count%10000 == 0:
        print "DEBUG: elapsed_time:",elapsed_time_t," for point-pairs:",pair_count
  print "DEBUG: total pointLOSpairs:",total_los_calc
  print "DEBUG: total intersection calls:",total_is_intersection_call
  print "DEBUG: Total time for points:",elapsed_time_t," for point-pairs:",pair_count
  #print "DEBUG: Total Edges created among nodes:",sum(node_adj_counter)/2
  return adj
if __name__ == '__main__':
  #xml = './map/lower_manhattan_full.osm'
  #xml = './map/chicago_michigan_avenue.osm'
  xml = './map/washington_dc_monroe_street.osm'
  #xml = './map/manhattan_1_4th.osm'
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
  node_x, node_y = select_nodes(building_x, building_y, interval_dist = 30) 
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
  bin_pair_intersection, bin_pair_distance = generate_bin_pair_line_intersection(
                                             bin_max_x_indx = max_bbin_x, 
                                             bin_max_y_indx = max_bbin_y, 
                                             bin_x_interval = bbin_x_interval, 
                                             bin_y_interval = bbin_y_interval, 
                                             bbin = bin_building, 
                                             building_x= building_x,
                                             building_y= building_y,
                                             threshold_dist = bin_dist_threshold, 
                                             max_fso_dist = max_fso_distance)
  elapsed_time_t = time.clock() - time_t
  print "elapsed_time for bin_pair_generation:",elapsed_time_t
  for v,l in bin_building.iteritems():
    v_x = v//bbin_x_interval
    v_y = v%bbin_y_interval
    print "DEBUG: v_x, v_y:",v_x, v_y, l 
  
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
                bin_pair_intersection = bin_pair_intersection,
                bin_pair_distance = bin_pair_distance)
  print" DEBUG: Done...."

  
  
  
  
  
  