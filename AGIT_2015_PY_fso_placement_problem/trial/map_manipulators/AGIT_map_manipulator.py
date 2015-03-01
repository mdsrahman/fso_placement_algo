'''
up to now: building_x: building_y max_x, max_y, min_x, min_y
1. generate valid nodes from all building corner points:
node[0]=(coord_x, coord_y)
node[1]= ...

2. big step: adj generation:
  i) calculate number of boxes box_no: 
      based on the number of buildings each box contains roughly max 10 buildings
  ii) divide the (max_x,min_x) and (max_y, min_y) [in cart-coord] 
    ranges in ceil(sqrt(box_no)), keep the bounding box data in box_boundary dict in cart-coord
  iii) associate each building with any of the box_no boxes that it intersects [use cart-coord]
     with that boxes and thus build the box[(i,j)] dict, 
     use the id of buildings as obtained from xml/osm file
  iv) for each pairs of nodes (i,j):
    a) if distance(i,j)[cart-coord] <=2000m
      for each boxes b:
        if b intersect line(i,j) [cart-coord]
           for each building b in box[b]:
             if b [cart-coord] intersects (i,j), mark (i,j) as non-edge quit upto (a)
      if (i,j) is not non-edge:
        if distance(i,j)[cart-coord] <= 100m:
          add to networkx.adj (i,j) edge and mark it as 'short' edge
        else
            add to networkx.adj (i,j) edge and mark it as 'long' edge
            
3. generate target-node (T) and node-target (T_N) associativity based on 100m coverage
4. assing sink nodes, preferably from different boxes as set sinks
return adj, sinks, T, T_N
tools: 
shapely: object.intersects(other)
geopy: dist = geopy.distance.distance(pt1, pt2).m
'''
import xml.etree.ElementTree as eTree
from  shapely import geometry as shgm
import geopy.distance
import numpy as np

import networkx as nx

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from collections import defaultdict

class MapToGraph():
  def __init__(self, mapFileName=None):
    print "Initializing MapToGraph object...."
    self.mapFileName = mapFileName
    #----the following are the class variables defined in different methods-----
    self.building_x={} # list of latitude (x) of  building corner-points keyed by building id
    self.building_y={} # list of longitude (y) of building corner-points keyed by building id
    #-- the following 4 defines the bounding box of the processed map in terms of lat, lon
    self.node_counter = 0
    self.node_x=[] #all valid nodes latitude (x)
    self.node_y=[] #all valid nodes longitude (y)
    
    self.min_lat = None
    self.max_lat = None
    self.min_lon = None
    self.max_lon = None
    
    self.box=None #hashes the buildings according to their coordinates
    self.box_coord = None #stores the coordinate of the boxes
    self.nbox_dim = None #dimension of 2D box array
    
    self.adj = None
    return
  
  def setMapFileName(self,mapFileName):
    '''sets the mapFileName as class data'''
    self.mapFileName = mapFileName
    
  def load_map(self, mapFileName = None):
    '''
    load the specified xml/osm file and generate the building_x, building_y for latitude and
    longitude of building corners
    '''
    if mapFileName:
      self.mapFileName = mapFileName
      
    if not self.mapFileName:
      print "Map File Name not found!"
      return
    #-----the following copied from map_xml_parse-----------#
    #----just parse the xml and find the building objects 
    #-----which are either ways or references with attrib building
    tree = eTree.parse(self.mapFileName)
    root = tree.getroot()
    
    lat={} #temporarily keeps all the latitude of points (keyed by id) for future references 
    lon={} #temporarily keeps all the longitude of points (keyed by id)  for future references 
    way = {} #temporarily keeps all the id's of the ways
    
    for n in root.iter('node'): # loads all the node lat,lon
      lat[n.attrib['id']] = float(n.attrib['lat'])
      lon[n.attrib['id']] = float(n.attrib['lon'])
    
    
    
    for a in root.iter('way'):
      way[a.attrib['id']]=[]
      for n in a.iter('nd'):
        if n.attrib['ref'] not in lat.keys(): 
          # ---no need to store the ways whose ref nodes not found
          del way[a.attrib['id']]
          break
        else:
          way[a.attrib['id']].append(n.attrib['ref'])
          
      for c in a.iter('tag'):
          if c.attrib['k'] =='building':# and c.attrib['v'] =='yes':
            self.building_x[a.attrib['id']]=[]
            self.building_y[a.attrib['id']]=[]
            #print "------building----",a.tag
            for n in a.iter('nd'):
              blat = float(lat[n.attrib['ref']])
              blon = float(lon[n.attrib['ref']])
              self.building_x[a.attrib['id']].append(blat)
              self.building_y[a.attrib['id']].append(blon)
              #if these are valid building corners, then are also valid nodes
              self.node_x.append(blat)
              self.node_y.append(blon)

            break # careful about this break indentation, it matches within the if{ seqment }
    #print way
    
    for a in root.iter('relation'):
      for c in a.iter('tag'):
          if c.attrib['k'] =='building': #and c.attrib['v'] =='yes':
            self.building_x[a.attrib['id']]=[]
            self.building_y[a.attrib['id']]=[]
            #print "------building----",a.tag
            for n in a.iter('member'):
              if n.attrib['role']=='outer':
                if n.attrib['ref'] not in way.keys():
                  continue
                bnodes = way[n.attrib['ref']]
                #print "------building----",a.tag
                for i in bnodes:
                  #print lat[i],",",lon[i]
                  blat = float(lat[i])
                  blon = float(lon[i])
                  self.building_x[a.attrib['id']].append(blat)
                  self.building_y[a.attrib['id']].append(blon)
                  #if these are valid building corners, then are also valid nodes
                  self.node_x.append(blat)
                  self.node_y.append(blon)

                break #once the member = outer found no need to iterate further
            break #the building tag is found, no need to iterate on this element
    # save memory now------
    
    self.min_lat = float( min( self.node_x ) )
    self.max_lat = float( max( self.node_x ) ) 
    self.min_lon = float( min( self.node_y ) )
    self.max_lon = float( max( self.node_y ) )
    
    del way
    del lat
    del lon
    return
  
  def get_relative_coord(self,lat,lon, in_ref_lat = None, in_ref_lon = None):
    ref_lat = in_ref_lat
    if not ref_lat:
      ref_lat = self.min_lat
      
    ref_lon = in_ref_lon
    if not ref_lon:
      ref_lon = self.min_lon
    x_ref = geopy.Point(ref_lat, lon)
    y_ref = geopy.Point(lat,          ref_lon)
    xy = geopy.Point(lat,lon)
    x = geopy.distance.distance(xy, x_ref).m
    y = geopy.distance.distance(xy, y_ref).m
    return x,y
  
  def get_relative_coord_haversine(self,lat,lon):
    x = self.get_haversine_distance( lat1 =  lat,
                                     lon1 = lon, 
                                     lat2 = self.min_lat, 
                                     lon2 = lon)
    y = self.get_haversine_distance( lat1 = lat,
                                     lon1 = lon, 
                                     lat2 = lat, 
                                     lon2 = self.min_lon)
    return x,y
  
  def get_haversine_distance(self, lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """ 
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 

    # 6367 km is the radius of the Earth
    m = 1000*6367 * c
    return m 
  
  def is_intersecting(self, obj1_xs, obj1_ys, obj2_xs, obj2_ys):
    #print "DEBUG@is_intersecting:",obj1_xs, obj1_ys, obj2_xs, obj2_ys
    #make shapely polygons for obj1 and obj2
    obj1_poly_coord = []
    for i,x in enumerate(obj1_xs):
      y =  obj1_ys[i]
      cart_x, cart_y =  self.get_relative_coord(x,y)
      obj1_poly_coord.append((cart_x, cart_y))
    
    obj2_poly_coord = []
    for i,x in enumerate(obj2_xs):
      y =  obj2_ys[i]
      cart_x, cart_y =  self.get_relative_coord(x,y)
      obj2_poly_coord.append((cart_x, cart_y))
    
    poly1 = shgm.Polygon(obj1_poly_coord)
    poly2 = shgm.Polygon(obj1_poly_coord)
    
    return poly1.intersects(poly2)
  
  def is_member_of_box_cart_coord(self, i , j, lat_list, lon_list):
    box_x_min = self.box_coord[i][j][0]
    box_x_max = self.box_coord[i][j][1]
    box_y_min = self.box_coord[i][j][2]
    box_y_max = self.box_coord[i][j][3]
    
    box_xs = [box_x_min, box_x_max, box_x_max, box_x_min ]
    box_ys = [box_y_min, box_y_min, box_y_max, box_y_max ]
    return self.is_intersecting(obj1_xs = lat_list , 
                                obj1_ys = lon_list , 
                                obj2_xs = box_xs, 
                                obj2_ys = box_ys )
    
  def is_member_of_box(self, i , j, lat_list, lon_list):
    box_lat_min = self.box_coord[i][j][0]
    box_lat_max = self.box_coord[i][j][1]
    box_lon_min = self.box_coord[i][j][2]
    box_lon_max = self.box_coord[i][j][3]
    #print "DEBUG@is_member_of_box:lat_list:",lat_list
    for index, lat in enumerate(lat_list):
      lon = lon_list[index]
      '''
      print "DEBUG:i,j:",i,j
      print "DEBUG:blat_min, lat, blat_max:",box_lat_min, lat, box_lat_max
      print "DEBUG:blon_min, lon, blon_max:",box_lon_min, lon, box_lon_max
      temp = raw_input("enter:")
      '''
      if (lat>=box_lat_min and lat<=box_lat_max):
        if (lon>=box_lon_min and lon<=box_lon_max):
          return True
    return False
  def hash_builidings(self, max_buildings_per_box = 16):
    total_builldings = len(self.building_x)
    nbox_dim =  int(np.ceil( np.sqrt(total_builldings)/ np.sqrt(max_buildings_per_box ) ))
    self.nbox_dim = nbox_dim
    nbox = nbox_dim * nbox_dim
    
    lat_range = (self.max_lat - self.min_lat)/nbox_dim
    lon_range = (self.max_lon - self.min_lon)/nbox_dim

    print "DEBUG:max_lat:",self.max_lat
    print "DEBUG:min_lat:",self.min_lat
    print "DEBUG:max_lon:",self.max_lon
    print "DEBUG:min_lon:",self.min_lon
    print "DEBUG:nbox_dim:",nbox_dim
    print "DEBUG:lat_range:",lat_range
    print "DEBUG:lon_range:",lon_range
  
    self.box = defaultdict(dict)
    self.box_coord = defaultdict(dict)
    
    for i in range(nbox_dim):
      box_lat_min = self.min_lat + i*lat_range
      box_lat_max = self.min_lat + (i+1) *lat_range
      for j in range(nbox_dim):
        box_lon_min = self.min_lon + j*lon_range
        box_lon_max = self.min_lon + (j+1)*lon_range
        self.box[i][j] = []
        self.box_coord[i][j] = [box_lat_min, box_lat_max, box_lon_min, box_lon_max]
        print "DEBUG: i,j, box_boundaries",i,j,self.box_coord[i][j]
        if box_lat_min < self.min_lat or \
           box_lat_max > self.max_lat or \
           box_lon_min < self.min_lon or \
           box_lon_max > self.max_lon  :
          print "DEBUG ALERT!!!!!"
          print "DEBUG: i,j, box_boundaries",i,j,self.box_coord[i][j]
    
    for bid,building_x_list in self.building_x.iteritems():
      #building_x_list = self.building_x[bid]
      building_y_list = self.building_y[bid]
      for i in range(nbox_dim):
        for j in range(nbox_dim):
          if self.is_member_of_box(i = i,
                                  j = j , 
                                  lat_list = building_x_list, 
                                  lon_list = building_y_list):
            self.box[i][j].append(bid)
            #print "DEBUG:i,j:",i,j,":",self.box[i][j]
    #print "DEBUG: number of buildings:",total_builldings
    #print "DEBUG: number of boxes required:",nbox
    for i in range(nbox_dim):
      for j in range(nbox_dim):
        print "DEBUG:i,j:",i,j,":",len(self.box[i][j])
    return
  
  def get_box_index(self,lat,lon):
    for i in range(self.nbox_dim):
      for j in range(self.nbox_dim):
        if self.is_member_of_box(i = i, j = j, lat_list = [lat], lon_list = [lon]):
          return i,j
    return -1,-1
  
  def get_building_set(self, in_i1, in_j1, in_i2 = None, in_j2 = None):
    if not in_i2 or not in_j2:
      return list(self.box[in_i1][in_i2])
    i1 = min(in_i1, in_i2)
    i2 = max(in_i1, in_i2)
    j1 = min(in_j1, in_j2)
    j2 = max(in_j1, in_j2)
    building_set = []
    for i in range(i1, i2+1):
      for j in range(j1, j2+1):
        building_set.extend(self.box[i][j])
    return building_set
  
  def has_intersected_with_buildings(self, building_ids, lat1, lon1, lat2,  lon2):
    for bid in building_ids:
      # build polygon and line objects from params
      x1,y1 =  self.get_relative_coord(lat1, lon1)
      x2,y2 =  self.get_relative_coord(lat2, lon2)
      line  = shgm.LineString([(x1,y1), (x2,y2)])
      polygon_coord = []
      building_lat = self.building_x[bid]
      building_lon = self.building_y[bid]
      for i,b_lat in enumerate(building_lat):
        b_lon = building_lon[i]
        polygon_coord.append( ( self.get_relative_coord(b_lat, b_lon) ) )
      polygon = shgm.Polygon(polygon_coord)   
      if polygon.intersects(line)          :
        return True         
    return False
  
  def build_adj_graph(self, max_fso_dist = 2000, min_fso_dist = 100):
    flog = open('./map/build_adj_logger.txt','a')
    flog.write("----------Total Nodes:"+str(len(self.node_x))+'\n')
    print "DEBUG: Number of nodes:", len(self.node_x)
    temp = raw_input("Press Enter To Continue:")
    self.adj = nx.Graph()
    self.adj.graph['name'] = 'Adjacency Graph'
    total_nodes = len(self.node_x)
    for u in range(total_nodes-1):
      lat_u = self.node_x[u]
      lon_u = self.node_y[u]
      for v in range(u+1,total_nodes):
        print "DEBUG: Processing u,v: ",u,v
        flog.write("Processing u,v:"+str(u)+','+str(v)+'\n')
        lat_v = self.node_x[v]
        lon_v = self.node_y[v]
        distance = self.get_relative_coord(lat = lat_u, 
                                             lon = lon_u, 
                                             in_ref_lat = lat_v, 
                                             in_ref_lon = lon_v)
        print "DEBUG:distance:",distance
        flog.write("\t distance="+str(distance))
        if distance > max_fso_dist:
          continue
        ubox_i, ubox_j = self.get_box_index(lat_u, lon_u)
        vbox_i, vbox_j = self.get_box_index(lat_v, lon_v)
        building_ids = self.get_building_set(ubox_i, ubox_j, vbox_i, vbox_j)
        print "DEBUG: Processing u,v, #buildings: ",u,v,len(building_ids)
        flog.write("\t number of buildings:"+str(len(building_ids)))
        if self.has_intersected_with_buildings(building_ids = building_ids, 
                                                 lat1 = lat_u, 
                                                 lon1 = lon_u, 
                                                 lat2 = lat_v, 
                                                 lon2 = lon_v):
          flog.write("\t intersected with buildings:")
          continue
        else:
          if distance <= min_fso_dist:
            self.adj.add_edge(u, v, con_type ='short')
          else:
            self.adj.add_edge(u, v, con_type ='long')
    flog.close()
    return
  
  def debug_visualize_buildings(self):
    patches = [] 
    for i in self.building_x.keys():
      #print "bid:",i,"------------------------------"
      v = np.zeros((len(self.building_x[i]),2))
      for j in range(len(self.building_x[i])):
        #print building_x[i][j],",",building_y[i][j]
        x,y = self.get_relative_coord(self.building_x[i][j], self.building_y[i][j])
        #x,y = self.get_relative_coord_haversine(self.building_x[i][j], self.building_y[i][j])
        v[j][0] = x
        v[j][1] = y
      polygon = Polygon(v, fc='grey')
      #polygon.set_facecolor('none')
      patches.append(polygon) 
    
    fig, ax = plt.subplots()
    p = PatchCollection(patches, match_original=True)
    ax.add_collection(p)
    #plt.grid(True)
    '''
    box_dim = len(self.box[0])
    for i in range(box_dim):
      for j in range(box_dim):
        box_lat_min = self.box_coord[i][j][0]
        box_lat_max = self.box_coord[i][j][1]
        box_lon_min = self.box_coord[i][j][2]
        box_lon_max = self.box_coord[i][j][3]
        x1,y1 =  self.get_relative_coord(box_lat_min, box_lon_min)
        x2,y2 =  self.get_relative_coord(box_lat_max, box_lon_min)
        x3,y3 =  self.get_relative_coord(box_lat_max, box_lon_max)
        x4,y4 =  self.get_relative_coord(box_lat_min, box_lon_max)
        plt.plot([x1, x2, x3, x4, x1],[y1, y2, y3, y4, y1],'r')
    '''
    #---also draw edges on the graph----
    for u,v in self.adj.nodes():
      ulat = self.node_x[u]
      ulon = self.node_y[u]
      vlat = self.node_x[v]
      vlon = self.node_y[v]
      x1, y1 = self.get_relative_coord(ulat, ulon)
      x2, y2 = self.get_relative_coord(vlat, vlon)
      if self.adj[u][v]['con_type'] == 'short':
        edge_color = 'g'
      else:
        edge_color ='b'
      plt.plot([x1,x2],[y1,y2],edge_color)
      
    plt.autoscale(enable=True, axis = 'both', tight= True)
    
    plt.show()
    return 
#-----------------unit testing-------------------#
if __name__ == '__main__':
  mtg = MapToGraph('./map/sf_v2.osm')
  mtg.load_map()
  mtg.hash_builidings(max_buildings_per_box = 20)
  mtg.build_adj_graph()
  mtg.debug_visualize_buildings()

