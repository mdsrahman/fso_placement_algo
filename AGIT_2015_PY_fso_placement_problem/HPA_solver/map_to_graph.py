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
import random
import cPickle as pkl
import networkx as nx
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
#from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict

class MapToGraph():
  def __init__(self, mapFileName = None,  max_fso_dist = 2000, min_fso_dist = 100):
    #print "Initializing MapToGraph object...."
    self.mapFileName = mapFileName
    #----the following are the class variables defined in different methods-----
    self.building = [] # list of latitude (x) of  building corner-points keyed by building id
    self.node = []
    #self.building_y={} # list of longitude (y) of building corner-points keyed by building id
    #-- the following 4 defines the bounding box of the processed map in terms of lat, lon

    #self.node_x=[] #all valid nodes x coord
    #self.node_y=[] #all valid nodes y coord
    self.max_fso_dist = max_fso_dist
    self.min_fso_dist = min_fso_dist
    self.min_lat = None
    self.max_lat = None
    self.min_lon = None
    self.max_lon = None
    self.max_x = None
    self.max_y = None
    self.box=None #hashes the buildings according to their coordinates
    #self.box_coord = None #stores the coordinate of the boxes
    self.nbox_dim = None #dimension of 2D box array
    self.building_in_box = None
    self.adj = None
    self.tnode = None
    self.sinks = None
    self.last_time = time.time()
    return
  
  def setMapFileName(self,mapFileName):
    '''sets the mapFileName as class data'''
    self.mapFileName = mapFileName
  
  def find_closest_point_indx(self, ref_x, ref_y, xs, ys):
    min_dist_indx = -1
    min_dist = float('inf')
    for indx,x in enumerate(xs):
      y=ys[indx]
      line = shgm.LineString( [(ref_x,ref_y),(x,y)])
      if min_dist > line.length:
        min_dist_indx = indx
        min_dist = line.length 
    return min_dist_indx
  
  def load_map(self, mapFileName = None, bounding_box_nodes_only = False):
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
    print 'Loading Map File: ',self.mapFileName," ....."
    tree = eTree.parse(self.mapFileName)
    root = tree.getroot()
    
    lat={} #temporarily keeps all the latitude of points (keyed by id) for future references 
    lon={} #temporarily keeps all the longitude of points (keyed by id)  for future references 
    way = {} #temporarily keeps all the id's of the ways
    building_lat = {}
    building_lon = {}
    self.min_lat = self.min_lon = float('inf')
    self.max_x = self.max_y = self.max_lat = self.max_lon = - float('inf')
     
    #node_lat = []
    #node_lon = []
    
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
            building_lat[a.attrib['id']]=[]
            building_lon[a.attrib['id']]=[]
            #print "------building----",a.tag
            for n in a.iter('nd'):
              blat = float(lat[n.attrib['ref']])
              blon = float(lon[n.attrib['ref']])
              building_lat[a.attrib['id']].append(blat)
              building_lon[a.attrib['id']].append(blon)
              self.min_lat = min(self.min_lat, blat)
              self.min_lon = min(self.min_lon, blon)
              self.max_lat = max(self.max_lat, blat)
              self.max_lon = max(self.max_lon, blon)
              #if these are valid building corners, then are also valid nodes
              #if not bounding_box_nodes_only:
              #node_lat.append(blat)
              #node_lon.append(blon)
              #else: #take only 4 nodes from the bounding box
            break # careful about this break indentation, it matches within the if{ seqment }
    #print way
    
    for a in root.iter('relation'):
      for c in a.iter('tag'):
          if c.attrib['k'] =='building': #and c.attrib['v'] =='yes':
            building_lat[a.attrib['id']]=[]
            building_lon[a.attrib['id']]=[]
            #print "------building----",a.tag
            for n in a.iter('member'):
              if n.attrib['role']=='outer':
                if n.attrib['ref'] not in way.keys():
                  continue
                bnodes = way[n.attrib['ref']]
                #print "------building----",a.tag
                
                if len(bnodes)>300: #if too much curvature, skip it
                  del building_lat[a.attrib['id']]
                  del building_lon[a.attrib['id']]
                  continue
                for i in bnodes:
                  #print lat[i],",",lon[i]
                  blat = float(lat[i])
                  blon = float(lon[i])
                  building_lat[a.attrib['id']].append(blat)
                  building_lon[a.attrib['id']].append(blon)
                  self.min_lat = min(self.min_lat, blat)
                  self.min_lon = min(self.min_lon, blon)
                  self.max_lat = max(self.max_lat, blat)
                  self.max_lon = max(self.max_lon, blon)
                  #if these are valid building corners, then are also valid nodes
                  #if not bounding_box_nodes_only:
                  #node_lat.append(blat)
                  #node_lon.append(blon)

                break #once the member = outer found no need to iterate further
            break #the building tag is found, no need to iterate on this element
    # save memory now------
    
    #self.min_lat = float( min( node_lat ) )
    #self.max_lat = float( max( node_lat ) ) 
    #self.min_lon = float( min( node_lon ) )
    #self.max_lon = float( max( node_lon) )
    
    del way
    del lat
    del lon
    
    #now build the cartesian coordinate for buildings
    for bid in building_lat.keys():
      blats =  building_lat[bid]
      #if len(blats)>200:# too many points for a single building !!!!!!!
      #  continue
      blons = building_lon[bid]
      xs = []
      ys = []
      for i,blat in enumerate(blats):
        blon = blons[i]
        x,y = self.get_relative_coord(lat = blat, 
                                      lon = blon, 
                                      in_ref_lat = self.min_lat, 
                                      in_ref_lon = self.min_lon)
        xs.append(x)
        ys.append(y)
      #print "DEBUG:@map_to_graph::load_map(..):len xs, len ys:",len(xs),len(ys)
      if xs[0]== xs[-1] and ys[0]==ys[-1]:
        del xs[-1]
        del ys[-1]
        #print "DEBUG:true"
      if len(xs)>2 and len(ys)>2:
        bldg = shgm.Polygon(zip(xs,ys))
        self.building.append(bldg)
        #bmin_x, bmin_y, bmax_x, bmax_y = bldg.bounds
        #print "DEBUG:",bmin_x, bmin_y, bmax_x, bmax_y
        self.max_x = max(xs+[self.max_x])
        self.max_y = max(ys+[self.max_y])
        if not bounding_box_nodes_only or len(xs)<=4:
          for x,y in zip(xs,ys):
            self.node.append(shgm.Point(x,y))
          
        else:
          #generate the bounding box,
          bmin_x, bmin_y, bmax_x, bmax_y = bldg.bounds
          ref_xs = [bmin_x, bmax_x, bmax_x, bmin_x]
          ref_ys = [bmin_y, bmin_y, bmax_y, bmax_y]
          for ref_x, ref_y in zip(ref_xs, ref_ys):
            if len(xs)<=0:
              break
            indx = self.find_closest_point_indx(ref_x = ref_x,
                                                  ref_y = ref_y,
                                                  xs = xs,
                                                  ys = ys)
            x = xs.pop(indx)
            y = ys.pop(indx)
            self.node.append(shgm.Point(x,y))
          #select exactly four points closest to the bounding box
          #add only these four points to the node list as point object
        #now if 
    #also build the cartesian products for the nodes:
    del building_lat
    del building_lon
    
    #for i,nlat in enumerate(node_lat):
    #  nlon = node_lon[i]
    #  x,y = self.get_relative_coord(lat = nlat, 
    #                                lon = nlon, 
    #                                in_ref_lat = self.min_lat, 
    #                                in_ref_lon = self.min_lon) 
    #  self.node.append(shgm.Point(x,y))
    #  self.max_x = max(self.max_x, x)
    #  self.max_y = max(self.max_y, y)
    print "\t>>Total Nodes to be processed: ",len(self.node)
    #print "DEBUG:-------------"
    #for bld in self.building:
    #  print "DEBUG: building_coord:",list(bld.exterior.coords)
    #del node_lat
    #del node_lon
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
  
  
  def hash_builidings(self, max_buildings_per_box = 16):
    total_builldings = len(self.building)
    nbox_dim =  int(np.ceil( np.sqrt(total_builldings)/ np.sqrt(max_buildings_per_box ) ))
    self.nbox_dim = nbox_dim
    nbox = nbox_dim * nbox_dim

    x_range = self.max_x/nbox_dim
    y_range = self.max_y/nbox_dim
  
    self.box = {}
    self.building_in_box={}
    self.box_coord = {}
    
    for i in range(nbox_dim):
      box_x_min = i * x_range
      box_x_max = (i+1) * x_range
      self.box[i]={}
      self.building_in_box[i]={}
      for j in range(nbox_dim):
        self.building_in_box[i][j] = []
        box_y_min =  j * y_range
        box_y_max = (j+1)* y_range
        self.box[i][j] = shgm.box(minx = box_x_min, 
                                  miny = box_y_min, 
                                  maxx = box_x_max, 
                                  maxy = box_y_max)

    
    for bindx, bld in enumerate(self.building):
      for i in range(nbox_dim):
        for j in range(nbox_dim):
          if bld.intersects( self.box[i][j]):
            self.building_in_box[i][j].append(bindx)
    print "\t\t\tDEBUG:Total boxes:",nbox_dim*nbox_dim
    '''      
    for i in range(nbox_dim):
      for j in range(nbox_dim):
        print "DEBUG:i,j:",i,j,":",list(self.box[i][j].exterior.coords)," total bldgs:",len(self.building_in_box[i][j])
    '''
    return
  def check_edge_type(self,u,v):
    #make a line of (u,v)
    line = shgm.LineString( list(self.node[u].coords) + list(self.node[v].coords) )
    distance =  line.length
    if distance > self.max_fso_dist:
      return 'nonedge'
    poly_cache = []
    for i in range(self.nbox_dim):
      for j in range(self.nbox_dim):
        box = self.box[i][j]
        if line.intersects(box):
          for bid in self.building_in_box[i][j]:
            if bid not in poly_cache:
              poly_cache.append(bid)
              polygon = self.building[bid]
              if line.intersects(polygon) and not line.touches(polygon):
                return 'nonedge'
    if distance<=self.min_fso_dist:
      return 'short'
    else:
      return 'long'      

  def build_adj_graph(self):
    self.adj = nx.Graph()
    self.adj.graph['name'] = 'Adjacency Graph'
    total_nodes = len(self.node)
    ##check the line intersection with each of the box: for each box, check intersection with each of the bld in it
    for u in range(total_nodes-1):
      if total_nodes>100: #give status:
        #if u%100 == 0:
        print "\t\t Building edges for node# ",u
      for v in range(u+1,total_nodes):
        edge_type = self.check_edge_type(u,v)
        if edge_type =='short' or edge_type =='long':
          self.adj.add_edge(u, v, con_type = edge_type)
    return
  
  def generate_targets(self, target_granularity = 10.0):
    self.tnode=[]
    tx = 0.0
    ty = 0.0
    while(tx <= self.max_x):
      while(ty <= self.max_y):
        self.tnode.append(shgm.Point(tx,ty))
        ty += target_granularity
      tx += target_granularity
      ty = 0.0
        
    #total_targets = int( self.adj.number_of_nodes() * 0.4 )
    #for t in range(total_targets):
    #  tx = round(random.uniform(0, self.max_x),2)
    #  ty = round(random.uniform(0, self.max_y),2)
    #  self.tnode.append(shgm.Point(tx,ty))
    #print "DEBUG: Total Targets: ",len(self.tnode)
    
    return

  def associate_targets(self):
    self.T=defaultdict(set)
    self.T_N=defaultdict(list)
    for tindx,t in enumerate(self.tnode):
      for nindx, n in enumerate(self.node):
        line = shgm.LineString( list(n.coords) + list(t.coords) )
        if line.length <= self.min_fso_dist:
          #print "DEBUG: t<-->n",tindx, nindx
          self.T[nindx].add(tindx) #node n covers target t
          self.T_N[tindx].append(nindx)
    return
  def select_sinks(self, sink_ratio = 0.01):
    self.sinks = []
    num_sinks = int(np.ceil((sink_ratio*np.sqrt(len(self.tnode))))) #<---now parred with targets
    
    self.sinks = random.sample(self.adj.nodes(), num_sinks)
    print "DEBUG:no_of_sink_nodes: ",num_sinks
    #print "DEBUG:sinks:",self.sinks
    return
  def generate_graph(self, fileName, target_granularity, sink_to_node_ratio, bounding_box_nodes_only = False):
    self.last_time = time.time()
    self.load_map(fileName, bounding_box_nodes_only = bounding_box_nodes_only)
    print"\t\t Time taken:",time.time() - self.last_time,"sec"
    self.last_time = time.time()
    
    
    print "\t\tassociating targets to nodes.."
    self.generate_targets(target_granularity= target_granularity)
    self.associate_targets()
    print"\t\t Time taken:",time.time() - self.last_time,"sec"
    self.last_time = time.time()
    
    
    self.generate_visual_map()#<----just for debugging
    
    
    print "\t\thashing building obstacle positions..."
    self.hash_builidings()
    print"\t\t Time taken:",time.time() - self.last_time,"sec"
    self.last_time = time.time()
    
    print "\t\tbulding adjacency graph..."
    self.build_adj_graph()
    print"\t\t Time taken:",time.time() - self.last_time,"sec"
    self.last_time = time.time()
    
    self.select_sinks(sink_ratio = sink_to_node_ratio)
    self.generate_visual_map()#<----just for debugging
    #self.debug_visualize_buildings()
    return self.adj, self.sinks, self.T, self.T_N, self.node, self.tnode
  
 
  
  def generate_visual_map(self, adj = None):
    patches = [] 
    for bldg in self.building:
      #print "bid:",i,"------------------------------" 
      pcoord = np.asarray(bldg.exterior.coords, dtype = float)
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
    for uindx, u in enumerate(self.node):
      if self.sinks and uindx in self.sinks:
        plt.plot([u.x],[u.y],"gs") 
      else:
        plt.plot([u.x],[u.y],"bo")      
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
        if str(self.adj[u][v]['con_type']) == 'short':
          edge_color = 'g'
        else:
          edge_color ='b'
        
        plt.plot([self.node[u].x, self.node[v].x],\
                 [self.node[u].y, self.node[v].y], color = edge_color,  ls ='dotted')
    
    
    ax.autoscale(enable=True, axis = 'both', tight= True)
    ax.set_aspect('equal', 'box')
    fig.set_size_inches(10, 10)
    fig.savefig(self.mapFileName+".pdf", bbox_inches = 'tight')
    #plt.show()

    return 
#-----------------unit testing-------------------#
if __name__ == '__main__':
   
  mapfiles = ['./map/dallas_v1.osm',\
           './map/dallas_v2.osm',\
           './map/dallas_v3.osm',\
           './map/nyc_hells_kitchen.osm'
           ]
  target_to_node_ratio = 0.4
  sink_to_node_ratio = 0.05
  for mfile in mapfiles:
    mtg = MapToGraph() 
    adj, sinks, T, T_N, node, tnode \
    =\
    mtg.generate_graph(fileName= mfile,  
                     target_to_node_ratio = target_to_node_ratio, 
                     sink_to_node_ratio = sink_to_node_ratio) 
    
    mtg.generate_visual_map()


