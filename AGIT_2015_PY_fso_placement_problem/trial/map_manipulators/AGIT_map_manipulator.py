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
#from matplotlib.backends.backend_pdf import PdfPages
#from collections import defaultdict

class MapToGraph():
  def __init__(self, mapFileName = None,  max_fso_dist = 2000, min_fso_dist = 100):
    print "Initializing MapToGraph object...."
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
    building_lat = {}
    building_lon = {}

    node_lat = []
    node_lon = []
    
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
              #if these are valid building corners, then are also valid nodes
              node_lat.append(blat)
              node_lon.append(blon)
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
                for i in bnodes:
                  #print lat[i],",",lon[i]
                  blat = float(lat[i])
                  blon = float(lon[i])
                  building_lat[a.attrib['id']].append(blat)
                  building_lon[a.attrib['id']].append(blon)
                  #if these are valid building corners, then are also valid nodes
                  node_lat.append(blat)
                  node_lon.append(blon)

                break #once the member = outer found no need to iterate further
            break #the building tag is found, no need to iterate on this element
    # save memory now------
    
    self.min_lat = float( min( node_lat ) )
    self.max_lat = float( max( node_lat ) ) 
    self.min_lon = float( min( node_lon ) )
    self.max_lon = float( max( node_lon) )
    
    del way
    del lat
    del lon
    
    #now build the cartesian coordinate for buildings
    for bid in building_lat.keys():
      blats =  building_lat[bid]
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
      self.building.append(shgm.Polygon(zip(xs,ys)))
    #also build the cartesian products for the nodes:
    del building_lat
    del building_lon
    for i,nlat in enumerate(node_lat):
      nlon = node_lon[i]
      x,y = self.get_relative_coord(lat = nlat, 
                                    lon = nlon, 
                                    in_ref_lat = self.min_lat, 
                                    in_ref_lon = self.min_lon) 
      self.node.append(shgm.Point(x,y))
      self.max_x = max(self.max_x, x)
      self.max_y = max(self.max_y, y)
    del node_lat
    del node_lon
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
    #flog = open('./map/build_adj_logger.txt','a')
    #flog.write("----------Total Nodes:"+str(len(self.node_x))+'\n')
    print "DEBUG: Number of nodes:", len(self.node)
    temp = raw_input("Press Enter To Continue:")
    self.adj = nx.Graph()
    self.adj.graph['name'] = 'Adjacency Graph'
    total_nodes = len(self.node)
    ##check the line intersection with each of the box: for each box, check intersection with each of the bld in it
    for u in range(total_nodes-1):
      for v in range(u+1,total_nodes):
        edge_type = self.check_edge_type(u,v)
        if edge_type =='short' or edge_type =='long':
          self.adj.add_edge(u, v, con_type = edge_type)
    return
  
  def debug_visualize_buildings(self, in_adj = None):
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
    
    #---also draw edges on the graph----
    for u in self.node:
      plt.plot([u.x],[u.y],"ro")
      
    adj = in_adj
    if not adj:
      adj = self.adj
    if adj:
      for u,v in adj.edges():
        if self.adj[u][v]['con_type'] == 'short':
          edge_color = 'g'
        else:
          edge_color ='b'
        plt.plot([self.node[u].x, self.node[v].x],\
                 [self.node[u].y, self.node[v].y], color = edge_color,  ls ='dotted')
    
    
    ax.autoscale(enable=True, axis = 'both', tight= True)
    ax.set_aspect('equal', 'box')
    fig.set_size_inches(10, 10)
    fig.savefig(self.mapFileName+".pdf", bbox_inches = 'tight')
    plt.show()

    return 
#-----------------unit testing-------------------#
if __name__ == '__main__':
  mtg = MapToGraph('./map/dallas_v2.osm')  
  mtg.load_map()
  mtg.hash_builidings(max_buildings_per_box = 20)
  mtg.build_adj_graph()
  mtg.debug_visualize_buildings()


