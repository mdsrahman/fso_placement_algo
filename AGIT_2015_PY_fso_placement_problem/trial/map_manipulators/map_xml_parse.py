import xml.etree.ElementTree as ET
from math import radians, cos, sin, asin, sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """ 
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 

    # 6367 km is the radius of the Earth
    m = 1000*6367 * c
    return m 


#tree = ET.parse('./nyc_frag_map.osm')
tree = ET.parse('./map_nyc.osm')
root = tree.getroot()
#print root.tag

lat={}
lon={}
way={}
building_x = {}
building_y = {}
for n in root.iter('node'):
  lat[n.attrib['id']] = n.attrib['lat']
  lon[n.attrib['id']] = n.attrib['lon']

ref_lat = float(min(lat.values()))
ref_lon = float(min(lon.values()))

for a in root.iter('way'):
  way[a.attrib['id']]=[]
  for n in a.iter('nd'):
    if n.attrib['ref'] not in lat.keys():
      del way[a.attrib['id']]
      break
    else:
      way[a.attrib['id']].append(n.attrib['ref'])
  for c in a.iter('tag'):
      if c.attrib['k'] =='building':# and c.attrib['v'] =='yes':
        building_x[a.attrib['id']]=[]
        building_y[a.attrib['id']]=[]
        #print "------building----",a.tag
        for n in a.iter('nd'):
          blat = float(lat[n.attrib['ref']])
          blon = float(lon[n.attrib['ref']])
          x = haversine( blat, blon, blat, ref_lon)
          y = haversine( blat, blon, ref_lat, blon)
          building_x[a.attrib['id']].append(x)
          building_y[a.attrib['id']].append(y)
        break
#print way

for a in root.iter('relation'):
  for c in a.iter('tag'):
      if c.attrib['k'] =='building': #and c.attrib['v'] =='yes':
        building_x[a.attrib['id']]=[]
        building_y[a.attrib['id']]=[]
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
              x = haversine( blat, blon, blat, ref_lon)
              y = haversine( blat, blon, ref_lat, blon)
              building_x[a.attrib['id']].append(x)
              building_y[a.attrib['id']].append(y)
          break
del way
del lat
del lon

patches = [] 
for i in building_x.keys():
  #print "bid:",i,"------------------------------"
  v = np.zeros((len(building_x[i]),2))
  for j in range(len(building_x[i])):
    #print building_x[i][j],",",building_y[i][j]
    v[j][0] = building_x[i][j]
    v[j][1] = building_y[i][j]
  polygon = Polygon(v, False)
  patches.append(polygon) 
  
fig, ax = plt.subplots()
p = PatchCollection(patches)
ax.add_collection(p)
plt.autoscale(enable=True, axis = 'both', tight= True)
plt.show()
      
      
      
      