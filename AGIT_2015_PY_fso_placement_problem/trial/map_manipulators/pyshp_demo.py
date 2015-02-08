import numpy as np
import matplotlib.pyplot as plt
import shapefile 

import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


#sf = shapefile.Reader("./shapefiles/blockgroups")
#sf = shapefile.Reader("./adelaide/adelaide_australia.osm-buildings.shp")
#sf = shapefile.Reader("./nyc/new-york_new-york.osm-buildings.shp")
#sf = shapefile.Reader("./alexandria/alexandria_egypt.osm-polygon.shp")
sf = shapefile.Reader("./adl/adelaide_australia.osm-polygon.shp")

shapes = sf.shapes()
shape_count = 0
patches = [] 
print "processing offestting..."
for s in shapes:
  if s.shapeType != 5:
    continue
  print "shapeType:",s.shapeType
  s.bbox[0] -= sf.bbox[0]
  s.bbox[1] -= sf.bbox[1]
  v = np.zeros((len(s.points),2))
  for i in range(len(s.points)):
    s.points[i][0] -= sf.bbox[0]
    s.points[i][1] -= sf.bbox[1]
    print "points:",s.points[i]
t = raw_input("press enter")
print "generating graphs...."
for s in shapes:
  shape_count += 1
  print "shape#:",shape_count
  if s.shapeType != 5:
    continue
  v = np.zeros((len(s.points),2))
  for i in range(len(s.points)):
    v[i][0] = s.points[i][0] 
    v[i][1] = s.points[i][1] 
  for j in range(len(s.parts)):
    k1 = s.parts[j]
    k2 = len(s.points)
    if j+1<len(s.parts):
      k2 = s.parts[j+1]
    #print "k1:",k1,"  k2:",k2
    polygon = Polygon(v[k1:k2], False)
    patches.append(polygon)
    #print polygon
  
fig, ax = plt.subplots()
p = PatchCollection(patches)
ax.add_collection(p)
plt.autoscale(enable=True, axis = 'both', tight= True)
#plt.draw()
plt.show()
'''
num_vertices = 4
vertices = np.zeros((num_vertices,2))
vertex_points = [(0,.3),(.10,.20),(.7,0),(.4,.5)]
for index, point in enumerate(vertex_points):
  vertices[index][0] = point[0]
  vertices[index][1] = point[1]

print vertices
vertices = np.array(vertices)
patches = []
polygon = Polygon(vertices, True)
patches.append(polygon)

fig, ax = plt.subplots()
p = PatchCollection(patches)
ax.add_collection(p)
plt.show()
'''




