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
from  shapely import geometry
import geopy.distance
