from  shapely import geometry
import geopy
import geopy.distance

pt1 = geopy.Point(48.853, 2.349)
pt2 = geopy.Point(52.516, -13.378)

dist = geopy.distance.distance(pt1, pt2).m
print dist

polygon = [(4.0, -2.0), (5.0, -2.0), (4.0, -3.0), (3.0, -3.0), (4.0, -2.0)]
shapely_poly = geometry.Polygon(polygon)

line = [(4.0, -2.0000000000000004), (2.0, -1.1102230246251565e-15)]
shapely_line = geometry.LineString(line)

intersection_line = list(shapely_poly.intersection(shapely_line).coords)
print intersection_line 