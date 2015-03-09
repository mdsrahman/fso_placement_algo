
import heapq
from collections import defaultdict
import random
import numpy as np
#import itertools

class MyHeap: 
  pq = []                         # list of entries arranged in a heap
  entry_finder = {}               # mapping of tasks to entries
  REMOVED = '<removed-task>'      # placeholder for a removed task
  #counter = itertools.count()     # unique sequence count
  
  def add_item(self, task, priority=0):
      'Add a new task or update the priority of an existing task'
      if task in self.entry_finder:
        self.remove_item(task)
      #count = next(self.counter)
      entry = [priority, task]
      self.entry_finder[task] = entry
      heapq.heappush(self.pq, entry)
  
  def remove_item(self,task):
      'Mark an existing task as REMOVED.  Raise KeyError if not found.'
      print "DEBUG:entry_finder",self.entry_finder
      entry = self.entry_finder.pop(task)
      entry[1] = self.REMOVED
  
  def pop_item(self):
      'Remove and return the lowest priority task. Raise KeyError if empty.'
      while self.pq:
          priority, task = heapq.heappop(self.pq)
          if task is not self.REMOVED:
            #del self.entry_finder[task]
            return task
      return None
      #raise KeyError('pop from an empty priority queue')


class Target_Node_Assoc:
  def __init__(self, number_of_nodes, max_x, max_y, target_interval, node_coverage, node_bin_interval):
    self.number_of_nodes = number_of_nodes
    self.max_x = max_x
    self.max_y = max_y
    self.target_interval = target_interval
    self.node_coverage = node_coverage
    self.node_x = []
    self.node_y = []
    self.node_target_count = np.zeros(self.number_of_nodes, dtype = int)
    self.node_cover = []
    self.node_bin_interval = node_bin_interval
    self.node_bin = {}
    #----random node coord for testing purpose----
    for n in xrange(self.number_of_nodes):
      x = random.uniform(0, max_x)
      y = random.uniform(0, max_y)
      self.node_x.append(x)
      self.node_y.append(y)
    #---------------------------------------------
    self.bin_nodes()
    
    max_target_x = int(self.max_x // self.target_interval)
    max_target_y = int(self.max_y // self.target_interval)
    self.total_targets = (max_target_x+1) * (max_target_y +1)
    print  "max_target_x, max_target_y:", max_target_x, max_target_y
    self.targets_covered = 0
    self.target = np.zeros((max_target_x+1, max_target_y+1), dtype=bool)
    self.h = MyHeap()
    #print self.node_x
    #print self.node_y
  
  def get_targets_covered_by_node(self,n):  #returns a SUPERSET.
    x = self.node_x[n]
    y = self.node_y[n]
    x_min = int(max(0,  x - self.node_coverage) // self.target_interval)
    x_max = int(min(max_x, x + self.node_coverage) // self.target_interval)
    y_min = int(max(0,  y - self.node_coverage) // self.target_interval)
    y_max = int(min(max_y, y + self.node_coverage) // self.target_interval)
    for k in xrange(x_min, x_max+1):
      for l in xrange(y_min, y_max+1):
        yield k,l

  def build_heap(self):
    for i in xrange(self.number_of_nodes):
      total_targets_covered = 0
      for k,l in self.get_targets_covered_by_node(i):
          #self.target[k][l] = True
          total_targets_covered += 1
      self.node_target_count[i] = total_targets_covered
      print "DEBUG:",total_targets_covered
      #heapq.heappush(self.h, ( -total_targets_covered, i))
      self.h.add_item(i, -total_targets_covered)
      #print "DEBUG:",self.h.pq
      #x_length = int ((x_max - x_min)/self.target_interval)
      #y_length = int ((y_max - y_min)/self.target_interval)
    return
  
  def bin_nodes(self):
    self.bin_x_max = int(max_x // self.node_bin_interval  )
    self.bin_y_max = int(max_y // self.node_bin_interval  )
    for i in xrange(self.bin_x_max+1):
      self.node_bin[i] = {}
      for j in xrange(self.bin_y_max+1):
        self.node_bin[i][j] = []
        
    for n in xrange(self.number_of_nodes):
      x = self.node_x[n]
      y = self.node_y[n]
      bin_x = int( x // self.node_bin_interval)
      bin_y = int( y // self.node_bin_interval)
      self.node_bin[bin_x][bin_y].append(n)
    return
  
  def get_nodes_covering_target(self,i,j):   #returns Supersete
    tx = i*self.target_interval
    ty = j*self.target_interval
  
    min_bin_x = max(0, tx - self.node_coverage) // self.node_bin_interval
    max_bin_x = min(self.max_x, tx + self.node_coverage) // self.node_bin_interval
    min_bin_y = max(0, ty - self.node_coverage) // self.node_bin_interval
    max_bin_y = min(self.max_y, ty + self.node_coverage) // self.node_bin_interval
    
    nodes = set()
    for i in xrange(min_bin_x, max_bin_x+1):
      for j in xrange(min_bin_y, max_bin_y+1):
        nodes.update(self.node_bin[i][j])
    return nodes
  
  def is_covered_by_node(self,i,j,n):
    threshold = self.node_coverage
    x1 = i*self.target_interval
    y1 = j*self.target_interval
    x2 = self.node_x[n]
    y2 = self.node_y[n]
    dist = (x1-x2)**2+(y1-y2)**2
    if dist*dist <= threshold*threshold:
      return True
    else:
      return False
    
  def heuristic_set_cover(self):
    print "DEBUG: total targets:",self.total_targets
    while self.h.pq and self.targets_covered < self.total_targets:
      print self.h.pq
      n = self.h.pop_item() #Never using targets_covered.
      self.node_cover.append(n)
      print "DEBUG>>:",n
      print "DEBUG-->>:",self.h.pq
      if n == None:
        print "DEBUG: QUITTING"
        break
      for k,l in self.get_targets_covered_by_node(n):
        if not self.target[k][l] and self.is_covered_by_node(k, l, n):
          self.target[k][l] = True
          self.targets_covered += 1
          covering_nodes = self.get_nodes_covering_target(k, l) 
          for v in covering_nodes:
            if v !=n and self.is_covered_by_node(k, l, v): 
              #decreaseKeyinHEap(heap,n)
              print "DEBUG: before removing",self.h.pq
              self.node_target_count[v] -= 1
              #self.h.remove_item(v)
              print "DEBUG: after removing",self.h.pq
              self.h.add_item(v, - self.node_target_count[v])
              print "DEBUG: after re-adding",self.h.pq
          print "targets covered:",self.targets_covered
      ##update heap for these nodes.....
      
      
    return
  
  def get_node_cover(self):
    if not self.node_cover:
      self.build_heap()
      print "DEBUG:association complete..."
      self.heuristic_set_cover()
    return self.node_cover
  
if __name__ =='__main__':
  node_bin_interval  = 100
  number_of_nodes = 1000
  node_coverage = 100
  max_x = 100
  max_y = 100
  target_interval = 20
  tna = Target_Node_Assoc(number_of_nodes, max_x, max_y, target_interval, node_coverage, node_bin_interval ) 
  node_cover = tna.get_node_cover()
  print sorted(node_cover)
  
  
  
  
  
  
  
  
