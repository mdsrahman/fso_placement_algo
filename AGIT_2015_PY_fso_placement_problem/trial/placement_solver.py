import copy
from collections import defaultdict
import random
import queue

class Placement_Solver:
  def __init__(self,sinks,nodes,targets):
    self.sink=copy.deepcopy(sinks)
    self.node=copy.deepcopy(nodes)
    self.target = copy.deepcopy(targets)
    self.numnodes = len(targets)
    self.bfs_path = None
    self.E=[]
    self.targets_covered = []
    self.sink_connected_nodes  =  None
    
  def greedy_set_cover(self):
    F=[]
    for i in range(self.numnodes):
      F.append(set(self.target[i]))
    D = defaultdict(list)
    for y,S in enumerate(F):
        for a in S:
            D[a].append(y)
    
    L=defaultdict(set)        
    # Now place sets into an array that tells us which sets have each size
    for x,S in enumerate(F):
        L[len(S)].add(x)
    
    #self.E=[] # Keep track of selected sets
    # Now loop over each set size
    for sz in range(max(len(S) for S in F),0,-1):
        if sz in L:
            P = L[sz] # set of all sets with size = sz
            while len(P):
                x = P.pop()
                self.E.append(x)
                for a in F[x]:
                    for y in D[a]:
                        if y!=x:
                            S2 = F[y]
                            L[len(S2)].remove(y)
                            S2.remove(a)
                            L[len(S2)].add(y)
    return
  
  def find_reachable_nodes(self, n):
    r=[]
    for i in range(self.numnodes):
      if i!=n and self.node[n][i]==1:
        r.append(i)
    return r
  #---------------------------------------------------------------------------%%%%%%%%%%%%%%%
  def calculate_node_degrees(self):
    self.total_nodes = 0
    self.degree=[0 for i in range(self.numnodes)]
    for i in range(self.numnodes):
      n_sum =  sum(self.bfs_path[i])
      if n_sum>0:
        self.total_nodes += 1
        self.degree[i] = n_sum
    print("Total Nodes after BFS: ",self.total_nodes)
    print("Total Nodes: ",self.numnodes)
    max_deg = max(self.degree)
    max_deg_node = [i for i, j in enumerate(self.degree) if j == max_deg]
    print("Max degree: ",max_deg)
    print("Max degree nodes:",max_deg_node)
    
    for i in range(self.numnodes):
      print("Node:",i," Degree:",self.degree[i],end=" ")
      if i in self.sink_connected_nodes:
        print("sink-connected")
      print()

    #calculate node degrees
    
    
    return
  
  def find_closest(self, S, G_p, T):
    q = queue.Queue()
    for i in S:
      q.put(i)
      
    vnodes = []
    path = [-1 for i in range(self.numnodes)]
    while not q.empty():
      s=q.get()
      print("DEBUG:expanding node: ",s)
      
      if s not in vnodes[:]:
        vnodes.append(s)
        if s in T:
            print("DEBUG:found target: ",s)
            pn = s
            print("path:",path)
            while path[pn]!=-1:
              x=pn
              y=path[pn]
              G_p[x][y]=G_p[y][x]=1
              pn = path[pn]
            return s, G_p 
        else:
            #expand fringe & push in the queue
            r=self.find_reachable_nodes(s)
            print("reachable nodes r: ",r)
            for n_node in r[:]:
              if n_node not in vnodes[:]:
                path[n_node] = s #update path to reflect parental relationship
                q.put(n_node)
            
        
    return -1, G_p
  #-------------------------------------
  def modified_bfs(self):
    S=[]
    
    for i in self.sink.values():
      S.extend(i)
    self.sink_connected_nodes = copy.deepcopy(S)
    t_covered=[] 
    T=copy.deepcopy(self.E)

    
    G_p = [[0 for x in range(self.numnodes)] for x in range(self.numnodes)] 
    
    print("-------")
    print(S)
    print(T)
    print(G_p)
    
    while T:
      print("cur T: ",T)
      print("cur S: ",S)
      t, G_p = self.find_closest(S, G_p, T)
      if(t==-1):
        print("DEBUG:@modified_bfs(): couldn't find any new target, terminating while loop!")
        break
      else:
        T.remove(t)
        S.append(t)
        t_covered.append(t)
    self.bfs_path=copy.deepcopy(G_p)
    print("finally covered Targets:",t_covered)
    self.targets_covered = copy.deepcopy(t_covered)
    count_not_covered =  len(self.E) - len(t_covered)
    print(">>>Total Target Locations: ",len(self.E))
    print(">>>Not covered:",count_not_covered)
    return
   #---------------------------------------------------------------------------%%%%%%%%%%%%%%%%%%%%%

def generate_input(numnodes, 
                   numsinks, 
                   numtargets, 
                   max_target_assoc = 2, 
                   max_sink_assoc = 5,
                   connectivity_prob= 0.5):
  #sink configuration
  node_collection=[i for i in range(numnodes)]
  print("DEBUG@generate_input(..):node-collection:",node_collection)
  sinks = defaultdict(list)
  max_k =  max_sink_assoc
  for i in range(numsinks):
    sinks[i].extend( random.sample(node_collection, random.randint(1, 1+max_k) ) )
  #end of sink configuration
  
  #node configuration
  nodes = [[0 for x in range(numnodes)] for x in range(numnodes)] 
  for i in range(numnodes):
    nodes[i][i] = 1
    for j in range(i+1, numnodes):
      p = random.random()
      #print("p:",p)
      if p >=  1 - connectivity_prob:
        nodes[i][j] = nodes[j][i] = 1
      else:
        nodes[i][j] = nodes[j][i] = 0
  #end of node configuration
  
  #target  configuration
  targets = {}
  for i in range(numnodes):
    targets[i]=[]
  max_k =  random.randint(1,max_target_assoc)
  for i in range(numtargets):
    node_for_target = random.sample(node_collection, random.randint(1, max_k) )
    for j in node_for_target:
      targets[j].append(i)
  #end of target configuration
  
  return sinks,nodes,targets  

if __name__ == "__main__":
  random.seed(1019307)
  numnodes = 200
  numsinks = 10
  numtargets = 100
  max_target_assoc = 10
  max_sink_assoc = 2
  connectivity_prob = 0.07
  
  sinks,nodes,targets = generate_input(numnodes, 
                                       numsinks, 
                                       numtargets,
                                       max_target_assoc,
                                       max_sink_assoc, 
                                       connectivity_prob)
  
  
  print("DEBUG:sink-node associativity-----------------")
  for i in sorted(sinks.keys()):
    print(i," : ",sinks[i])
  
  print("DEBUG:node-node associativity-----------------") 
  for i in range(numnodes):
    print("node ",i," :",end='')
    for j in range(numnodes):
      if nodes[i][j]==1:
        print(j,end=', ')
    print()
  
  print("DEBUG:target-node associativity-----------------")
  for i in sorted(targets.keys()):
    print(i," : ",sorted(targets[i]))

  
  ps=Placement_Solver(sinks,nodes,targets)
  ps.greedy_set_cover()
  print("E-----------------------------*******************")
  print(sorted(ps.E))
  
  ps.modified_bfs()
  for i in range(ps.numnodes):
    print("Node: ",i," :",end=" ")
    for j in range(ps.numnodes):
      if ps.bfs_path[i][j]==1:
        print(j,end=" ")
    print()
  ps.calculate_node_degrees()
    #print(ps.bfs_path)
  
  
  
  
  
  
  
  
  
  
  