######## Importing the essential libraries and functions ########

import numpy as np
import random
from scipy.stats import wasserstein_distance
import warnings
warnings.filterwarnings('ignore')


######## Distance Functions based on different criterions ########

## Euclidean Distance ##
def euclidean_distance(v1,v2):
  '''
  Function to calculate euclidean distance between two vectors v1,v2
  Input: 
        v1, v2: np.array
  Output: 
        Euclidean distance: np.array 
  '''
  return np.sqrt(np.sum(np.power(v2 - v1, 2)))



## Earth Mover distance (support) ##
def get_signature(img):
  '''
  Support function to calculate signature given an image as input
  Input: 
        img: np.array
  Output: 
        Histogram signature : np.array
  '''
  h, w = img.shape
  img = (img*255).astype(int)
  hist = [0.0] * 256
  for i in range(h):
    for j in range(w):
      hist[img[i, j]] += 1
  return np.array(hist) / (h * w) 


def earth_movers_distance(img_a, img_b):
  '''
  Function to calculate Earth Mover distance between two vectors v1,v2
  Input: 
        v1, v2: np.array
  Output: 
        Earth Mover distance: np.array 
  '''
  hist_a = get_signature(img_a)
  hist_b = get_signature(img_b)
  return wasserstein_distance(hist_a, hist_b)

######## Class definition for Metric Tree ########


class MTree:
  '''
  Class- Object to generate a Metric Tree

  Input
  -----
  S: np.array 2D
     Array of points to be distributed in tree nodes
  distance_func: str, default = 'eucd' (Euclidean)
                 Distance function to be used for calculation.
                 {'eucd' for Euclidean, 'emd' for Earth Mover Distance}
  Output
  ------
  Metric Tree: MTree class-Object


  Methods
  -------
  @get_nn_pruning: 

    For searching 'n_neighbors' nearest neighbors 
    a query point 'q' using the Pruning search method

    input: q: np.array 
           n_neighbors: int

    Output: 
           distance: tuple (d,nn)
              distance to q,nearest neighbors
           nodes_visited: int
              number of nodes visited in the search
  

  @get_nn_defeatist: 

    For searching 'n_neighbors' nearest neighbors 
    a query point 'q' using the Defeatist search method

    input: q: np.array 
           n_neighbors: int

    Output: 
           distance: tuple (d,nn)
              distance to q,nearest neighbors
           nodes_visited: int
              number of nodes visited in the search
  '''
  def __init__(self,S, distance_func = 'eucd'):
    # initializing
    self.S = S
    self.r_min = np.inf
    self.r_max = 0
    self.l_min = np.inf
    self.l_max = 0
    self.distance_func = distance_func

    self.left = None
    self.right = None

    # Choosing right distance function
    self.distance_func_dict = {'eucd':euclidean_distance,'emd': earth_movers_distance}
    self.distance_func = self.distance_func_dict[self.distance_func]


    # Return nothing for empty S
    if not len(self.S):
      return
    
    # initial node random choice
    node_i = random.randint(0,len(S)-1)
    self.v = S[node_i]
    p = np.delete(S, node_i, axis=0)


    #calculating the distance
    d_list = [self.distance_func(self.v,p1) for p1 in p]
    self.mu = np.median(d_list)

    # Start making the tree
    # First split into right and left
    left_S = []
    right_S = []

    # Distribution of points into left and right of the tree
    # based on the distance. Right node is outer region of metric space of
    # a node and left is for inner

    for point, distance in zip(p, d_list):
        if distance >= self.mu:
            self.r_min = min(distance, self.r_min)
            if distance > self.r_max:
                self.r_max = distance
                right_S.insert(0, point)
            else:
                right_S.append(point)
        else:
            self.l_min = min(distance, self.l_min)
            if distance > self.l_max:
                self.l_max = distance
                left_S.insert(0, point)
            else:
                left_S.append(point)

    # Recursively calling the class to generate split on left node
    if len(left_S) > 0:
        self.left = MTree(S=left_S)

    # Recursively calling the class to generate split on right node
    if len(right_S) > 0:
        self.right = MTree(S=right_S)

  def _is_leaf(self):
    #to check if the node is a leaf
    return (self.left is None) and (self.right is None)

  def get_nn_pruning(self, q, n_neighbors = 1):
      # search using pruning technique

      neighbors = []
      nodes_to_visit = [(self, 0)]
      nodes_visited = nodes_to_visit.copy()

      tau = np.inf

      while len(nodes_to_visit) > 0:
          node, d0 = nodes_to_visit.pop(0)
          if node is None or d0 > tau:
              continue

          l = self.distance_func(q, node.v)
          if l < tau:
              neighbors.append((l, node.v))
              tau, _ = neighbors[-1]
              neighbors.sort(key=lambda x: x[0])

          if node._is_leaf():
              continue

          if node.l_min <= l <= node.l_max:
              nodes_to_visit.insert(0, (node.left, 0))
              nodes_visited.insert(0, (node.left, 0))
          elif node.l_min - tau <= l <= node.l_max + tau:
              nodes_to_visit.append((node.left,
                                      node.l_min - l if l < node.l_min
                                      else l - node.l_max))
              nodes_visited.append((node.left,
                                      node.l_min - l if l < node.l_min
                                      else l - node.l_max))

          if node.r_min <= l <= node.r_max:
              nodes_to_visit.insert(0, (node.right, 0))
              nodes_visited.insert(0, (node.right, 0))
          elif node.r_min - tau <= l <= node.r_max + tau:
              nodes_to_visit.append((node.right,
                                      node.r_min - l if l < node.r_min
                                      else l - node.r_max))
              nodes_visited.append((node.right,
                                      node.r_min - l if l < node.r_min
                                      else l - node.r_max))

      return neighbors[:n_neighbors],len(nodes_visited)

  def get_nn_defeatist(self, q, n_neighbors = 1):
      # search using defeatist technique   

      nodes_to_visit = [(self, 0)]

      while True:
          node, d0 = nodes_to_visit[-1]
            
          
          d = self.distance_func(q, node.v)
          
          if (len(node.S) < 2 ):
                return (d, node.v),len(nodes_to_visit)
          
          

          if d < node.mu:
              if node.left is None:
                return (d, node.v),len(nodes_to_visit)
              else:
                nodes_to_visit.append((node.left, d ))
              

          else:
              if node.right is None:
                return (d, node.vp),len(nodes_to_visit)
              else:
                nodes_to_visit.append(( node.right, d))

######## Class definition for Metric Forest ########

class MForest:
  '''
  Class- Object to generate a Metric Forest given a number trees 'n_trees' 
  and input space 'S'

  Input
  -----
  S: np.array 2D
     Array of points to be distributed in tree nodes
  n_trees: int, default = 10
           number of trees in the forest
  distance_func: str, default = 'eucd' (Euclidean)
                 Distance function to be used for calculation.
                 {'eucd' for Euclidean, 'emd' for Earth Mover Distance}
  Output
  ------
  Metric Forest: MForest class-Object


  Methods
  -------
  @get_nn: 

    For searching single nearest neighbors 
    a query point 'q' using the Pruning or defeatist search method

    input: q: np.array 
           approach: str,default = 'dft'
                Choose 'dft' for defeatist and 'prn' for pruning

    Output: 
           list_nn: tuple (d,nn)
              distance to q,nearest neighbors point
           nodes_visited: int
              number of nodes visited in the search 

 '''
  
  def __init__(self,S, n_trees = 10,distance_func = 'eucd'):
    # init Forest
    self.n_trees = n_trees
    self.S = S
    self.distance_func = distance_func

    # initializing 'n_trees' in the forest
    self.forest_trees = [MTree(S,distance_func = distance_func) for i in range(n_trees)] 

  def get_nn(self, q, approach = 'dft'):

    list_nn = []
    nodes_visited = 0
    for tree in self.forest_trees:
      if approach == 'dft':
        l,nv = tree.get_nn_defeatist(q)
        
      else:
        l,nv = tree.get_nn_pruning(q)
      
      list_nn.append(l)
      nodes_visited += nv

    if approach == 'dft':
      list_nn = sorted(list_nn,key = lambda x: x[0])[0]
    else:
      list_nn = list_nn[0][0]
    return list_nn,nodes_visited


