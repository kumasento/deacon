import math

def num_btree_nodes(N):
  return 2 ** math.ceil(math.log2(N)) - 1
