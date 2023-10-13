# from node import Node
# import math
# from utils import *
# import ID3
# def prune_tree(node, examples, subtrees):
  
#   if node is None:
#     return subtrees

#   original_children = node.children

#   valid = examples[len(examples)//2:3*len(examples)//4]

#   og_acc = ID3.test(node, valid)
#   print("does this test work??")
#   node.make_leaf()  # Convert the node into a leaf
#   pruning_acc = ID3.test(node, valid)  # Calculate the cost on the validation set
#   print("how about this one?")
#   # If pruning is better, keep the leaf node; otherwise, revert to the original children
#   if pruning_acc >= og_acc:
#       subtrees.append(node)
#       node.make_leaf()  # Make the node a leaf by removing its children

#   # Recursively prune child nodes
#   for branch in original_children:
#       prune_tree(original_children[branch], valid, subtrees)

#   return subtrees



