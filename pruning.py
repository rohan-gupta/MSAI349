from node import Node
import math
from utils import *
import ID3

def prune(node, examples):
    # Reduced Error Pruning
    og_cost = ID3.get_entropy(node, examples)
    
    node = prune_tree(node, examples, og_cost)
    
    return node

def prune_tree(node, examples, og_cost):
    if node is None or node.is_leaf():
        return node
    
    original_node = node
    original_children = node.children
    
    node.make_leaf(get_majority_class(examples))  
    
    pruning_cost = ID3.get_entropy(node, examples)
    alpha = (og_cost - pruning_cost) / (len(examples) - 1)
    
    if alpha >= 0:
        return node  
    
    best_subtree = None
    for branch in original_children:
        child_subtree = prune_tree(original_children[branch], examples, og_cost)
        if child_subtree:
            if best_subtree is None:
                best_subtree = Node("", {})
            best_subtree.children[branch] = child_subtree
    
    return best_subtree

