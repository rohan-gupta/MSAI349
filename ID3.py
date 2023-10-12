from node import Node
import math

from utils import *

THRESHOLD_ENTROPY = 0.05
THRESHOLD_DATASET_SIZE = 3

def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''
  root = Node("", {})

  get_decision_tree(root, examples)

  return root


def prune(node, examples):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''


def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''
  
  correct_labels = 0

  for e in examples:
    label = evaluate(node, e)
    
    if str(label) == str(e["Class"]):
      correct_labels += 1

  return correct_labels / len(examples)


def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''
  
  while node != None and node.children != {}:
    branch = example[node.label]
    node = node.children[branch]

  return node.label


def get_decision_tree(ptr, dataset):
  
  if is_dataset_empty(dataset):
    return None

  if get_entropy(dataset) <= THRESHOLD_ENTROPY or len(dataset) <= THRESHOLD_DATASET_SIZE:
    ptr.label = get_majority_class(dataset)
    return ptr

  attribute = get_best_attribute_by_max_information_gain(dataset)
  sub_dataset_by_attribute = get_sub_datasets_by_attribute(dataset, attribute)
  
  if attribute == "":
    ptr.label = get_majority_class(dataset)
    return ptr

  ptr.label = attribute
    
  for s in sub_dataset_by_attribute:  
    branch = s[0][attribute]
    ptr.children[branch] = Node("", {})
    get_decision_tree(ptr.children[branch], s)
  
  return ptr


def get_best_attribute_by_max_information_gain(dataset):
  attributes = get_all_attributes(dataset)
  best_attribute = ""
  best_attribute_information_gain = -1

  for a in attributes:
    temp = get_information_gain(dataset, a)
    
    if temp > best_attribute_information_gain:
      best_attribute = a
      best_attribute_information_gain = temp
  
  if best_attribute_information_gain == 0:
    return get_non_trivial_attribute(dataset)
  
  return best_attribute


def get_information_gain(dataset, attribute):
  sub_dataset = get_sub_datasets_by_attribute(dataset, attribute)
  parent_entropy = get_entropy(dataset)
  child_entropy = 0

  for s in sub_dataset:
    child_entropy += (len(s) / len(dataset)) * get_entropy(s)
  
  return parent_entropy - child_entropy


def get_entropy(dataset):
  target_class_probabilities = get_target_class_probabilities(dataset)
  H = 0

  for _, v in target_class_probabilities.items():
    H += -1 * v * math.log2(v)

  return H
