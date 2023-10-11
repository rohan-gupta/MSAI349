from node import Node
import math
import json

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
    
    if label == e["Class"]:
      correct_labels += 1

  return correct_labels / len(examples)


def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''

  while node.label != "+" and node.label != "-":
    branch = example[node.label]
    node = node.children[branch]

  return 1 if node.label == "+" else 0


def get_information_gain(dataset, attribute):
  sub_dataset = get_sub_datasets_by_attribute(dataset, attribute)
  parent_entropy = get_entropy(dataset)
  child_entropy = 0

  for s in sub_dataset:
    child_entropy += (len(s) / len(dataset)) * get_entropy(s)
  
  return parent_entropy - child_entropy
  

def get_sub_datasets_by_attribute(dataset, attribute):
  sub_datasets = {}

  for d in dataset:
    if attribute not in d:
      continue

    if d[attribute] not in sub_datasets:
      sub_datasets[d[attribute]] = []

    sub_datasets[d[attribute]].append(d)

  return list(sub_datasets.values())


def get_entropy(dataset):
  target_class_probabilities = get_target_class_probabilities(dataset)
  H = 0

  for _, v in target_class_probabilities.items():
    H += -1 * v * math.log2(v)

  return H


def get_target_class_probabilities(dataset):
  target_class_frequencies = get_target_class_frequencies(dataset)
  probabilities = {}

  for k, v in target_class_frequencies.items():
    probabilities[k] = v / len(dataset)

  return probabilities


def get_target_class_frequencies(dataset):
  target_class_frequencies = {}
  
  for d in dataset:
    for k, v in d.items():
      if k != "Class":
        continue

      if v not in target_class_frequencies:
        target_class_frequencies[v] = 0

      target_class_frequencies[v] += 1

  return target_class_frequencies


def is_dataset_empty(dataset):
  return not dataset


def is_dataset_positive(dataset):
  for d in dataset:
    if "Class" not in d:
      continue

    if d["Class"] == 0:
      return False

  return True


def is_dataset_negative(dataset):
  for d in dataset:
    if "Class" not in d:
      continue

    if d["Class"] == 1:
      return False

  return True


def get_all_attributes(dataset):
  all_attributes = set()

  for d in dataset:
    for k in d:
      if k == "Class":
        continue
      
      all_attributes.add(k)

  return list(all_attributes)


def get_best_attribute_by_max_information_gain(dataset):
  attributes = get_all_attributes(dataset)

  best_attribute = ""
  best_attribute_information_gain = -1

  for a in attributes:
    temp = get_information_gain(dataset, a)

    if temp > best_attribute_information_gain:
      best_attribute = a
      best_attribute_information_gain = temp
  
  return best_attribute


def get_decision_tree(ptr, dataset):
  if is_dataset_empty(dataset):
    return None

  if is_dataset_positive(dataset):
    ptr.label = "+"
    return ptr

  if is_dataset_negative(dataset):
    ptr.label = "-"
    return ptr

  attribute = get_best_attribute_by_max_information_gain(dataset)
  sub_dataset_by_attribute = get_sub_datasets_by_attribute(dataset, attribute)
  
  ptr.label = attribute

  for s in sub_dataset_by_attribute:
    branch = s[0][attribute]
    ptr.children[branch] = Node("", {})
    get_decision_tree(ptr.children[branch], s)
  
  return ptr