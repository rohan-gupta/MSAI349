import ID3
import math
import random, utils

def get_sub_datasets_by_attribute(dataset, attribute):
  if attribute == "":
    return dataset
  
  sub_datasets = {}

  for d in dataset:
    if attribute not in d:
      continue

    if d[attribute] not in sub_datasets:
      sub_datasets[d[attribute]] = []

    sub_datasets[d[attribute]].append(d)

  return list(sub_datasets.values())


def get_target_class_probabilities(dataset):
  target_class_frequencies = get_target_class_frequencies(dataset)
  probabilities = {}

  for k, v in target_class_frequencies.items():
    probabilities[k] = v / len(dataset)

  return probabilities


def get_target_class_frequencies(dataset):
  target_class_frequencies = {}
  
  for d in dataset:
    if "Class" not in d:
      continue

    if d["Class"] not in target_class_frequencies:
      target_class_frequencies[d["Class"]] = 0

    target_class_frequencies[d["Class"]] += 1

  return target_class_frequencies


def is_dataset_empty(dataset):
  return not dataset


def is_dataset_trivial(dataset):
  classes = set()

  for d in dataset:
    if "Class" not in d:
      continue

    classes.add(d["Class"])

  return len(classes) == 1


def get_all_attributes(dataset):
  all_attributes = set()

  for d in dataset:
    for k in d:
      if k == "Class":
        continue
      
      all_attributes.add(k)

  return list(all_attributes)


def get_all_target_classes(dataset):
  all_classes = set()

  for d in dataset:
    if "Class" not in d:
      continue

    all_classes.add(d["Class"])

  return list(all_classes)


def get_majority_class(dataset):
  target_class_frequencies = get_target_class_frequencies(dataset)
  majority_class = ""
  majority_freq = 0
  
  for k, v in target_class_frequencies.items():
    if v > majority_freq:
      majority_class = k
      majority_freq = v

  return majority_class


def get_non_trivial_attribute(dataset):
  attributes = {}
  
  for d in dataset:
    for k, v in d.items():
      if k == "Class":
        continue
      
      if k not in attributes:
        attributes[k] = set()

      attributes[k].add(v)

      if len(attributes[k]) > 1:
        return k
      
  return ""


def update_missing_attributes_with_majority_value(dataset):
  attributes = get_all_attributes(dataset)
  
  for a in attributes:
    frequencies = {}
    
    for d in dataset:
      if d[a] == "?":
        continue

      if d[a] not in frequencies:
        frequencies[d[a]] = 0

      frequencies[d[a]] += 1
    
    majority_value = max(frequencies, key = frequencies.get)
    
    for d in dataset:
      if d[a] == "?":
        d[a] = majority_value

  return dataset



'''

#node is a fully trained tree.
def prune_tree(node, examples):
    if node is None:
        return

    valid = examples[len(examples)//2:3*len(examples)//4]

    # Attempt to prune the node
    og_cost = get_entropy(examples)
    original_children = node.children
    print("TEST", original_children)

    #og_cost works

    node.make_leaf()  # Convert the node into a leaf
    pruning_cost = get_entropy(valid)  # Calculate the cost on validation set

    print("diff", og_cost - pruning_cost)

    #og and pruning cost functions work and are different, but the difference is going to 0 
    alpha = (og_cost - pruning_cost) / (len(valid) - 1)
    #AS OF NOW, ALPHA IS ALWAYS 0.

    # If pruning is better, keep the leaf node; otherwise, revert to the original children
    if alpha <= 0:
        node.children = original_children

    # Recursively prune child nodes
    for branch in original_children:
      prune_tree(original_children[branch], examples)
'''


def prune_tree(node, examples, subtrees):
  
  if node is None:
    return subtrees

  original_children = node.children

  valid = examples[len(examples)//2:3*len(examples)//4]
  valid = examples 
  if len(valid) >5 :

    og_acc = ID3.test(node, valid)

    node.make_leaf(node.label)  # Convert the node into a leaf
    pruning_acc = ID3.test(node, valid)  # Calculate the cost on the validation set
    # print(len(valid))
    # If pruning is better, keep the leaf node; otherwise, revert to the original children
    if pruning_acc >= og_acc:
        subtrees.append(node)
        node.make_leaf(node.label)  # Make the node a leaf by removing its children

  # Recursively prune child nodes
  if len(valid) > 5:
    for branch in original_children:
      prune_tree(original_children[branch], valid, subtrees)
        

  return subtrees





