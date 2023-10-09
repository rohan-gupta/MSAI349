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


def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.
  '''

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

		if d[attribute] not in sub_dataset:
			sub_datasets[d[attribute]] = []

		sub_datasets[d[attribute]].append(d)

	return list(sub_datasets.values())
