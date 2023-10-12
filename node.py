class Node:
  def __init__(self, label = "", children = {}):
    self.label = label
    self.children = children


  #functions for pruning!!
  def is_leaf(self):
    return len(self.children) == 0

  #clears the children of a given node to make it into a leaf
  def make_leaf(self, class_label):
        self.label = class_label
        self.children = {}