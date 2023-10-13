class Node:
  def __init__(self, label = "", children = {}):
    self.label = label
    self.children = children
    self.is_leaf = False

  def make_leaf(self):
    self.is_leaf = True
    self.children = {}