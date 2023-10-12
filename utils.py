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
