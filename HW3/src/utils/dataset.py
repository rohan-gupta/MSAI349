import torch
import numpy as np


def split_X_y(dataset, dataset_name="insurability"):
    if dataset_name == "insurability":
        if isinstance(dataset, list):
            X, y = [], []

            for t in dataset:
                X.append(t[1])
                y.append(t[0][0])

            X, y = torch.tensor(X), torch.tensor(y)

        # elif isinstance(dataset, np.ndarray):
        #     X, y = dataset[:, 0].astype("float32"), dataset[:, 1:-1].astype("float32")
        #     X, y = torch.from_numpy(X), torch.from_numpy(y)

    if dataset_name == "mnist":
        X, y = [], []

        if not isinstance(dataset, list):
            return X, y

        for t in dataset:
            X.append([int(n) for n in t[1]])
            y.append(int(t[0]))

        X, y = torch.tensor(X), torch.tensor(y)

    return X.float(), y.long()
