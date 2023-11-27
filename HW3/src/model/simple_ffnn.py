import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from ..utils.dataset import split_X_y

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bias):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size, bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = custom_softmax(x)
        return x


def train(parameters, dataset_training, dataset_validation, dataset_name, path):
    input_size = parameters["input_size"]
    hidden_size = parameters["hidden_size"]
    output_size = parameters["output_size"]
    learning_rate = parameters["learning_rate"]
    bias = parameters["bias"]
    epochs = parameters["epochs"]

    model = SimpleNN(input_size, hidden_size, output_size, bias)
    criterion = nn.NLLLoss()
    optimizer = CustomSGD(model.parameters(), learning_rate)

    X_train, y_train = split_X_y(dataset_training, dataset_name)
    X_validation, y_validation = split_X_y(dataset_validation, dataset_name)

    loss_train_vals, loss_valid_vals = [], []

    for epoch in range(epochs):
        temp = []

        for i in range(len(X_train)):
            output_train = model(X_train[i])
            loss_train = criterion(torch.log(output_train),  y_train[i])

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            temp.append(loss_train.item())

        output_validation = model(X_validation)
        loss_validation = criterion(torch.log(output_validation), y_validation)

        loss_train_vals.append(sum(temp) / len(temp))
        loss_valid_vals.append(loss_validation.item())

        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {sum(temp) / len(temp)}, Validation Loss: {loss_validation.item():.4f}')

    path_model = path + "/model"
    path_loss = path + "/loss"

    torch.save(model.state_dict(), path_model)
    pickle.dump({"train": loss_train_vals, "valid": loss_valid_vals}, open(path_loss, "wb"))

    return model


def test(test_dataset, model):
    model.eval()

    with torch.no_grad():
        validation_X, validation_y = split_X_y(test_dataset)

        _, predicted_y = model(validation_X).max(1)
        return (predicted_y == validation_y).sum().item() / len(validation_y), predicted_y


def custom_softmax(x):
    return torch.exp(x) / torch.exp(x).sum(axis=0)


class CustomSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(CustomSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                p.data.add_(-group['lr'], grad)

        return loss
