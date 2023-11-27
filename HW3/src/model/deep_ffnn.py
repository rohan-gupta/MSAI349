import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.metrics import f1_score

from ..utils.dataset import split_X_y

class DeepNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def train(parameters, dataset_train, dataset_validation, dataset_name, path):
    input_size = parameters["input_size"]
    hidden_size = parameters["hidden_size"]
    output_size = parameters["output_size"]
    learning_rate = parameters["learning_rate"]
    weight_decay = parameters["weight_decay"]
    epochs = parameters["epochs"]

    model = DeepNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)

    X_train, y_train = split_X_y(dataset_train, dataset_name)
    X_valid, y_valid = split_X_y(dataset_validation, dataset_name)

    loss_train_vals, loss_valid_vals = [], []

    early_stopping_count = 50

    for epoch in range(epochs):
        outputs_train = model(X_train)
        loss_train = criterion(outputs_train, y_train)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        outputs_valid = model(X_valid)
        loss_valid = criterion(outputs_valid, y_valid)

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss_train.item():.4f}, Valid: {loss_valid.item():.4f}')

        loss_train_vals.append(loss_train.item())
        loss_valid_vals.append(loss_valid.item())

        if len(loss_valid_vals) < 2:
            continue

        if early_stopping_count < 0:
            print("Early stopping")
            break

        if loss_valid.item() - loss_valid_vals[-2] > 0.001:
            print("Validation loss increasing warning -", 50 - early_stopping_count)
            early_stopping_count -= 1

    path_model = path + "/model"
    path_loss = path + "/loss"

    torch.save(model.state_dict(), path_model)
    pickle.dump({"train": loss_train_vals, "valid": loss_valid_vals}, open(path_loss, "wb"))

    return model

def test(test_dataset, model):
    model.eval()

    with torch.no_grad():
        validation_X, validation_y = split_X_y(test_dataset, "mnist")

        _, predicted_y = model(validation_X).max(1)
        f1 = f1_score(validation_y, predicted_y, average='micro')
        return (predicted_y == validation_y).sum().item() / len(validation_y), f1, predicted_y
