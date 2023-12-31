import torch
import pickle

from ..deep_ffnn import DeepNN, test
from ...utils.dataset import split_X_y

from .confusion_matrix import generate_confusion_matrix
from .learning_curve import generate_learning_curve


def generate_metrics_for_q3(test_data):
    loss_values = pickle.load(open("src/model/trained/q3/loss", "rb"))
    model = DeepNN(784, 100, 10)
    model.load_state_dict(torch.load("src/model/trained/q3/model"))

    _, y = split_X_y(test_data, "mnist")
    accuracy, f1, y_predicted = test(test_data, model)

    generate_learning_curve(loss_values["train"], loss_values["valid"])
    generate_confusion_matrix(y, y_predicted)

    print("Test accuracy / Q3: ", accuracy)
    print("F1 score / Q3: ", f1)
