import torch

from ..simple_ffnn import SimpleNN, test
from ...utils.dataset import split_X_y

from .confusion_matrix import generate_confusion_matrix


def generate_metrics_for_q4(test_data):
    model = SimpleNN(3, 2, 3, False)
    model.load_state_dict(torch.load("src/model/trained/q4/model"))

    _, y = split_X_y(test_data)
    accuracy, f1, y_predicted = test(test_data, model)

    generate_confusion_matrix(y, y_predicted)

    print("Test accuracy / Q4: ", accuracy)
    print("F1 score / Q4: ", f1)
