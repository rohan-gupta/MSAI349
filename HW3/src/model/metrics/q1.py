import torch
import pickle

from ..simple_ffnn import SimpleNN, test
from ...utils.dataset import split_X_y

from .confusion_matrix import generate_confusion_matrix
from .learning_curve import generate_learning_curve


def generate_metrics_for_q1(test_data):
    # loss_values1 = pickle.load(open("src/model/trained/q1/q1-model-bias-lr-01-loss", "rb"))
    # loss_values2 = pickle.load(open("src/model/trained/q1/q1-model-bias-lr-001-loss", "rb"))
    # loss_values3 = pickle.load(open("src/model/trained/q1/q1-model-no-bias-lr-001-loss", "rb"))

    # generate_learning_curve(loss_values1["train"], loss_values1["valid"])
    # generate_learning_curve(loss_values2["train"], loss_values2["valid"])
    # generate_learning_curve(loss_values3["train"], loss_values3["valid"])

    # loss_values = pickle.load(open("src/model/trained/q1/loss", "rb"))
    model = SimpleNN(3, 2, 3, True)
    model.load_state_dict(torch.load("src/model/trained/q1/q1-model-bias-lr-001"))

    _, y = split_X_y(test_data)
    accuracy, f1, y_predicted = test(test_data, model)

    # generate_learning_curve(loss_values["train"], loss_values["valid"])
    generate_confusion_matrix(y, y_predicted)

    print("Test accuracy / Q1: ", accuracy)
    print("F1 score / Q1: ", accuracy)


