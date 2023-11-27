import matplotlib.pyplot as plt

def generate_learning_curve(loss_values_train, loss_values_valid):
    epochs = list(range(1, len(loss_values_train) + 1))

    plt.plot(epochs, loss_values_train, linestyle='-', label='Training')
    plt.plot(epochs, loss_values_valid, linestyle='-', label='Validation')

    plt.title('Training and Validation Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (or other metric)')
    plt.legend()

    plt.grid(True)
    plt.show()

