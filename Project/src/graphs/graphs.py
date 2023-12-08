import matplotlib.pyplot as plt

def plot_loss_curve(train_loss, val_loss):
    # Ensure the length of train_losses and val_losses are the same
    assert len(train_loss) == len(val_loss), "Length of train_losses and val_losses must be the same"

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

