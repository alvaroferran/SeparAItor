import matplotlib.pyplot as plt
import os


def plot_results(history, iteration, directory):
    # Clear any old plot
    plt.clf()

    # Plot loss
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    loss = "loss_fold" + str(iteration) + ".png"
    plt.savefig(os.path.join(directory, loss))

    # Plot accuracy
    plt.clf()
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    accuracy = "accuracy_fold" + str(iteration) + ".png"
    plt.savefig(os.path.join(directory, accuracy))
