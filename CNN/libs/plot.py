import os
import datetime
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def create_plot_directory(plot_dir):
    dt = datetime.datetime.now()
    subdir = f"{dt.year}-{dt.month}-{dt.day}_{dt.hour}-{dt.minute}"
    plot_subdir = os.path.join(plot_dir, subdir)
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    if not os.path.isdir(plot_subdir):
        os.mkdir(plot_subdir)
    return plot_subdir


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
