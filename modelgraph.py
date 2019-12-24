import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

model_name = "model-1577204302"


def create_acc_loss_graph(model_name, EPOCHS, BATCH_SIZE, TRAIN_SIZE):
    contents = open("model.log", "r").read().split("\n")

    times = []
    accuracies = []
    losses = []

    val_accs = []
    val_losses = []

    for c in contents:
        if model_name in c:
            name, time, training, train_acc, train_loss, validation, val_acc, val_loss = c.split(",")

            times.append(float(time))
            accuracies.append(float(train_acc))
            losses.append(float(train_loss))

            val_accs.append(float(val_acc))
            val_losses.append(float(val_loss))

    fig = plt.figure()
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)

    ax1.plot(times, accuracies, label=f'training accuracy, max={max(accuracies)}')
    ax1.plot(times, val_accs, label=f'validation accuracy, max={max(val_accs)}')
    ax1.legend(loc=2)

    ax2.plot(times, losses, label=f'training loss, min={min(losses)}')
    ax2.plot(times, val_losses, label=f'validation loss, min={min(val_losses)}')
    ax2.legend(loc=2)

    fig.suptitle(f'Epoch: {EPOCHS}, Batch size: {BATCH_SIZE}, Train size: {TRAIN_SIZE}', fontsize=16)
    plt.show()
