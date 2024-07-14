
import matplotlib.pyplot as plt

# Draw training performance results
def save_history_plot(history, path: str = './training_results.png'):

    plt.figure(figsize=(16, 5))

    epochs = len(history.history['loss'])

    # Loss (based on Dice coefficient)
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), history.history['loss'], label='Loss - training')
    plt.plot(range(epochs), history.history['val_loss'], label='Loss - validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # The dice coefficient itself
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), history.history['dice_coefficient'], label='Dice coefficient - training')
    plt.plot(range(epochs), history.history['val_dice_coefficient'], label='Dice coefficient - validation')
    plt.title('Training and Validation Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    plt.savefig(path)
