import matplotlib.pyplot as plt
from utils.pickle_opt import pickle_write, pickle_read
import numpy as np
def loss_manipulation(world_size):
    all_loss = []
    for i in range(world_size):
        all_loss.append(pickle_read(f"intermediate_data/rank{i}_loss.pickle"))
    all_loss = np.mean(np.array(all_loss), axis = 0)
    return all_loss



def plot_loss_in_training_process(all_loss):
    plt.figure(figsize=(8, 6))
    plt.plot(all_loss, color='blue')
    plt.title("Training loss")
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.grid(True)
    plt.show()
    plt.savefig('plot/training_loss.png')

def main_visualization(world_size):
    plot_loss_in_training_process(loss_manipulation(world_size))