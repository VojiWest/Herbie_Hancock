import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def plot_scatter(data, title="Scatter Plot of Notes"):
    """Plots a scatter plot of the notes for the four voices over time."""
    time_steps = np.arange(data.shape[0])
    plt.figure(figsize=(10, 6))
    for voice in range(data.shape[1]):
        plt.scatter(time_steps, data.iloc[:, voice], label=f"Voice {voice + 1}")
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Note Value")
    plt.legend()
    plt.grid()
    plt.show()

def plot_histograms(data, title="Histogram of Notes"):
    """Plots histograms for the note values of the four voices."""
    plt.figure(figsize=(10, 8))
    for voice in range(data.shape[1]):
        plt.subplot(2, 2, voice + 1)
        plt.hist(data.iloc[:, voice], bins=100, color=f"C{voice}", alpha=0.7)
        plt.title(f"Voice {voice + 1}")
        plt.xlabel("Note Value")
        plt.ylabel("Frequency")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_3d(data, title="3D Visualization of Notes"):
    """Plots a 3D visualization of the notes for the four voices over time."""
    time_steps = np.arange(data.shape[0])
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    for voice in range(data.shape[1]):
        ax.plot(time_steps, [voice + 1] * len(time_steps), data.iloc[:, voice], label=f"Voice {voice + 1}")
    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Voice")
    ax.set_zlabel("Note Value")
    ax.legend()
    plt.show()

def plot_zero_distribution_per_voice(data, title="Distribution of Silence (0) vs Non-Zero Values"):
    """Plots the distribution of 0 values (silence) to non-zero values for all voices."""
    zero_counts = (data == 0).sum(axis=0)
    non_zero_counts = (data != 0).sum(axis=0)
    voices = [f"Voice {i+1}" for i in range(data.shape[1])]

    plt.figure(figsize=(10, 6))
    bar_width = 0.4
    x = np.arange(len(voices))
    plt.bar(x - bar_width/2, zero_counts, width=bar_width, label="Zero Values", color="skyblue")
    plt.bar(x + bar_width/2, non_zero_counts, width=bar_width, label="Non-Zero Values", color="orange")
    plt.xticks(x, voices)
    plt.title(title)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # Load the data from the file "F.txt"
    path = "F.txt"  # File path to the text file
    data = pd.read_csv(path, sep="\t", header=None)  # Load tab-separated values

    plot_scatter(data, title="Scatter Plot of Notes")
    plot_histograms(data, title="Histogram of Notes")
    plot_3d(data, title="3D Visualization of Notes")
    plot_zero_distribution_per_voice(data, title="Distribution of Silence (0) vs Non-Zero Values")

if __name__ == "__main__":
    main()
