import matplotlib.pyplot as plt
import numpy as np

def plot_histogram(data, title, xlabel, ylabel):
    # get each column and plot a histogram
    for i in range(data.shape[1]):
        voice_title = title + " Voice" + str(i + 1)
        plt.hist(data.iloc[:, i], bins=max(data.iloc[:, i]))
        plt.title(voice_title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig("Plots/" + voice_title + ".png")
        plt.show()
        plt.close()

def plot_data(data, title, xlabel, ylabel):
    # data is in form of a list with each element being a list of values
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig("Plots/" + title + ".png")
    plt.show()

def plot_certainty(preds, title, xlabel, ylabel):
    timesteps = len(preds)
    num_voices = preds[0].shape[0]
    num_classes = preds[0].shape[2]

    predictions = np.array(preds)
    predictions = np.squeeze(predictions, axis=2)

    for voice_idx in range(num_voices):
        # Create the heatmap for the current voice
        plt.imshow(predictions[:, voice_idx, :], aspect='auto', cmap='viridis', origin='lower')

        # Add a colorbar to indicate certainty (probabilities)
        plt.colorbar(label='Certainty (Probability)')
        
        # Set the labels and title
        plt.title(f"Certainty/Probability Heatmap for Voice {voice_idx + 1}")
        plt.xlabel("Class")
        plt.ylabel("Timestep")
        
        # Set class labels along the x-axis
        plt.xticks(ticks=np.arange(num_classes), labels=np.arange(1, num_classes + 1))

        # Save the plot
        plt.savefig(f"Plots/{title}_voice{voice_idx + 1}.png")

        # Show the plot
        plt.show()

def plot_class_weights(class_weights):
    # create main figure
    fig = plt.figure()
    
    # plot class weights for each voice
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        # plot histogram of class weights
        print(class_weights[i])
        plt.bar(range(len(class_weights[i])), class_weights[i])
        plt.title(f"Voice {i + 1} Class Weights")
    plt.savefig("Saved Plots/class_weights.png")
    plt.show()



