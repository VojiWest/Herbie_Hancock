import matplotlib.pyplot as plt

def main():
    data_counts = [0, 7, 51, 28, 5, 28, 99, 71, 12, 19, 62, 2]
    corresponding_notes = [0, 54, 59, 61, 62, 63, 64, 66, 68, 69, 73, 74]

    # Create data with data_counts for each corresponding_note
    data = []
    for i in range(len(data_counts)):
        data += [corresponding_notes[i]] * data_counts[i]

    # Define bin edges to ensure bars are of width 1
    bins = range(min(data), max(data) + 2)  # +2 to include the last edge

    # Plot the histogram
    plt.xlim(0, max(data) + 1)  # Adjust x-axis limits for better visualization
    plt.hist(data, bins=bins, align="left")  # 'align' ensures bars align with bin edges
    plt.title("Predicted Note Frequency Histogram")
    plt.xlabel("Note")
    plt.ylabel("Frequency")
    plt.show()


if __name__ == "__main__":
    main()
