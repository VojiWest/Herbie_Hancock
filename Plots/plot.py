import matplotlib.pyplot as plt

def plot_histogram(data, title, xlabel, ylabel):
    # get each column and plot a histogram
    for i in range(data.shape[1]):
        title = title + " Voice" + str(i + 1)
        plt.hist(data.iloc[:, i], bins=20)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

def plot_data(data, title, xlabel, ylabel):
    # data is in form of a list with each element being a list of values
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    