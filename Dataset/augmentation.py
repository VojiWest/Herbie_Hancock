import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_circle_of_fifths(note):
    """
    Calculate the Circle of Fifths representation for a piano key index.
    Assumes note is a piano key index (e.g., 1 = A0, 40 = C4).

    Args:
        note (int): The piano key index (0 for silence).

    Returns:
        tuple: A 2D coordinate (x, y) representing the note's position on the Circle of Fifths.
    """
    chromatic_index = (note - 1) % 12

    circle_of_fifths_index = (chromatic_index * 7) % 12

    angle = circle_of_fifths_index * (2 * np.pi / 12)

    x = np.cos(angle)
    y = np.sin(angle)

    return x, y

def calculate_chromatic_circle(note):

    chromatic_index = (note - 1) % 12

    angle = chromatic_index * (2 * np.pi / 12)

    x = np.cos(angle)
    y = np.sin(angle)

    return x, y


def convert_voices_chromatic_circle(data):

    chromatic_circle_representation = []
    for note in data:
        if note == 0: 
            chromatic_circle_representation.append((0, 0))
        else:
            chromatic_circle_representation.append(calculate_chromatic_circle(note))

    return np.array(chromatic_circle_representation)


def convert_voices_circle_of_fifths(data):

    circle_of_fifths_representation = []
    for note in data:
        if note == 0: 
            circle_of_fifths_representation.append((0, 0))
        else:
            circle_of_fifths_representation.append(calculate_circle_of_fifths(note))

    return np.array(circle_of_fifths_representation)

def augmented_encoding(data, non_zero_min, max_note, max_duration):

    encoded_data = []
    duration = 0
    previous_note = None

    for note in data:
        if note == previous_note:
            duration += 1

        else:
            duration = 0 # Reset duration

        if note == 0: 
            chroma_x, chroma_y = 0, 0
            fifths_x, fifths_y = 0, 0
            
            normalized_note = -1
        else:
            chroma_x, chroma_y = calculate_chromatic_circle(note)
            fifths_x, fifths_y = calculate_circle_of_fifths(note)
                        
        normalized_note = (note - non_zero_min) / (max_note - non_zero_min)
        normalized_duration = (duration / max_duration)
        # normalized_duration = duration # For testing with normalization off
        
        encoded_data.append([normalized_note, chroma_x, chroma_y, fifths_x, fifths_y, normalized_duration])
        previous_note = note

    return np.array(encoded_data)


def plot_circle(coordinates):
    coordinates = np.array(coordinates)

    # Plot the results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], color='blue', label='Notes')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.title('Circle of Fifths Representation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
